import scipy as sp
import numpy as np
from scipy.sparse.linalg import LinearOperator

from sksparse import __version__ as sk_sp_version
from sksparse.cholmod import cholesky as cholesky_decomp_sparse
assert sk_sp_version >= '0.4.3'
SPARSE_MODE = True

from polara import SVDModel
from polara.recommender.coldstart.models import ItemColdStartEvaluationMixin
from polara.lib.similarity import stack_features
from polara.tools.timing import track_time
from string import Template

from scaledsvd import ScaledSVD


class CholeskyFactor:
    def __init__(self, factor):
        self._factor = factor
        self._L = None
        self._transposed = False
        
    @property
    def L(self):
        if self._L is None:
            self._L = self._factor.L()
        return self._L
        
    @property
    def T(self):
        self._transposed = True
        return self
    
    def dot(self, v):
        if self._transposed:
            self._transposed = False
            return self.L.T.dot(self._factor.apply_P(v))
        else:
            return self._factor.apply_Pt(self.L.dot(v))
    
    def solve(self, y):
        x = self._factor
        if self._transposed:
            self._transposed = False
            return x.apply_Pt(x.solve_Lt(y, use_LDLt_decomposition=False))
        else:
            raise NotImplementedError

    def update_inplace(self, A, beta):
        self._factor.cholesky_inplace(A, beta=beta)
        self._L = None


class CholeskyFactorsMixin:
    def __init__(self, *args, **kwargs):
        self._sparse_mode = SPARSE_MODE
        self.return_factors = True
        
        super().__init__(*args, **kwargs)
        entities = [self.data.fields.userid, self.data.fields.itemid]
        self._cholesky  = dict.fromkeys(entities)
        
        self._features_weight = 0.999
        self.data.subscribe(self.data.on_change_event, self._clean_cholesky)

    def _clean_cholesky(self):
        self._cholesky = {entity:None for entity in self._cholesky.keys()}
        
    def _update_cholesky(self):
        for entity, cholesky in self._cholesky.items():
            if cholesky is not None:
                self._update_cholesky_inplace(entity)

    @property
    def features_weight(self):
        return self._features_weight

    @features_weight.setter
    def features_weight(self, new_val):
        if new_val != self._features_weight:
            self._features_weight = new_val
            self._update_cholesky()
            self._renew_model()

    @property
    def item_cholesky_factor(self):
        itemid = self.data.fields.itemid
        return self.get_cholesky_factor(itemid)
    
    @property
    def user_cholesky_factor(self):
        userid = self.data.fields.userid
        return self.get_cholesky_factor(userid)    
    
    def get_cholesky_factor(self, entity):
        cholesky = self._cholesky.get(entity, None)
        if cholesky is None:
            self._update_cholesky_factor(entity)
        return self._cholesky[entity]

    def _update_cholesky_factor(self, entity):
        entity_similarity = self.data.get_relations_matrix(entity)
        if entity_similarity is None:
            self._cholesky[entity] = None
        else:
            if self._sparse_mode:
                cholesky_decomp = cholesky_decomp_sparse
                mode = 'sparse'
            else:
                raise NotImplementedError
            
            weight = self.features_weight
            beta = (1.0 - weight) / weight
            if self.verbose:
                print('Performing {} Cholesky decomposition for {} similarity'.format(mode, entity))
            
            msg = Template('Cholesky decomposition computation time: $time')
            with track_time(verbose=self.verbose, message=msg):
                self._cholesky[entity] = CholeskyFactor(cholesky_decomp(entity_similarity, beta=beta))

    def _update_cholesky_inplace(self, entity):
        entity_similarity = self.data.get_relations_matrix(entity)
        if self._sparse_mode:
            weight = self.features_weight
            beta = (1.0 - weight) / weight
            if self.verbose:
                print('Updating Cholesky decomposition inplace for {} similarity'.format(entity))
            
            msg = Template('    Cholesky decomposition update time: $time')
            with track_time(verbose=self.verbose, message=msg):
                self._cholesky[entity].update_inplace(entity_similarity, beta)
        else:
            raise NotImplementedError

    def build_item_projector(self, v):
        cholesky_items = self.item_cholesky_factor
        if cholesky_items is not None:
            if self.verbose:
                print(f'Building {self.data.fields.itemid} projector for {self.method}')
            msg = Template('    Solving triangular system: $time')
            with track_time(verbose=self.verbose, message=msg):
                self.factors['items_projector_left'] = cholesky_items.T.solve(v)
            msg = Template('    Applying Cholesky factor: $time')
            with track_time(verbose=self.verbose, message=msg):
                self.factors['items_projector_right'] = cholesky_items.dot(v)

    def get_item_projector(self):
        vl = self.factors.get('items_projector_left', None)
        vr = self.factors.get('items_projector_right', None)
        return vl, vr


class HybridSVD(CholeskyFactorsMixin, SVDModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'HybridSVD'
        self.precompute_auxiliary_matrix = False

    def _check_reduced_rank(self, rank):
        super()._check_reduced_rank(rank)
        self.round_item_projector(rank)
        
    def round_item_projector(self, rank):
        vl, vr = self.get_item_projector()
        if (vl is not None) and (rank < vl.shape[1]):
            self.factors['items_projector_left'] = vl[:, :rank]
            self.factors['items_projector_right'] = vr[:, :rank]
    
    def build(self, *args, **kwargs):
        if not self._sparse_mode:
            raise(ValueError)
        
        # the order matters - trigger on_change events first 
        svd_matrix = self.get_training_matrix(dtype=np.float64)        
        cholesky_items = self.item_cholesky_factor
        cholesky_users = self.user_cholesky_factor
        
        if self.precompute_auxiliary_matrix:
            if cholesky_items is not None:
                svd_matrix = cholesky_items.T.dot(svd_matrix.T).T
                cholesky_items._L = None
            if cholesky_users is not None:
                svd_matrix = cholesky_users.T.dot(svd_matrix)
                cholesky_users._L = None
            operator = svd_matrix
        else:
            if cholesky_items is not None:
                L_item = cholesky_items
            else:
                L_item = sp.sparse.eye(svd_matrix.shape[1])
            if cholesky_users is not None:
                L_user = cholesky_users
            else:
                L_user = sp.sparse.eye(svd_matrix.shape[0])
            
            def matvec(v):
                return L_user.T.dot(svd_matrix.dot(L_item.dot(v)))
            def rmatvec(v):
                return L_item.T.dot(svd_matrix.T.dot(L_user.dot(v)))
            operator = LinearOperator(svd_matrix.shape, matvec, rmatvec)
        
        super().build(*args, operator=operator, **kwargs)
        self.build_item_projector(self.factors[self.data.fields.itemid])

    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        test_matrix, slice_data = self.get_test_matrix(test_data, shape, (start, stop))
        vl, vr = self.get_item_projector()
        scores = test_matrix.dot(vr).dot(vl.T)
        return scores, slice_data


class ScaledHybridSVD(ScaledSVD, HybridSVD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'HybridSVDs'


class HybridSVDColdStart(ItemColdStartEvaluationMixin, HybridSVD):
    def __init__(self, *args, item_features=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'HybridSVD(cs)'
        self.item_features = item_features
        self.use_raw_features = item_features is not None
    
    def build(self, *args, **kwargs):
        super().build(*args, return_factors=True, **kwargs)
    
    def get_recommendations(self):
        userid = self.data.fields.userid
        itemid = self.data.fields.itemid
                
        u = self.factors[userid]
        v = self.factors['items_projector_right']
        s = self.factors['singular_values']
        
        if self.use_raw_features:
            item_info = self.item_features.reindex(self.data.index.itemid.training.old.values,
                                                   fill_value=[])
            item_features, feature_labels = stack_features(item_info, normalize=False)
            w = item_features.T.dot(v).T
            cold_info = self.item_features.reindex(self.data.index.itemid.cold_start.old.values,
                                                   fill_value=[])
            cold_item_features, _ = stack_features(cold_info, labels=feature_labels, normalize=False)
        else:
            w = self.data.item_relathions.T.dot(v).T
            cold_item_features = self.data.cold_items_similarity
        
        wwt_inv = np.linalg.pinv(w @ w.T)
        cold_items_factors = cold_item_features.dot(w.T) @ wwt_inv
        scores = cold_items_factors @ (u * s[None, :]).T
        top_similar_users = self.get_topk_elements(scores)
        return top_similar_users


class ScaledHybridSVDColdStart(ScaledSVD, HybridSVDColdStart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'HybridSVDs(cs)'
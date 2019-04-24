"""
Python Implementation of Local Collective Embeddings

__author__ : Abhishek Thakur
__original__ : https://github.com/msaveski/LCE

"""
import math

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

from polara.recommender.models import RecommenderModel
from polara.recommender.coldstart.models import ItemColdStartEvaluationMixin
from polara.lib.similarity import stack_features
from polara.tools.timing import track_time


def tr(A, B):
    if sparse.issparse(A):
        return A.multiply(B).sum()
    return (A * B).sum()

def construct_A(X, k, binary=False):
    nbrs = NearestNeighbors(n_neighbors=1 + k).fit(X)
    if binary:
        return nbrs.kneighbors_graph(X)
    else:
        return nbrs.kneighbors_graph(X, mode='distance')


def LCE(Xs, Xu, A, k=15, alpha=0.1, beta=0.05, lamb=1, epsilon=0.0001, maxiter=15, seed=None, verbose=True):

    n = Xs.shape[0]
    v1 = Xs.shape[1]
    v2 = Xu.shape[1]
    
    random = np.random if seed is None else np.random.RandomState(seed)
    W = random.rand(n, k)
    Hs = random.rand(k, v1)
    Hu = random.rand(k, v2)

    D = sparse.dia_matrix((A.sum(axis=0), 0), A.shape)

    gamma = 1. - alpha
    trXstXs = tr(Xs, Xs)
    trXutXu = tr(Xu, Xu)

    WtW = W.T.dot(W)
    WtXs = Xs.T.dot(W).T
    WtXu = Xu.T.dot(W).T
    WtWHs = WtW.dot(Hs)
    WtWHu = WtW.dot(Hu)
    DW = D.dot(W)
    AW = A.dot(W)

    itNum = 1
    delta = 2.0 * epsilon

    ObjHist = []

    while True:

        # update H
        Hs_1 = np.divide(
            (alpha * WtXs), np.maximum(alpha * WtWHs + lamb * Hs, 1e-10))
        Hs = np.multiply(Hs, Hs_1)

        Hu_1 = np.divide(
            (gamma * WtXu), np.maximum(gamma * WtWHu + lamb * Hu, 1e-10))
        Hu = np.multiply(Hu, Hu_1)

        # update W
        W_t1 = alpha * Xs.dot(Hs.T) + gamma * Xu.dot(Hu.T) + beta * AW
        W_t2 = alpha * W.dot(Hs.dot(Hs.T)) + gamma * \
            W.dot(Hu.dot(Hu.T)) + beta * DW + lamb * W
        W_t3 = np.divide(W_t1, np.maximum(W_t2, 1e-10))
        W = np.multiply(W, W_t3)

        # calculate objective function
        WtW = W.T.dot(W)
        WtXs = Xs.T.dot(W).T
        WtXu = Xu.T.dot(W).T
        WtWHs = WtW.dot(Hs)
        WtWHu = WtW.dot(Hu)
        DW = D.dot(W)
        AW = A.dot(W)

        tr1 = alpha * (trXstXs - 2. * tr(Hs, WtXs) + tr(Hs, WtWHs))
        tr2 = gamma * (trXutXu - 2. * tr(Hu, WtXu) + tr(Hu, WtWHu))
        tr3 = beta * (tr(W, DW) - tr(W, AW))
        tr4 = lamb * (np.trace(WtW) + tr(Hs, Hs) + tr(Hu, Hu))

        Obj = tr1 + tr2 + tr3 + tr4
        ObjHist.append(Obj)

        if itNum > 1:
            delta = abs(ObjHist[-1] - ObjHist[-2])
            if verbose:
                print("Iteration: ", itNum, "Objective: ", Obj, "Delta: ", delta)
            if itNum > maxiter or delta < epsilon:
                break

        itNum += 1

    return W, Hu, Hs


class LCEModel(RecommenderModel):
    def __init__(self, *args, item_features=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rank = 10
        self.factors = {}
        self.alpha = 0.1
        self.beta = 0.05
        self.max_neighbours = 250
        self.item_features = item_features
        self.binary_features = True
        self._item_data = None
        self.feature_labels = None
        self.seed = None
        self.show_error = False
        self.regularization = 1
        self.max_iterations = 15
        self.tolerance = 0.0001
        self.method = 'LCE'
        self.data.subscribe(self.data.on_change_event, self._clean_metadata)

    def _clean_metadata(self):
        self._item_data = None
        self.feature_labels = None
        
    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, new_value):
        if new_value != self._rank:
            self._rank = new_value
            self._is_ready = False
            self._recommendations = None

    @property
    def item_data(self):
        if self.item_features is not None:
            if self._item_data is None:
                itemid = self.data.fields.itemid
                index_data = getattr(self.data.index, 'itemid')

                try:
                    item_index = index_data.training
                except AttributeError:
                    item_index = index_data
                
                self._item_data = self.item_features.reindex(item_index.old.values, # make correct sorting
                                                            fill_value=[]) 
        else:
            self._item_data = None
        return self._item_data


    def build(self):
        # prepare input matrix for learning the model
        Xs, lbls = stack_features(self.item_data, normalize=False) # item-features sparse matrix
        Xu = self.get_training_matrix().T # item-user sparse matrix

        n_nbrs = min(self.max_neighbours, int(math.sqrt(Xs.shape[0])))
        A = construct_A(Xs, n_nbrs, binary=self.binary_features)
        
        with track_time(self.training_time, verbose=self.verbose, model=self.method):
            W, Hu, Hs = LCE(Xs, Xu, A,
                            k=self.rank,
                            alpha=self.alpha,
                            beta=self.beta,
                            lamb=self.regularization,
                            epsilon=self.tolerance,
                            maxiter=self.max_iterations,
                            seed=self.seed,
                            verbose=self.show_error)
        
        userid = self.data.fields.userid
        itemid = self.data.fields.itemid
        self.factors[userid] = Hu.T
        self.factors[itemid] = W
        self.factors['item_features'] = Hs.T
        self.feature_labels = lbls
        
        
    def get_recommendations(self):
        if self.data.warm_start:
            raise NotImplementedError
        else:
            return super().get_recommendations()        

        
    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        userid = self.data.fields.userid
        itemid = self.data.fields.itemid
        slice_data = self._slice_test_data(test_data, start, stop)

        user_factors = self.factors[userid][test_users[start:stop], :]
        item_factors = self.factors[itemid]
        scores = user_factors.dot(item_factors.T)
        return scores, slice_data


class LCEModelItemColdStart(ItemColdStartEvaluationMixin, LCEModel):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'LCE(cs)'
    
    def get_recommendations(self):
        userid = self.data.fields.userid
        itemid = self.data.fields.itemid

        Hu = self.factors[userid].T
        Hs = self.factors['item_features'].T
        cold_info = self.item_features.reindex(self.data.index.itemid.cold_start.old.values,
                                               fill_value=[])
        cold_item_features, _ = stack_features(cold_info, labels=self.feature_labels, normalize=False)        
        
        cold_items_factors = cold_item_features.dot(Hs.T).dot(np.linalg.pinv(Hs @ Hs.T))
        cold_items_factors[cold_items_factors < 0] = 0
        
        scores = cold_items_factors @ Hu
        top_relevant_users = self.get_topk_elements(scores).astype(np.intp)
        return top_relevant_users        
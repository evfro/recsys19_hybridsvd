from numpy import power
from scipy.sparse import diags
from scipy.sparse.linalg import norm as spnorm
from polara import SVDModel
from polara.recommender.coldstart.models import SVDModelItemColdStart


def rescale_matrix(matrix, scaling, axis, binary=False):
    '''Function to scale either rows or columns of the sparse rating matrix'''
    if scaling == 1: # no scaling (standard SVD case)
        return matrix
    
    if binary:
        norm = np.sqrt(matrix.getnnz(axis=axis)) # compute Euclidean norm as if values are binary
    else:
        norm = spnorm(matrix, axis=axis, ord=2) # compute Euclidean norm

    scaling_matrix = diags(power(norm, scaling-1, where=norm!=0))
    
    if axis == 0: # scale columns
        return matrix.dot(scaling_matrix)
    if axis == 1: # scale rows
        return scaling_matrix.dot(matrix)


class ScaledSVD(SVDModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.row_scaling = 1
        self.col_scaling = 1
        self.method = 'PureSVDs'

    def get_training_matrix(self, *args, **kwargs):
        svd_matrix = super().get_training_matrix(*args, **kwargs)
        svd_matrix = rescale_matrix(svd_matrix, self.row_scaling, 1)
        svd_matrix = rescale_matrix(svd_matrix, self.col_scaling, 0)
        return svd_matrix


class ScaledSVDItemColdStart(ScaledSVD, SVDModelItemColdStart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'PureSVDs(cs)'
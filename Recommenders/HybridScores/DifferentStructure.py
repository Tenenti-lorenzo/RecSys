from numpy import linalg as LA
import scipy.sparse as sps
from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.BaseRecommender import BaseRecommender

class TwoDifferentModelRecommender12(BaseRecommender):
    '''
    Hybrid of 2 predictions
    '''

    '''
    hyperparameters_range_dictionary = {
        "norm" : Categorical([1, 2, np.inf, -np.inf]),
        "alpha" :Real(low = 0.0001, high = 0.9999, prior = 'uniform')
    }
    '''

    RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"


    def __init__(self, URM_train, recommender_1, recommender_2):
        super(TwoDifferentModelRecommender12, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        
        
        
    def fit(self, norm = None, alpha = 0.5):

        self.alpha = alpha
        self.norm = norm


    def _compute_item_score(self, user_id_array,items_to_compute = None):
        
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        norm_item_weights_1 = 1 if self.norm is None else LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = 1 if self.norm is None else LA.norm(item_weights_2, self.norm)
        
        
        if norm_item_weights_1 == 0:
            raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
        
        if norm_item_weights_2 == 0:
            raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
        
        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (1-self.alpha)

        return item_weights

class TwoDifferentModelRecommender(BaseRecommender):
    '''
    Hybrid of 2 predictions
    '''

    '''
    hyperparameters_range_dictionary = {
        "norm" : Categorical([1, 2, np.inf, -np.inf]),
        "alpha" :Real(low = 0.0001, high = 0.9999, prior = 'uniform')
    }
    '''

    RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"


    def __init__(self, URM_train, recommender_1, recommender_2):
        super(TwoDifferentModelRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        
        
        
    def fit(self, norm = None, alpha = 0.5):

        self.alpha = alpha
        self.norm = norm


    def _compute_item_score(self, user_id_array,items_to_compute = None):
        
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        norm_item_weights_1 = 1 if self.norm is None else LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = 1 if self.norm is None else LA.norm(item_weights_2, self.norm)
        
        
        if norm_item_weights_1 == 0:
            raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
        
        if norm_item_weights_2 == 0:
            raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
        
        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (1-self.alpha)

        return item_weights

class ThreeDifferentModelRecommender(BaseRecommender):
    '''
    Hybrid of 3 predictions
    '''

    '''
    hyperparameters_range_dictionary = {
        "norm" : Categorical([1, 2, np.inf, -np.inf]),
        "alpha" :Real(low = 0, high = 1, prior = 'uniform'),
        "beta" :Real(low = 0, high = 1, prior = 'uniform'),
        "gamma" :Real(low = 0, high = 1, prior = 'uniform')
    }
    '''

    RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"


    def __init__(self, URM_train, recommender_1, recommender_2, recommender_3):
        super(ThreeDifferentModelRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3
        
        
        
    def fit(self, norm,norm2, alpha = 0.5, beta = 0.5, gamma = 0.5):

        sump = alpha+beta+gamma

        self.alpha = alpha/sump
        self.beta = beta/sump
        self.gamma = gamma/sump
        self.norm = norm
        self.norm = norm2

        print(f"CURRENT CONFIGURATION:\n{self.recommender_1.RECOMMENDER_NAME} with weight alpha: {self.alpha}")
        print(f"{self.recommender_2.RECOMMENDER_NAME} with weight beta: {self.beta}")
        print(f"{self.recommender_3.RECOMMENDER_NAME} with weight gamma: {self.gamma}")
        print(f"Norm type: {self.norm}")


    def _compute_item_score(self, user_id_array,items_to_compute = None):
        
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.recommender_3._compute_item_score(user_id_array)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm) if self.norm is not None else 1
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm) if self.norm is not None else 1
        norm_item_weights_3 = LA.norm(item_weights_3, self.norm) if self.norm is not None else 1

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm2) if self.norm is 2 and self.norm is 1 else LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm2) if self.norm is 2 and self.norm is 1 else LA.norm(item_weights_2, self.norm)
        norm_item_weights_3 = LA.norm(item_weights_3, self.norm2) if self.norm is 2 and self.norm is 1 else LA.norm(item_weights_3, self.norm)
        
        
        if norm_item_weights_1 == 0:
            raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
        
        if norm_item_weights_2 == 0:
            raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
            
        if norm_item_weights_3 == 0:
            raise ValueError("Norm {} of item weights for recommender 3 is zero. Avoiding division by zero".format(self.norm))
        
        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * self.beta + item_weights_3 / norm_item_weights_3 * self.gamma

        return item_weights

    def save_model(self, folder_path, file_name = None):
        print("Not saving")

class ThreeDifferentModelRecommender1(BaseRecommender):
    '''
    Hybrid of 3 predictions
    '''

    '''
    hyperparameters_range_dictionary = {
        "norm" : Categorical([1, 2, np.inf, -np.inf]),
        "alpha" :Real(low = 0, high = 1, prior = 'uniform'),
        "beta" :Real(low = 0, high = 1, prior = 'uniform'),
        "gamma" :Real(low = 0, high = 1, prior = 'uniform')
    }
    '''

    RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"


    def __init__(self, URM_train, recommender_1, recommender_2, recommender_3):
        super(ThreeDifferentModelRecommender1, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3
        
        
        
    def fit(self, norm,norm2, alpha = 0.5, beta = 0.5, gamma = 0.5):

        sump = alpha+beta+gamma

        self.alpha = alpha/sump
        self.beta = beta/sump
        self.gamma = gamma/sump
        self.norm = norm
        self.norm = norm2

        print(f"CURRENT CONFIGURATION:\n{self.recommender_1.RECOMMENDER_NAME} with weight alpha: {self.alpha}")
        print(f"{self.recommender_2.RECOMMENDER_NAME} with weight beta: {self.beta}")
        print(f"{self.recommender_3.RECOMMENDER_NAME} with weight gamma: {self.gamma}")
        print(f"Norm type: {self.norm}")


    def _compute_item_score(self, user_id_array,items_to_compute = None):
        
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.recommender_3._compute_item_score(user_id_array)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm) if self.norm is not None else 1
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm) if self.norm is not None else 1
        norm_item_weights_3 = LA.norm(item_weights_3, self.norm) if self.norm is not None else 1

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm2) if self.norm is 2 and self.norm is 1 else LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm2) if self.norm is 2 and self.norm is 1 else LA.norm(item_weights_2, self.norm)
        norm_item_weights_3 = LA.norm(item_weights_3, self.norm2) if self.norm is 2 and self.norm is 1 else LA.norm(item_weights_3, self.norm)
        
        
        if norm_item_weights_1 == 0:
            raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
        
        if norm_item_weights_2 == 0:
            raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
            
        if norm_item_weights_3 == 0:
            raise ValueError("Norm {} of item weights for recommender 3 is zero. Avoiding division by zero".format(self.norm))
        
        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * self.beta + item_weights_3 / norm_item_weights_3 * self.gamma

        return item_weights

    def save_model(self, folder_path, file_name = None):
        print("Not saving")

class ThreeDifferentModelRecommender12(BaseRecommender):
    '''
    Hybrid of 3 predictions
    '''

    '''
    hyperparameters_range_dictionary = {
        "norm" : Categorical([1, 2, np.inf, -np.inf]),
        "alpha" :Real(low = 0, high = 1, prior = 'uniform'),
        "beta" :Real(low = 0, high = 1, prior = 'uniform'),
        "gamma" :Real(low = 0, high = 1, prior = 'uniform')
    }
    '''

    RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"


    def __init__(self, URM_train, recommender_1, recommender_2, recommender_3):
        super(ThreeDifferentModelRecommender12, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3
        
        
        
    def fit(self, norm, alpha = 0.5, beta = 0.5, gamma = 0.5):

        sump = alpha+beta+gamma

        self.alpha = alpha/sump
        self.beta = beta/sump
        self.gamma = gamma/sump
        self.norm = norm

        print(f"CURRENT CONFIGURATION:\n{self.recommender_1.RECOMMENDER_NAME} with weight alpha: {self.alpha}")
        print(f"{self.recommender_2.RECOMMENDER_NAME} with weight beta: {self.beta}")
        print(f"{self.recommender_3.RECOMMENDER_NAME} with weight gamma: {self.gamma}")
        print(f"Norm type: {self.norm}")


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.recommender_3._compute_item_score(user_id_array)

        norm_item_weights_1 = 1 if self.norm is None else LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = 1 if self.norm is None else LA.norm(item_weights_2, self.norm)
        norm_item_weights_3 = 1 if self.norm is None else LA.norm(item_weights_3, self.norm)
        
        
        if norm_item_weights_1 == 0:
            raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
        
        if norm_item_weights_2 == 0:
            raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
            
        if norm_item_weights_3 == 0:
            raise ValueError("Norm {} of item weights for recommender 3 is zero. Avoiding division by zero".format(self.norm))
        
        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * self.beta + item_weights_3 / norm_item_weights_3 * self.gamma

        return item_weights

    def save_model(self, folder_path, file_name = None):
        print("Not saving")
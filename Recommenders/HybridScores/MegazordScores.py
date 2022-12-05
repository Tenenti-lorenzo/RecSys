from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sp
from numpy import linalg as LA



class MegazordScoresHybrid(BaseRecommender):
    """ MegazordScoresHybrid
    Hybrid of two prediction scores R = R1/norm*alpha + R2/norm*(1-alpha) where R1 and R2 come from
    algorithms trained on different loss functions.

    """

    RECOMMENDER_NAME = "MegazordScoresHybrid"


    def __init__(self, URM_train, recommender_1, recommender_2):
        super(MegazordScoresHybrid, self).__init__(URM_train)

        self.URM_train = sp.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        
        
        
    def fit(self, norm, alpha = 0.5, normalize = True):

        self.alpha = alpha
        self.norm = norm
        self.normalize = normalize


    def _compute_item_score(self, user_id_array, items_to_compute):
        
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm)
        
        
        if norm_item_weights_1 == 0:
            raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
        
        if norm_item_weights_2 == 0:
            raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
            
        
        if self.normalize:
            item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (1-self.alpha)
        else:
            item_weights = item_weights_1 * self.alpha + item_weights_2 * (1-self.alpha)

        return item_weights


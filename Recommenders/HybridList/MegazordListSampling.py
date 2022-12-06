import numpy as np
import scipy.sparse as sp
from Recommenders.BaseRecommender import BaseRecommender

class MegazordListSampling(BaseRecommender):
    """ MegazordListSampling 
    Hybrid of two recommenders using sampling of the recommandation lists of them

    """

    RECOMMENDER_NAME = "MegazordListSampling"


    def __init__(self, URM_train, recommender_1, recommender_2):
        super(MegazordListSampling, self).__init__(URM_train)

        self.URM_train = sp.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        
        
    def fit(self,prob):

        self.prob = prob #probability of choosing the second list


    
    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):


        if np.isscalar(user_id_array):
            list1 = self.recommender_1.recommend(user_id_array, cutoff = cutoff, remove_seen_flag=remove_seen_flag, items_to_compute = items_to_compute,
                  remove_top_pop_flag = remove_top_pop_flag, remove_custom_items_flag = remove_custom_items_flag, return_scores = return_scores)
            list2 = self.recommender_2.recommend(user_id_array, cutoff = cutoff, remove_seen_flag=remove_seen_flag, items_to_compute = items_to_compute,
                  remove_top_pop_flag = remove_top_pop_flag, remove_custom_items_flag = remove_custom_items_flag, return_scores = return_scores)
        else:
            list1,score1 = self.recommender_1.recommend(user_id_array, cutoff = cutoff, remove_seen_flag=remove_seen_flag, items_to_compute = items_to_compute,
                    remove_top_pop_flag = remove_top_pop_flag, remove_custom_items_flag = remove_custom_items_flag, return_scores = return_scores)
            list2,_ = self.recommender_2.recommend(user_id_array, cutoff = cutoff, remove_seen_flag=remove_seen_flag, items_to_compute = items_to_compute,
                    remove_top_pop_flag = remove_top_pop_flag, remove_custom_items_flag = remove_custom_items_flag, return_scores = return_scores)
            
        final_list = []
        
        
        for i in range(len(user_id_array)):
            final_sub_list = []
            iterator_on_list1 = 0
            iterator_on_list2 = 0
            sub_list1 = list1[i]
            sub_list2 = list2[i]
            

            while len(final_sub_list) < cutoff:
                rand = np.random.binomial(1, self.prob, 1)[0]
                if (rand == 1):
                    if sub_list2[iterator_on_list2] not in final_sub_list:
                        final_sub_list.append(sub_list2[iterator_on_list2])


                    #we should increase the iterator anyway, or will always be stucked in the element 
                    #already present in the final list
                    iterator_on_list2 += 1
                    iterator_on_list2 = min(iterator_on_list2, cutoff-1) #list ha max lenght cutoff
                    
                else:
                    if sub_list1[iterator_on_list1] not in final_sub_list:
                        final_sub_list.append(sub_list1[iterator_on_list1])

                    iterator_on_list1 += 1
                    iterator_on_list1 = min(iterator_on_list1, cutoff-1) #list ha max lenght cutoff
                    
            
            
            final_list.append(final_sub_list)
        
        if np.isscalar(user_id_array):
            return final_list
        else:
            return final_list,score1 # we don't care about the scores, we'll just pass the scores of the first recommender
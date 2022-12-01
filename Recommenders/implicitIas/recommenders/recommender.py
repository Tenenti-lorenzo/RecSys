import numpy as np
import scipy.sparse as sp
from Recommenders.implicitIas.utils.official.Recommender_utils import check_matrix


class Recommender:

    def __init__(self, URM_train: sp.csr_matrix, ICM, exclude_seen=True):
        if not sp.isspmatrix_csr(URM_train):
            raise TypeError(f"We expected a CSR matrix, we got {type(URM_train)}")
        self.URM_train = URM_train.copy()
        self.ICM = ICM.copy()
        self.predicted_URM_train = None
        self.exclude_seen = exclude_seen
        self.recommendations = None

    def fit(self):
        """
        Performs fitting and training of the recommender
        Prepares the predicted_URM_train matrix
        All needed parameters must be passed through init
        :return: Nothing
        """
        raise NotImplementedError()

    # def recommend(self, user_id, cutoff=10):
    #     """
    #     Provides a list of 'at' recommended items for the given user
    #     :param user_id: id for which provide recommendation
    #     :param at: how many items have to be recommended
    #     :return: recommended items list
    #     """

    #     predicted_ratings = self.compute_predicted_ratings(user_id)

    #     if self.exclude_seen:
    #         predicted_ratings = self.__filter_seen(user_id, predicted_ratings)

    #     # ideally do
    #     # ordered_items = np.flip(np.argsort(predicted_ratings))
    #     # recommended_items = ordered_items[:cutoff]
    #     # return recommended_items

    #     # BUT O(NlogN) -> MORE EFFICIENT O(N+KlogK)

    #     # top k indices in sparse order
    #     ind = np.argpartition(predicted_ratings, -cutoff)[-cutoff:]
    #     # support needed to correctly index
    #     f = np.flip(np.argsort(predicted_ratings[ind]))
    #     # assert((predicted_ratings[recommended_items] == predicted_ratings[ind[f]]).all())
    #     return ind[f]
    def _remove_seen_on_scores(self, user_id, scores):
    
        assert self.URM_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores
        
    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        cutoff = min(cutoff, self.URM_train.shape[1] - 1)

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            if remove_seen_flag:
                scores_batch[user_index,:] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
            # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
            # ranking = relevant_items_partition[relevant_items_partition_sorting]
            #
            # ranking_list.append(ranking)


        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_custom_items_flag:
            scores_batch = self._remove_custom_items_on_scores(scores_batch)

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:,0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = [None] * ranking.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(len(user_id_array)):
            user_recommendation_list = ranking[user_index]
            user_item_scores = scores_batch[user_index, user_recommendation_list]

            not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

            user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
            ranking_list[user_index] = user_recommendation_list.tolist()



        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]


        if return_scores:
            return ranking_list, scores_batch

        else:
            return ranking_list
    def compute_predicted_ratings(self, user_id):

        """ Compute the predicted ratings for a given user_id """

        return self.predicted_URM_train[user_id].toarray().ravel()
    def _remove_custom_items_on_scores(self, scores_batch):
        scores_batch[:, self.items_to_ignore_ID] = -np.inf
        return scores_batch
    def _remove_TopPop_on_scores(self, scores_batch):
        scores_batch[:, self.filterTopPop_ItemsID] = -np.inf
        return scores_batch
    def __filter_seen(self, user_id, predicted_ratings):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        predicted_ratings[user_profile] = -np.inf

        return predicted_ratings

    def compute_predicted_ratings_top_k(self, user_id, k):
        predicted_ratings = self.compute_predicted_ratings(user_id)

        if self.exclude_seen:
            predicted_ratings = self.__filter_seen(user_id, predicted_ratings)

        # top k indices in sparse order
        mask = np.argpartition(predicted_ratings, -k)[-k:]

        return predicted_ratings[mask], mask

    def add_side_information(self, beta):

        self.URM_train = self.URM_train.copy()
        self._stack(self.ICM.T, beta)

    def get_URM_train(self):
        return self.URM_train.copy()

    def _stack(self, to_stack, param, format='csr'):

        """
        Stacks a new sparse matrix under the A matrix used for training
        :param to_stack: sparse matrix to add
        :param param: regularization
        :param format: default 'csr'
        """

        tmp = check_matrix(to_stack, 'csr', dtype=np.float32)
        tmp = tmp.multiply(param)
        self.URM_train = sp.vstack((self.URM_train, tmp), format=format, dtype=np.float32)

from Recommenders.DataIO import DataIO
class MatrixFactorizationRecommender(Recommender):
    """ ABSTRACT MATRIX FACTORIZATION RECOMMENDER """

    def __init__(self, URM_train: sp.csr_matrix, ICM, exclude_seen=True):
        super().__init__(URM_train, ICM, exclude_seen)
        self.user_factors = None  # playlist x latent_factors
        self.item_factors = None  # tracks x latent_factors

    def compute_predicted_ratings(self, user_id):
        """ Compute predicted ratings for a given playlist in case of
        matrix factorization algorithm """

        return np.dot(self.user_factors[user_id], self.item_factors.T)

    def fit(self):
        raise NotImplementedError()
    
    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """
        user_factors is n_users x n_factors
        item_factors is n_items x n_factors

        The prediction for cold users will always be -inf for ALL items

        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        assert self.user_factors.shape[1] == self.item_factors.shape[1], \
            "{}: User and Item factors have inconsistent shape".format(self.RECOMMENDER_NAME)

        assert self.user_factors.shape[0] > np.max(user_id_array),\
                "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.RECOMMENDER_NAME, self.user_factors.shape[0], np.max(user_id_array))

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.item_factors.shape[0]), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = np.dot(self.user_factors[user_id_array], self.item_factors[items_to_compute,:].T)

        else:
            item_scores = np.dot(self.user_factors[user_id_array], self.item_factors.T)


        # No need to select only the specific negative items or warm users because the -inf score will not change
        if self.use_bias:
            item_scores += self.ITEM_bias + self.GLOBAL_bias
            item_scores = (item_scores.T + self.USER_bias[user_id_array]).T

        return item_scores

    def save_model(self, folder_path, file_name = None):
    
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"user_factors": self.user_factors,
                              "item_factors": self.item_factors,
                              "use_bias": self.use_bias,
                            }

        if self.use_bias:
            data_dict_to_save["ITEM_bias"] = self.ITEM_bias
            data_dict_to_save["USER_bias"] = self.USER_bias
            data_dict_to_save["GLOBAL_bias"] = self.GLOBAL_bias

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)


        self._print("Saving complete")

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

    def recommend(self, user_id, at=10):
        """
        Provides a list of 'at' recommended items for the given user
        :param user_id: id for which provide recommendation
        :param at: how many items have to be recommended
        :return: recommended items list
        """

        predicted_ratings = self.compute_predicted_ratings(user_id)

        if self.exclude_seen:
            predicted_ratings = self.__filter_seen(user_id, predicted_ratings)

        # ideally do
        # ordered_items = np.flip(np.argsort(predicted_ratings))
        # recommended_items = ordered_items[:at]
        # return recommended_items

        # BUT O(NlogN) -> MORE EFFICIENT O(N+KlogK)

        # top k indices in sparse order
        ind = np.argpartition(predicted_ratings, -at)[-at:]
        # support needed to correctly index
        f = np.flip(np.argsort(predicted_ratings[ind]))
        # assert((predicted_ratings[recommended_items] == predicted_ratings[ind[f]]).all())
        return ind[f]

    def compute_predicted_ratings(self, user_id):

        """ Compute the predicted ratings for a given user_id """

        return self.predicted_URM_train[user_id].toarray().ravel()

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
        return self.URM_train_train.copy()

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

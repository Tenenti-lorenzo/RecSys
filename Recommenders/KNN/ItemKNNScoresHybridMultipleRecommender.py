#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/11/20

@author: GitMasters
"""

from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from numpy import linalg as LA
import numpy as np
import numpy.ma as ma
import scipy.sparse as sps
import pandas as pd



def build_URM_ICM(dataset_, dataset_type_, dataset_lenght_, implicit=True,data_weight=1):
    df      = dataset_.copy()
    df_l    = dataset_lenght_.copy()
    df_t    = dataset_type_.copy()

    #Delete Data all one
    df_t.drop(columns=["data"],inplace = True)
    #build five columns 1 if that item has this feature
    df_t = pd.get_dummies(df_t.feature_id)
    df_t.columns = ["feature_0","feature_1","feature_2","feature_3","feature_4"]
    df_t["item_id"]= dataset_type_["item_id"]
    # Add to df_t data of length
    df_t["lenght"] = df_l["data"]
    
    # In union there are only unique item between data_ICM e interactions_and_impressions
    union=np.union1d(df["item_id"].unique(),df_t["item_id"].unique())
    # Item to set all feature to 0
    sup = np.setdiff1d(union,df_t["item_id"].unique())
    support = pd.DataFrame(columns=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4','item_id', 'lenght'])
    support.item_id = sup
    column_fake=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'lenght']
    for col in column_fake:
        support[col]=0
    #Add dummies item
    ICM_All = pd.concat([support,df])
    ICM_All = ICM_All.sort_values(by=['item_id'])
    ICM_All.reset_index().drop(columns=['index',"item_id"],inplace = True)
    ICM_All = sps.csr_matrix(ICM_All.astype(pd.SparseDtype("int64",0)).sparse.to_coo())

    if(implicit):
        df = df.groupby(by=['user_id','item_id']).count().reset_index()
        df['data'] = 1
        

        support = pd.DataFrame(columns=["user_id","item_id","data"])
        support.item_id = sup
        support.user_id=0
        support.data=0
        df = pd.concat([sup,df])
        df = df.reset_index()

        unique_users = df.user_id.unique()
        unique_items = df.item_id.unique()

        num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
        num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

        URM_ALL = sps.csr_matrix((df.data, (df.user_id, df.item_id)), 
                                shape=(num_users, num_items))
    else:

        #TODO
        dataset['Data'] = dataset['Data'].replace({0:data_weight})
        dataset = dataset.groupby(by=['UserID','ItemID']).sum('Data').reset_index()

        URM_ALL = sps.csr_matrix((dataset.Data, (dataset.UserID, dataset.ItemID)), 
                                shape=(num_users, num_items))


    return URM_ALL,ICM_All


def build_URM(dataset_,implicit=True,data_weight=1):
    #           Args 
    #           Implicit = True item is one if it watch or not watch
    #           data_weight weight of data for esplicit URM

    dataset = dataset_.copy()
    columns_name = ['user_id','item_id','impression_list','data']
    dataset.columns = columns_name
    dataset.drop(columns=['impression_list'], inplace=True)

    if(implicit):
        #Delete impression_list
        dataset = dataset.groupby(by=['user_id','item_id']).count().reset_index()
        dataset['data'] = 1
        unique_users = dataset.user_id.unique()
        unique_items = dataset.item_id.unique()

        num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
        num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

        URM_ALL = sps.csr_matrix((dataset.data, (dataset.user_id, dataset.item_id)), 
                                shape=(num_users, num_items))
    else:
        #impr weight, dataset type e lenght non usate per ora
        dataset['Data'] = dataset['Data'].replace({0:data_weight})
        dataset = dataset.groupby(by=['UserID','ItemID']).sum('Data').reset_index()

        URM_ALL = sps.csr_matrix((dataset.Data, (dataset.UserID, dataset.ItemID)), 
                                shape=(num_users, num_items))

    return URM_ALL

class ItemKNNCFWrapper(ItemKNNCFRecommender):
    def __init__(self,dataset, dataset_type, dataset_lenght, URM_train, verbose = True):
        super(ItemKNNCFWrapper, self).__init__(URM_train, verbose = verbose)
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.dataset_lenght = dataset_lenght

    
    
    def fit(self, num_users, num_items, data_weight, impr_weight, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", 
            URM_bias = False, **similarity_args):
        
        self.URM_train = build_URM(self.dataset, self.dataset_type, self.dataset_lenght, data_weight, impr_weight, num_users = num_users, num_items = num_items)
        
       

        super(ItemKNNCFWrapper, self).fit(topK=topK, shrink=shrink, similarity=similarity, normalize=normalize, feature_weighting =feature_weighting, 
            URM_bias = URM_bias, **similarity_args)



class ItemKNNScoresHybridMultipleRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*beta + R3*(1-alpha-beta)

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridMultipleRecommender"


    def __init__(self, URM_train, Recommender_1, Recommender_2, Recommender_3, verbose = True):
        super(ItemKNNScoresHybridMultipleRecommender, self).__init__(URM_train, verbose = verbose)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        self.Recommender_3 = Recommender_3
        
        
    def fit(self, alpha = 0.5, beta = 0.5, gamma = 0.5):

        sump = alpha + beta + gamma

        self.alpha = alpha/sump
        self.beta = beta/sump
        self.gamma = gamma/sump

        
        print(f"CURRENT CONFIGURATION:\n{self.Recommender_1.RECOMMENDER_NAME} with weight alpha: {self.alpha}")
        print(f"{self.Recommender_2.RECOMMENDER_NAME} with weight beta: {self.beta}")
        print(f"{self.Recommender_3.RECOMMENDER_NAME} with weight gamma: {self.gamma}")
        


    def _compute_item_score(self, user_id_array = None, items_to_compute = None):
        
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.Recommender_3._compute_item_score(user_id_array)

        l2_1 = LA.norm(item_weights_1, 2)
        l2_1_scores = item_weights_1 / l2_1

        l2_2 = LA.norm(item_weights_2, 2)
        l2_2_scores = item_weights_2 / l2_2

        l2_3 = LA.norm(item_weights_3, 2)
        l2_3_scores = item_weights_3 / l2_3

        item_weights = item_weights_1*self.alpha + item_weights_2*self.beta + item_weights_3*self.gamma

        return item_weights

class ItemKNNScoresHybridTwoRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridTwoRecommender"


    def __init__(self, URM_train, Recommender_1, Recommender_2, verbose = True):
        super(ItemKNNScoresHybridTwoRecommender, self).__init__(URM_train, verbose = verbose)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        
        
        
    def fit(self, alpha = 0.5):

        self.alpha = alpha

        

        '''
        print(f"CURRENT CONFIGURATION:\n{self.Recommender_1.RECOMMENDER_NAME} with weight alpha: {self.alpha}")
        print(f"{self.Recommender_2.RECOMMENDER_NAME} with weight beta: {1 - self.alpha}")
        '''

    def _compute_item_score(self, user_id_array = None, items_to_compute = None):
        
        
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)

        l2_1 = LA.norm(item_weights_1, 2)
        l2_1_scores = item_weights_1 / l2_1

        l2_2 = LA.norm(item_weights_2, 2)
        l2_2_scores = item_weights_2 / l2_2
        

        item_weights = l2_1_scores*self.alpha + l2_2_scores*(1 - self.alpha)

        return item_weights


class ItemKNNScoresHybridTwoRecommender_FAST(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridTwoRecommender_FAST"


    def __init__(self, URM_train, Recommender_1, Recommender_2, verbose = True):
        super(ItemKNNScoresHybridTwoRecommender_FAST, self).__init__(URM_train, verbose = verbose)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        
        
        
    def fit(self, alpha = 0.5):

        self.alpha = alpha

        

        '''
        print(f"CURRENT CONFIGURATION:\n{self.Recommender_1.RECOMMENDER_NAME} with weight alpha: {self.alpha}")
        print(f"{self.Recommender_2.RECOMMENDER_NAME} with weight beta: {1 - self.alpha}")
        '''

    def _compute_item_score(self, user_id_array = None, items_to_compute = None):
        
        
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)

        item_weights = item_weights_1*self.alpha + item_weights_2*(1 - self.alpha)

        return item_weights


class ItemKNNScoresHybridTwoRecommender_PRELOAD(BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "ItemKNNScoresHybridTwoRecommender_PRELOAD"


    def __init__(self, URM_train, Recommender_1, Recommender_2, verbose = True):
        super(ItemKNNScoresHybridTwoRecommender_PRELOAD, self).__init__(URM_train, verbose = verbose)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
    
        n_users = self.URM_train.shape[0]

        item_weights_1 = Recommender_1._compute_item_score(np.arange(n_users))
        item_weights_2 = Recommender_2._compute_item_score(np.arange(n_users))

        th1 = item_weights_1.mean()*0.2
        th2 = item_weights_2.mean()*0.2
        masked1 = ma.array(item_weights_1, mask = item_weights_1<th1).filled(fill_value = 0)
        masked2 = ma.array(item_weights_2, mask = item_weights_2<th2).filled(fill_value = 0)

        mas1 = sps.csr_matrix(masked1)
        mas2 = sps.csr_matrix(masked2)

        print("Starting init on matrix with shape: " + str(item_weights_1.shape))

        l2_1 = sps.linalg.norm(mas1, ord=1)
        self.l2_1_scores = mas1 / l2_1

        l2_2 = sps.linalg.norm(mas2, ord=1)
        self.l2_2_scores = mas2 / l2_2
        
        print("Completed init")
        
        
    def fit(self, alpha = 0.5):

        self.alpha = alpha

        

        '''
        print(f"CURRENT CONFIGURATION:\n{self.Recommender_1.RECOMMENDER_NAME} with weight alpha: {self.alpha}")
        print(f"{self.Recommender_2.RECOMMENDER_NAME} with weight beta: {1 - self.alpha}")
        '''

    def _compute_item_score(self, user_id_array = None, items_to_compute = None):
        

        item_weights = self.l2_1_scores[user_id_array, items_to_compute]*self.alpha + self.l2_2_scores[user_id_array, items_to_compute]*(1 - self.alpha)

        return item_weights

class ItemKNNScoresHybridTwoRecommender_PRUNE(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridTwoRecommender"


    def __init__(self, URM_train, Recommender_1, Recommender_2, verbose = True):
        super(ItemKNNScoresHybridTwoRecommender_PRUNE, self).__init__(URM_train, verbose = verbose)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        
        
        
    def fit(self, alpha = 0.5):

        self.alpha = alpha

        

        '''
        print(f"CURRENT CONFIGURATION:\n{self.Recommender_1.RECOMMENDER_NAME} with weight alpha: {self.alpha}")
        print(f"{self.Recommender_2.RECOMMENDER_NAME} with weight beta: {1 - self.alpha}")
        '''

    def _compute_item_score(self, user_id_array = None, items_to_compute = None):
        
        
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)

        th1 = item_weights_1.mean()*0.2
        th2 = item_weights_2.mean()*0.2
        masked1 = ma.array(item_weights_1, mask = item_weights_1<th1).filled(fill_value = 0)
        masked2 = ma.array(item_weights_2, mask = item_weights_2<th2).filled(fill_value = 0)

        l2_1 = LA.norm(masked1, 2)
        l2_1_scores = masked1 / l2_1

        l2_2 = LA.norm(masked2, 2)
        l2_2_scores = masked2 / l2_2
        

        item_weights = l2_1_scores*self.alpha + l2_2_scores*(1 - self.alpha)

        return item_weights

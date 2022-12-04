import numpy as np
import numpy.ma as ma
import scipy.sparse as sps
import pandas as pd
from Evaluation.Evaluator import EvaluatorHoldout



def build_URM_ICM(dataset_, dataset_type_, dataset_lenght_, implicit=True,data_weight=1):
    df      = dataset_.copy()
    columns_name = ['user_id','item_id','impression_list','data']
    df.columns = columns_name
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
    ICM_All = pd.concat([support,df_t])
    ICM_All = ICM_All.sort_values(by=['item_id'])
    ICM_All=ICM_All.reset_index()
    ICM_All.drop(columns=['index',"item_id"],inplace = True)
    n_items, n_features=ICM_All.shape
    ICM_All = sps.csr_matrix(ICM_All, 
                                shape=(n_items, n_features)).tocoo()

    if(implicit):
        df = df.groupby(by=['user_id','item_id']).count().reset_index()
        df['data'] = 1
        
        sup = np.setdiff1d(union,df["item_id"].unique())
        support = pd.DataFrame(columns=["user_id","item_id","data"])
        support.item_id = sup
        support.user_id=0
        support.data=0
        df = pd.concat([support,df])
        df = df.reset_index()

        unique_users = df.user_id.unique()
        unique_items = df.item_id.unique()

        num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
        num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

        URM_ALL = sps.csr_matrix((df.data, (df.user_id, df.item_id)), 
                                shape=(num_users, num_items))
    else:

        
        dataset['Data'] = dataset['Data'].replace({0:data_weight})
        dataset = dataset.groupby(by=['UserID','ItemID']).sum('Data').reset_index()
        support = pd.DataFrame(columns=["user_id","item_id","data"])
        support.item_id = sup
        support.user_id=0
        support.data=0
        df = pd.concat([support,df])
        df = df.reset_index()

        unique_users = df.user_id.unique()
        unique_items = df.item_id.unique()

        num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
        num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

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

        unique_users = dataset.user_id.unique()
        unique_items = dataset.item_id.unique()

        num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
        num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

        URM_ALL = sps.csr_matrix((dataset.Data, (dataset.UserID, dataset.ItemID)), 
                                shape=(num_users, num_items))

    return URM_ALL

def userwise(num_group,cutoff = 10,recommender_object_dict={},URM_train,URM_test):
    profile_length = np.ediff1d(sps.csr_matrix(URM_train).indptr)
    block_size = int(len(profile_length)/num_group)
    sorted_users = np.argsort(profile_length)
    MAP_recommender_per_group = {}
    for group_id in range(0, num_group):
        
        start_pos = group_id*block_size
        end_pos = min((group_id+1)*block_size, len(profile_length))
        
        users_in_group = sorted_users[start_pos:end_pos]
        
        users_in_group_p_len = profile_length[users_in_group]
        
        print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
            group_id, 
            users_in_group.shape[0],
            users_in_group_p_len.mean(),
            np.median(users_in_group_p_len),
            users_in_group_p_len.min(),
            users_in_group_p_len.max()))
        
        
        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group = sorted_users[users_not_in_group_flag]
        
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)
        
        for label, recommender in recommender_object_dict.items():
            result_df, _ = evaluator_test.evaluateRecommender(recommender)
            if label in MAP_recommender_per_group:
                MAP_recommender_per_group[label].append(result_df.loc[cutoff]["MAP"])
            else:
                MAP_recommender_per_group[label] = [result_df.loc[cutoff]["MAP"]]

        return MAP_recommender_per_group
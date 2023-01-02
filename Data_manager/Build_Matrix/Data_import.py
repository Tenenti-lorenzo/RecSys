import numpy as np
import numpy.ma as ma
import scipy.sparse as sps
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Evaluation.Evaluator import EvaluatorHoldout

#useless comment

def build_URM_impression(df1 : pd.DataFrame, num_users, num_items):
    df = df1.copy()
    df_impressions = df.drop(columns = ['data','item_id'])
    df_impressions.dropna(inplace=True)
    
    df_impressions = df_impressions.groupby(by=['user_id'])['impression_list'].apply(list).reset_index()

    def helper(list_string):
        data = []
        for string in list_string:
            stuff = string.split(',')
            for s in stuff:
                int(s)
                data.append(int(s))
        return data

    df_impressions['impression_list']= df_impressions['impression_list'].apply(helper)
    df_impressions = df_impressions.groupby(by=['user_id'])['impression_list'].apply(np.array).reset_index()
    df_impressions['impression_list'] = df_impressions['impression_list'].apply(np.concatenate)
    df_impressions['impression_list'] = df_impressions['impression_list'].apply(np.unique)
    
    A  = np.zeros(shape=(num_users,num_items))
    for i in df_impressions['user_id']:
        for j in df_impressions.iloc[i]['impression_list']:
            A[i,j]=1

    URM_impression = sps.csr_matrix(A)
    return URM_impression

def build_ICM_lengh(dataset_l):
    df_l = dataset_l.copy()
    df_l.drop(columns = ['feature_id'], inplace = True)
    df = dataset_l[dataset_l.item_id <= 24506]
    num_items, min_item_id, max_item_id = len(df.data.unique()), df.data.min(), df.data.max()
    item = range(0,24506)
    df_1 = pd.get_dummies(df.data)
    column = range(num_items)
    df_1.columns = column
    df_1.index = df.item_id
    ICM_lenght = pd.DataFrame(np.zeros((24507,210)))
    ICM_lenght[ICM_lenght.index.isin(df_1.index)] = df_1
    return sps.csr_matrix(ICM_lenght)

def preprocess(df):
    """Preprocess data for KMeans clustering"""
    
    df_log = np.log1p(df)
    scaler = StandardScaler()
    scaler.fit(df_log)
    df_norm = scaler.transform(df_log)
    
    return df_norm
def build_ICM_lengh_kmeans_3(dataset_l, n_clusters=3):
    df_l = dataset_l.copy()
    df_l.drop(columns = ['feature_id'], inplace = True)
    df = dataset_l[dataset_l.item_id <= 24506]
    data = df.data.values
    dataPR = data.reshape(-1,1)
    dataPR = preprocess(dataPR)
    kmeans = KMeans(n_clusters, random_state=0).fit(dataPR)
    df.data = kmeans.labels_.copy()
    num_items, min_item_id, max_item_id = len(df.data.unique()), df.data.min(), df.data.max()
    item = range(0,24506)
    df_1 = pd.get_dummies(df.data)
    column = range(num_items)
    df_1.columns = column
    df_1.index = df.item_id
    ICM_lenght = pd.DataFrame(np.zeros((24507,n_clusters)))
    ICM_lenght[ICM_lenght.index.isin(df_1.index)] = df_1
    return sps.csr_matrix(ICM_lenght)
def build_ICM_lengh_kmeans(dataset_l):
    df_l = dataset_l.copy()
    df_l.drop(columns = ['feature_id'], inplace = True)
    df = dataset_l[dataset_l.item_id <= 24506]
    data = df.data.values
    dataPR = data.reshape(-1,1)
    kmeans = KMeans(n_clusters=7, random_state=0).fit(dataPR)
    df.data = kmeans.labels_.copy()
    num_items, min_item_id, max_item_id = len(df.data.unique()), df.data.min(), df.data.max()
    item = range(0,24506)
    df_1 = pd.get_dummies(df.data)
    column = range(num_items)
    df_1.columns = column
    df_1.index = df.item_id
    ICM_lenght = pd.DataFrame(np.zeros((24507,7)))
    ICM_lenght[ICM_lenght.index.isin(df_1.index)] = df_1
    return sps.csr_matrix(ICM_lenght)

def build_ICM_type(dataset_t):
    df_t = dataset_t.copy()
    df_t.drop(columns = ['data'], inplace = True)
    df = dataset_t[dataset_t.item_id <= 24506]
    num_items, min_item_id, max_item_id = len(df.feature_id.unique()), df.feature_id.min(), df.feature_id.max()
    item = range(0,24506)
    df_1 = pd.get_dummies(df.feature_id)
    column = range(num_items)
    df_1.columns = column
    df_1.index = df.item_id
    ICM_type = pd.DataFrame(np.zeros((24507,5)))
    ICM_type[ICM_type.index.isin(df_1.index)] = df_1
    return sps.csr_matrix(ICM_type)

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
        dataset['data'] = dataset['data'].replace({0:data_weight})
        dataset = dataset.groupby(by=['user_id','item_id']).sum('data').reset_index()

        unique_users = dataset.user_id.unique()
        unique_items = dataset.item_id.unique()

        num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
        num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

        URM_ALL = sps.csr_matrix((dataset.data, (dataset.user_id, dataset.item_id)), 
                                shape=(num_users, num_items))

    return URM_ALL

def userwise(URM_train,URM_test,num_group,cutoff = 10,recommender_object_dict={}):
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

def build_URM_ICM_onlyURM_item(dataset_, dataset_type_, dataset_lenght_, implicit=True,data_weight=1):
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
    ICM_All=ICM_All[ICM_All.item_id < 24507]
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
        #df = pd.concat([support,df])
        #df = df.reset_index()

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
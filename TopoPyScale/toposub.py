'''
Clustering routines for TopoSUB
S. Filhol, Oct 2021

'''

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

def scale_df(df_param, scaler=StandardScaler()):
    '''
    Function to scale features of a pandas dataframe

    :param df_param: pandas dataframe with features to scale
    :param scaler: scikit learn scaler. Default is StandardScaler()
    :return: scaled pandas dataframe
    '''
    df_scaled = pd.DataFrame(scaler.fit_transform(df_param.values),
                      columns=df_param.columns, index=df_param.index)
    return df_scaled, scaler

def inverse_scale_df(df_scaled, scaler):
    '''
    Function to inverse feature scaling of a pandas dataframe

    :param df_scaled: pandas dataframe to rescale to original (inverse transfrom
    :param scaler: original scikit learn scaler
    :return:
    '''
    df_inv = pd.DataFrame(scaler.inverse_transform(df_scaled.values),
                      columns=df_scaled.columns, index=df_scaled.index)
    return df_inv


def kmeans_clustering(df_param, n_clusters=100, **kwargs):
    '''
    Function to perform K-mean clustering

    :param df_param: pandas dataframe with features
    :param n_clusters: number of clusters
    :param kwargs:
    :return:
    '''
    X = df_param.to_array()
    col_names = df_param.columns
    kmeans = KMeans(n_clusters=n_clusters, **kwargs).fit(X)
    df_centers = pd.DataFrame()
    df_centers[col_names] = kmeans.cluster_centers_

    return df_centers, kmeans
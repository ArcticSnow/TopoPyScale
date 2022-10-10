import os

import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def correct_trend(df,
                  reference_col='obs',
                  target_col='dow',
                  apply_correction=True):
    """
    Function to estimate linear trend correction.
    Args:
        df (dataframe):
        reference_col (str): name of the reference column, i.e. observation timeseries
        target_col (str): name of the target column, i.e. downscale timeseries
        apply_correction (bool): applying correction to target data

    Returns:
        dict: metrics of the linear trend estimate
        dataframe (optional): corrected values

    """
    print('---> Estimating linear detrending of target data')
    reg = stats.linregress(df[target_col].values, df[reference_col].values)

    # Compute metrics
    n = df.shape[0]
    RMSE = np.round(np.sqrt((df[reference_col] - df[target_col]).sum()**2/n),1)
    bias = np.round((df[reference_col] - df[target_col]).sum()/n,2)

    tbox = 'N = {}\ny = {} + {}x\nr = {}; p-value = {}\nRMSE = {}\nbias = {}'.format(n,
                                                                                     np.round(reg.intercept,2),
                                                                                     np.round(reg.slope,2),
                                                                                     np.round(reg.rvalue,2),
                                                                                     np.round(reg.pvalue,2),
                                                                                     RMSE,
                                                                                     bias)
    print('=> Results <=')
    print(tbox)
    metrics = {'reg': reg, 'RMSE': RMSE, 'bias': bias, 'tbox': tbox}

    if apply_correction:
        df['cor'] = df[target_col] * reg.slope + reg.intercept
        return metrics, df['cor']
    else:
        return metrics

def correct_seasonal(df, reference_col='obs', target_col='dow', plot=True, apply_correction=True):
    """
    Function to correct for seasonal signal.
        1. it groups all days by day_of_year and extract the median difference.
        2. it computes the 31 day rolling mean (padded on both sides)

    Args:
        df (dataframe): index must be a datetime, with daily timestep. other columns are reference and target.
        reference_col (str): name of the reference column, i.e. observation timeseries
        target_col (str): name of the target column, i.e. downscale timeseries
        plot (bool):
        apply_correction (bool): apply correction and return corrected data

    Returns:
        dataframe - correction for each day of year, starting on January 1.

    """
    df['doy'] = pd.to_datetime(df.index).day_of_year
    df['dif'] = df[reference_col] - df[target_col]
    se = df.groupby(['doy'])['dif'].median()

    # account for cyclicity of the data for computing the rolling mean aver a 31 day period
    se = pd.concat([se.iloc[-16:], se, se[:16]]).rolling(31, center=True).mean().reset_index()
    se = se.iloc[np.arange(16, 16+366)].reset_index()
    if plot:
        plt.figure()
        df.groupby(['doy'])['dif'].median().plot(label='Daily median')
        se.dif.plot(label='31D mean')
        plt.legend()
        plt.show()

    if apply_correction:
        df['cor'] = 0
        for i, row in se.iterrows():
            df['cor'][df.doy==row.doy] = df[target_col][df.doy==row.doy] + row.dif
        return se, df['cor']
    else:
        return se

def obs_vs_downscaled(df,
                      reference_col='obs',
                      target_col='dow',
                      trend_correction=True,
                      seasonal_correction=True,
                      param={'xlab': 'Downscaled [unit]',
                             'ylab': 'Observation [unit]',
                             'xlim': (-20,20),
                             'ylim': (-20,20),
                             'title': None},
                      plot='heatmap'):
    """
    Function to compare Observation to Downscaled for one given variable.

    Args:
        df (dataframe): pandas dataframe containing corresponding Observation and Downscaled values
        reference_col (str): name of the reference column. Observation
        target_col (str): name of the target column. Downscaled timeseries
        trend_correction (bool): remove trend by applying a linear regression
        seasonal_correction (bool): remove seasonal signal by deriving median per day of the year and computing the rolling mean over 31d on the median
        param (dict): parameter for comparison and plotting
        plot (str): plot type: heatmap or timeseries

    Returns:
        dict: metrics of regression and comparison
        dataframe: dataframe containing the seasonal corrections to applied
        dataframe: corrected values

    Inspired by this study: https://reader.elsevier.com/reader/sd/pii/S0048969722015522?token=106483481240DE6206D83D9C7EC9D4990C2C3DE6F669EDE39DCB54FF7495A31CC57BDCF3370A6CA39D09BE293EACDDBB&originRegion=eu-west-1&originCreation=20220316094240

    """
    metrics = None
    se = None

    if trend_correction and not seasonal_correction:
        metrics, df['cor'] = correct_trend(df, reference_col=reference_col, target_col=target_col, apply_correction=True)
    elif trend_correction and seasonal_correction:
        metrics, df['cor_tmp'] = correct_trend(df, reference_col=reference_col, target_col=target_col, apply_correction=True)
        se, df['cor'] = correct_seasonal(df, reference_col=reference_col, target_col='cor_tmp', apply_correction=True, plot=False)

    elif not trend_correction and seasonal_correction:
        se, df['cor'] = correct_seasonal(df, reference_col=reference_col, target_col=target_col, apply_correction=True, plot=False)

    if plot == 'heatmap':
        # plot Obs. against Downscaled in a 2D histogram
        plt.figure()
        ax = sns.histplot(x=df[target_col], y=df[reference_col], cmap='magma')
        plt.plot(param['xlim'], param['ylim'], c='b', linewidth=2, label='1:1 line')

        if metrics is not None:
            plt.plot(param['xlim'], np.array(param['xlim'])*metrics['reg'].slope + metrics['reg'].intercept, c='r', label='regression')
            props = dict(boxstyle='round', facecolor='white', edgecolor='0.8')
            plt.text(0.03, 0.97, metrics['tbox'], transform=ax.transAxes, verticalalignment='top', bbox=props)
        plt.legend(loc='lower right')
        plt.ylabel(param['ylab'])
        plt.xlabel(param['xlab'])
        plt.xlim(param['xlim'])
        plt.ylim(param['ylim'])
        if param['title'] is not None:
            plt.title(param['title'])
        plt.show()

    elif plot == 'timeseries':

        fig, ax = plt.subplots(2,1, sharex=True,gridspec_kw={'height_ratios':[2,1]})
        df[target_col].plot(ax=ax[0], label='Downscaled')
        df[reference_col].plot(ax=ax[0], label='Observation')
        dif = df[reference_col] - df[target_col]
        dif.plot(ax=ax[1], label='1:1 difference')
        dif.rolling('30D').mean().plot(ax=ax[1], c='r', label='30D mean')

        ax[0].legend()
        ax[1].legend()
        if param['title'] is not None:
            ax[0].set_title(param['title'])
        ax[0].set_ylabel(param['ylab'])
        ax[1].set_ylim((-5,5))
        ax[1].set_ylabel('Diff [$^{o}C$]')

        plt.show()

    return metrics, se, df['cor']
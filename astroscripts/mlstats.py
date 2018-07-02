"""
MLSTATS: MACHINE LEARNING AND STATISTICS ROUTINES

This package has an optimized set of functions for my daily work.
"""

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import scipy #pearson correlation
import matplotlib.pyplot as plt

def rotate_axis(x):
    mean_value = np.mean(x)
    x = x - mean_value
    x_new = -1 * x#invert the sign
    return x_new + mean_value

def pearson_ica_test(ica_signal,original_signal,save_dir):
    '''
    Return Pearson Statistcs about which column in the ica output is
    correlate with the main first principal component that corresponds
    to the light curve transit.

    Input:

    ica_signal: pandas dataframe
    original_signal: pandas dataframe

    '''

    pca = PCA(n_components=len(original_signal.columns))
    H = pca.fit_transform(original_signal)
    H = pd.DataFrame(H)

    H.plot(grid=True)
    print('Scatter 1st component = ',np.std(H[0]))
    plt.title('PCA Components')
    plt.savefig(save_dir+'PCA_components_.png')
    plt.savefig(save_dir+'PCA_components_.pdf')
    plt.close()

    pearson,pvalue = np.zeros(ica_signal.shape[1]), np.zeros(ica_signal.shape[1])

    component_id = 0
    for i in range(ica_signal.shape[1]):
        pearson[i], pvalue[i] = scipy.stats.pearsonr(H[0],ica_signal[i])
        print(pearson[i], pvalue[i])
        if abs(pearson[i]) == abs(pearson).max():
            print('** Light curve on column = ',i,'\n')
            component_id = i
        else:
            print('** Probabily, this is not the light curve \n')
    return component_id
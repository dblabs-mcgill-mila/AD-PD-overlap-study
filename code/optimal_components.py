"""
Script to find optimal number of PLS components based on cross-validation.
"""

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import seaborn as sns

from joblib import Parallel, delayed
from numpy.random import default_rng
from numpy import random

import scipy
from scipy.stats import pearsonr, ttest_ind, mode
from scipy.optimize import curve_fit

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_validate, KFold, StratifiedKFold, cross_val_score, LeaveOneGroupOut, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
import anndata as ann
import joblib
import gseapy as gp
from gseapy import Biomart
from collections import namedtuple
from copy import deepcopy
import os
from itertools import chain, permutations
from collections import OrderedDict
import gc
from scipy.linalg import pinv as pinv2
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import run
from scipy.stats import scoreatpercentile
from sklearn.utils import resample



def gridCvPLS(X_train, y_train, max_comp, numFolds, cell, n_jobs):
    scPLS = make_pipeline(PLSRegression(scale=False))
    gridcv = GridSearchCV(
        estimator = scPLS, 
        param_grid = {'plsregression__n_components':[i for i in range(2,max_comp+1)]},
        refit=False, 
        scoring='r2', 
        cv=StratifiedKFold( n_splits = numFolds, shuffle = True, random_state = 1), 
        n_jobs = n_jobs, 
        return_train_score = False )
    _ = gridcv.fit( X_train, y_train )
    print(f"{cell}: {gridcv.best_params_} selected")
    
    return gridcv


####################  REGRESS OUT COVARIATE ####################
def _do_lr_bootstrap(X, y, iter, lr_model, **model_params):
    model = lr_model(**model_params)
    X_bootstrap, y_bootstrap = resample(X, y, replace=True, random_state=iter)
    model.fit(y, X)
    return model.coef_

def _thresh_bs(bs_coefs, true_coef, threshold=95):
    high_per = (100 + threshold)/2
    low_per = 100-high_per
    permute_perc_high = scoreatpercentile(bs_coefs, per = high_per, axis=0)
    permute_perc_low = scoreatpercentile(bs_coefs, per = low_per, axis=0)
    # zero loadings that are <higher_percentile or >lower_percentile
    return ((permute_perc_high>0) & (permute_perc_low<0))

def regress_out(X, y, n_bootstrap=1, threshold=95):
    scaler = StandardScaler(copy=False)
    y = scaler.fit_transform(y)
    model_params = {'fit_intercept': True}
    lr_model = Ridge
    model = lr_model(**model_params)
    model.fit(y, X)
    bs_coefs = Parallel(n_jobs=20, verbose=10)(delayed(_do_lr_bootstrap)(X, y, iter, lr_model, **model_params) for iter in range(n_bootstrap))
    mask_idx = _thresh_bs(bs_coefs, model.coef_, threshold=threshold)[:,0]
    print(f'pmi unrelated to {mask_idx.shape} genes, regressing the rest out')
    X_res = X
    X_res[:, ~mask_idx] = X[:, ~mask_idx] -  model.predict(y)[:, ~mask_idx]
    return X_res

########################################################################################################################


_modelScoreAndParams = namedtuple('ModelScoreAndParams', 'id score median_score n_components')

def doPLSRegression(adata, celltypes, M, K, trn_frac, max_comp, n_jobs=7, n_cells_max = None, do_grid_search = True, covariate = 'pmi'):
    model_scores = {cell: [] for cell in celltypes}
    celltype_scores = {cell: [] for cell in celltypes}
    print(adata.obs['cell_type'].unique())
    for cell in  celltypes:
        print(cell.upper())

        ### PREPROCESSING ####
        data = adata[adata.obs['cell_type']==cell]
        
        if n_cells_max and (data.shape[0]>n_cells_max):
            sc.pp.subsample(data, n_obs=n_cells_max, random_state=0)

        # balance classes
        data_PD = data[data.obs['diagnosis'] > 0]
        data_ctrl = data[data.obs['diagnosis'] <= 0 ]

        n_PD = data_PD.shape[0]
        n_ctrl = data_ctrl.shape[0]

        if(n_PD > n_ctrl):
            sc.pp.subsample(data_PD, fraction=n_ctrl/n_PD)
        else:
            sc.pp.subsample(data_ctrl, fraction=n_PD/n_ctrl)

        data = data_PD.concatenate(data_ctrl)
        print(f'DATA SHAPE: {cell}, {data_PD.shape}, {data_ctrl.shape}')
        del(data_PD)
        del(data_ctrl) 
        sc.pp.filter_genes(data, min_cells=int(1+data.shape[0]/1000))
        
        X = data.X.toarray().astype('float32')
        y = data.obs['diagnosis'].values
        if covariate:
            y_cov = data.obs[covariate].fillna(0).values.reshape(-1, 1)

        del(data)

        ####### MODEL FIT ####
        
        scaler = StandardScaler(copy=False)
        X_ = scaler.fit_transform(X)
        X_reg = X_
        
        if covariate:
            X_reg = regress_out(X, y_cov, n_bootstrap=500, threshold=95)
            scaler = StandardScaler(copy=False)
            X_reg = scaler.fit_transform(X_reg)
        del(X)
        pca_comps = min(500, X_reg.shape[0])
        pca = PCA(n_components=pca_comps, copy=True, random_state=0)
        X_ = pca.fit_transform(X_reg)

        gridcv = None

        for idx in range(M):
            X_train, X_test, y_train, y_test = train_test_split(X_, y, train_size=trn_frac, shuffle=True)
            if do_grid_search:
                gridcv = gridCvPLS(X_train, y_train, max_comp, K, cell, n_jobs) 
                scPLS_optimal = make_pipeline(PLSRegression(n_components=gridcv.best_params_['plsregression__n_components'], scale=False))   
            else:
                scPLS_optimal = make_pipeline(PLSRegression(n_components=max_comp, scale=False))

            del(X_train)
            del(y_train)
            scores = cross_val_score(scPLS_optimal, X_test, y_test, cv=StratifiedKFold(n_splits=K, shuffle=True, random_state=0), n_jobs=n_jobs, scoring = 'r2')
            model_scores[cell].append(_modelScoreAndParams(idx, scores, np.mean(scores), gridcv.best_params_['plsregression__n_components'] if gridcv else max_comp ))

        celltype_scores[cell] = np.mean([ score.median_score for score in model_scores[cell]])
        
    return celltype_scores

def optimal_ros():
    prefix_adata = '/Users/anwesha/Library/CloudStorage/OneDrive-McGillUniversity/PHD/Research/adatas/'
    #### AD Ros
    adata_ros = sc.read_h5ad(prefix_adata+'mathys19_pp_filtered_June21.h5ad')
    genes_ros = sc.read_h5ad(prefix_adata+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    celltypes = adata_ros.obs['cell_type'].unique().tolist()
    celltype_scores = doPLSRegression(adata_ros, celltypes, M=10, K=10, trn_frac=0.8, max_comp = len(celltypes), n_cells_max= 15000)
    return celltype_scores

def optimal_sea(celltypes = None, n_jobs = 20):
    prefix_adata = '/Users/anwesha/Library/CloudStorage/OneDrive-McGillUniversity/PHD/Research/adatas/'
    #### AD SEA

    ADATA_PRE = run.ADATA_PRE
    celltypes = celltypes if celltypes else ['sncg', 'sst', 'pvalb'] #['astro', 'opc', 'micro', 'oligo', 'l4_it', 'l5_it', 'vip'] #'endo', 
    celltype_scores = {}
    for celltype in celltypes:
        print(f'reading {run.ADATA_PRE}local_{celltype}.h5ad')
        adata = sc.read_h5ad(run.ADATA_PRE+f'local_{celltype}.h5ad')

        genesym_path = run.PATH_DIR+'ensemble2HGNC.csv'
        genesym_map = pd.read_csv(genesym_path, index_col=0)
        def get_hgnc_sym(x):
            res = genesym_map[genesym_map['Ensemble_ID']==x]['HGNC_symbol'].values
            if res.size>0:
                return res[0] if not res[0]=='nan' else x
            else:
                return x
        # adata.var.reset_index(inplace=True)
        adata.var['gene_symbol'] = adata.var.index.map( lambda x: get_hgnc_sym(x))
        adata.var['gene_symbol'] = adata.var['gene_symbol'].astype(str)
        adata.var.set_index('gene_symbol', inplace=True)
        genes_ros = sc.read_h5ad(run.ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
        adata = adata[:, adata.var.index.isin(genes_ros)]
        sc.pp.filter_cells(adata, min_genes=int(1+adata.shape[0]/1000))
        sc.pp.filter_genes(adata, min_cells=int(1+adata.shape[1]/1000))
        
        adata.var_names_make_unique()
        diagnosis_map = {'Low':1, 'Reference':-1, 'Intermediate':1, 'High':1, 'Not AD':-1}
        adata.obs['diagnosis'] = adata.obs.ADNC.map(lambda x: diagnosis_map[x])
        adata.obs['cell_type'] = celltype
        adata = adata[~adata.obs['Age at death'].isin(['Less than 65 years old', '65 to 77 years old'])]
        pmi_map = {'3.2 to 5.9 hours': 3.5, '5.9 to 8.7 hours': 7.5, '8.7 to 11.4 hours': 10.5, 'Reference':float('nan')}
        adata.obs['pmi'] = adata.obs.PMI.map(lambda x: pmi_map[x]).astype('float32')
        adata.obs.drop(columns='PMI', inplace=True)
        print('read, now starting optimal grid cv')
        scores = doPLSRegression(adata, [celltype], M=10, K=10, trn_frac=0.8, max_comp = 5, n_cells_max= 15000, n_jobs=n_jobs)
        celltype_scores.update(scores)
    return celltype_scores

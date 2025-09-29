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
from itertools import chain, permutations, product
from collections import OrderedDict
import gc
from scipy.linalg import pinv as pinv2
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import run
from sklearn.utils import resample

PATH_DIR = run.PATH_DIR
ADATA_PRE = run.ADATA_PRE

def AD_optimal_map():
    return OrderedDict({ 
                'Ex': 6,
                'Oli': 3,
                'In': 4,
                'Mic': 2,
                'Opc': 2,
                'Ast': 3,
                'End': 2,
                'Per': 2,
                })


def PD_optimal_map():
    return OrderedDict({ 
                'SOX6': 7,
                'CALB1': 10,
                'Ependyma': 2,
                'Microglia': 7,
                'Macrophage': 3,
                'Astrocyte': 7,
                'Endothelial': 8,
                'OPC': 7,
                'Excitatory neuron': 10,
                'Inhibitory neuron': 8,
                'Oligodendrocyte': 7,
                })


def regress_out(self, X, y, n_bootstrap=1, threshold=95):
    scaler = StandardScaler(copy=False)
    y = scaler.fit_transform(y)
    model_params = {'fit_intercept': True}
    lr_model = Ridge
    model = lr_model(**model_params)
    model.fit(y, X)
    bs_coefs = Parallel(n_jobs=20, verbose=10)(delayed(self._do_lr_bootstrap)(X, y, iter, lr_model, **model_params) for iter in range(n_bootstrap))
    mask_idx = self._thresh_bs(bs_coefs, model.coef_, threshold=threshold)[:,0]
    print(f'pmi unrelated to {mask_idx.shape} genes, regressing the rest out')
    X_res = X
    X_res[:, ~mask_idx] = X[:, ~mask_idx] -  model.predict(y)[:, ~mask_idx]
    return X_res


_modelScoreAndParams_bs = namedtuple('ModelScoreAndParams', 'id score')


def stratified_resampling(data, target_count, random_state=0):
    resampled_data = {}
    np.random.seed(random_state)
    for diag in data.diagnosis.unique():
        resampled_data[diag] = {}
        # Calculate total cells and proportion for each ID
        _df = data[data.diagnosis==diag]
        total_cells = _df.shape[0]
        for id in _df.donor_id.unique():
            proportion = data[data.donor_id==id].shape[0] / total_cells
            resample_count = int(proportion * target_count)
            # Sample cells for each ID based on calculated proportion
            resampled_data[diag][id] = resample(_df[_df.donor_id==id].index, n_samples=resample_count, replace=True).tolist()
    
    return resampled_data


def clustered_bootstrap(adata, celltype, idx, n_comp, covariate = 'pmi', pca_max = 500, trial = 0):
    try:
        print(f'{celltype} IDX: {idx}')
        data_PD_ = adata[adata.obs['diagnosis'] > 0]
        data_ctrl_ = adata[adata.obs['diagnosis'] <= 0]
        
        # del(adata)

        #Get bootstrap indexes for cases
        np.random.seed(idx)

        bs_ids = np.random.choice(data_PD_.obs.donor_id.unique(), size = data_PD_.obs.donor_id.unique().shape[0],replace = True)
        exec_ids = [id for id in data_PD_.obs.donor_id.unique() if id not in bs_ids]
        data_PD = data_PD_[data_PD_.obs.donor_id==bs_ids[0]]
        test_dataPD = data_PD_[data_PD_.obs.donor_id.isin(exec_ids)]
        for id in bs_ids[1:]:
            data_PD = ann.concat([data_PD, data_PD_[data_PD_.obs.donor_id==id]])
        del(data_PD_)

        bs_ids = np.random.choice(data_ctrl_.obs.donor_id.unique(), size = data_ctrl_.obs.donor_id.unique().shape[0], replace = True)
        exec_ids = [id for id in data_ctrl_.obs.donor_id.unique() if id not in bs_ids]
        data_ctrl = data_ctrl_[data_ctrl_.obs.donor_id==bs_ids[0]]
        test_datactrl = data_ctrl_[data_ctrl_.obs.donor_id.isin(exec_ids)]
        for id in bs_ids[1:]:
            data_ctrl = ann.concat([data_ctrl, data_ctrl_[data_ctrl_.obs.donor_id==id]])
        del(data_ctrl_)

        train_dataPD = data_PD #sc.pp.subsample(data_PD, fraction=0.8, random_state=0, copy = True)
        # test_dataPD = data_PD[~data_PD.obs_names.isin(train_dataPD.obs_names)]

        train_datactrl = data_ctrl #sc.pp.subsample(data_ctrl, fraction=0.8, random_state=0, copy = True)
        # test_datactrl = data_ctrl[~data_ctrl.obs_names.isin(train_datactrl.obs_names)]

        del(data_PD)
        del(data_ctrl) 

        # balance classes
        data = ann.concat([train_dataPD, train_datactrl])
        resampled_idxs = stratified_resampling(data.obs, min(data[data.obs.diagnosis==1].shape[0], data[data.obs.diagnosis==-1].shape[0]), random_state=idx)
        obs_res = pd.DataFrame()
        for diag in resampled_idxs:
            for id in resampled_idxs[diag]:
                _df = data.obs.loc[resampled_idxs[diag][id], :]
                obs_res = pd.concat([obs_res, _df])
        
        data = data[data.obs.index.isin(obs_res.index), :]
        del(resampled_idxs)
        del(obs_res)

        test_data = ann.concat([test_dataPD, test_datactrl])
        
        del(train_dataPD)
        del(train_datactrl) 
        del(test_dataPD)
        del(test_datactrl) 

        sc.pp.filter_genes(data, min_cells=int(1+data.shape[0]/1000))
        test_data = test_data[:, data.var.index]

        X_train = data.X.toarray().astype('float32')
        y_train = data.obs['diagnosis'].values
        X_test = test_data.X.toarray().astype('float32')
        y_test = test_data.obs['diagnosis'].values

        if covariate:
            y_cov = data.obs[covariate].fillna(0).values.reshape(-1, 1)

        del(data)

        ####### MODEL FIT ####
        
        scaler = StandardScaler(copy=False)
        X_ = scaler.fit_transform(X_train)
        X_reg = X_
        
        if covariate:
            X_reg = regress_out(X_train, y_cov, n_bootstrap=500, threshold=95)
            scaler_reg = StandardScaler(copy=False)
            X_reg = scaler_reg.fit_transform(X_reg)
        del(X_train)
        pca_comps = min(pca_max, X_reg.shape[0])
        pca = PCA(n_components=pca_comps, copy=True, random_state=0)
        X_ = pca.fit_transform(X_reg)


        scPLS_optimal = PLSRegression(n_components=n_comp,  scale=False, copy=True)
        scPLS_optimal.fit(X_, y_train)
        # del(y_train)
        # Transform test set
        X_test_scaled = scaler.transform(X_test)  # Use the scaler fitted on the training set
        if covariate:
            X_test_scaled = scaler_reg.transform(X_test_scaled)  # Use the scaler fitted on the training set
        X_test_pca = pca.transform(X_test_scaled)  # Apply PCA transformation

        # Predict and score
        scores = roc_auc_score(y_test, scPLS_optimal.predict(X_test_pca))
        print(f'FINISHED INDEX {celltype} {idx}')
        
        return _modelScoreAndParams_bs(idx, scores)
    except Exception as e:
        print(e)
        trial += 1
        if trial < 3:
            print(f'Trying {trial} time')
            return clustered_bootstrap(adata, celltype, idx+100000+trial, n_comp, covariate = covariate, pca_max = pca_max, trial = trial)
        else:
            return _modelScoreAndParams_bs(idx, float('nan'))


def doPLSRegression(adata, celltypes, M, optimalMapping, n_jobs=10, n_cells_max = None, covariate = None, pca_max=500, start_idx = 0):
    model_scores = {} # {cell: [] for cell in celltypes}
    try:
        for cell in  celltypes:
            print(cell.upper())
            ### PREPROCESSING ####
            data = adata[adata.obs['cell_type']==cell]
            
            if n_cells_max and (data.shape[0]>n_cells_max):
                sc.pp.subsample(data, n_obs=n_cells_max, random_state=0)

            donor_ids = data.obs[['donor_id', 'diagnosis']].reset_index()
            
            _df = donor_ids.groupby('diagnosis')['donor_id'].unique().values

            scores = Parallel(n_jobs=n_jobs, verbose = 2)(delayed(clustered_bootstrap)(data, cell, idx, optimalMapping[cell], covariate, pca_max)for idx in range(start_idx, M))
            # scores = Parallel(n_jobs=n_jobs)(delayed(_do_fit)(X_, y, idx, trn_frac, optimalMapping[cell]) for idx in range(M))
            model_scores[cell] = scores
            
            gc.collect()
    except KeyboardInterrupt as e:
        print(e)
        score_df = pd.DataFrame()
        print(model_scores.items())
        for k, vals in model_scores.items():
            _df = pd.DataFrame.from_dict(vals)
            _df['celltype'] = k
            print(cell)
            print(_df.head())
            _df.set_index('id', inplace=True)
            score_df = pd.concat([score_df, _df])
        print('SCORES written')
    return model_scores


def run_ros(M=10, covariate='pmi', trn_frac = 0.8, n_cells_max=None, n_jobs=10, pca_max = 10000, start_idx = 0):
    path_dir = PATH_DIR
    adata = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad')
    adata.obs.rename(columns={'id': 'donor_id'}, inplace=True)
    celltypes = adata.obs['cell_type'].unique().tolist()
    print(celltypes)
    celltype_scores_ros = doPLSRegression(adata, celltypes, M=M, optimalMapping = AD_optimal_map(), n_cells_max= n_cells_max, n_jobs=n_jobs, pca_max=pca_max, start_idx = start_idx)
    score_df = pd.DataFrame()
    for k, vals in celltype_scores_ros.items():
        _df = pd.DataFrame.from_dict(vals)
        _df['celltype'] = k
        _df.set_index('id', inplace=True)
        score_df = pd.concat([score_df, _df])

    score_df.to_csv(PATH_DIR + f'AD/Rosmap/score_df_{start_idx}_to_{M}_{covariate}_pca{pca_max}_bs_execid.csv')
    return score_df


def run_kam(celltypes = None, M=10, covariate='pmi', trn_frac = 0.8, n_cells_max=None, n_jobs=10, pca_max = 10000, start_idx=0):
    path_dir = PATH_DIR

    adata = sc.read_h5ad(ADATA_PRE+'kadata_pp_agematch_Oct31.h5ad')
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.gene_ids.isin(genes_ros)]

    celltypes = ['SOX6',
                'CALB1',
                'Microglia',
                'Astrocyte',
                'Endothelial',
                'OPC',
                'Excitatory neuron',
                'Inhibitory neuron',
                'Oligodendrocyte'] #celltypes if celltypes else adata.obs['cell_type'].unique().tolist()[::-1]
    print('CELLTYPES: ', celltypes)
    celltype_scores = doPLSRegression(adata, celltypes, M=M, optimalMapping = PD_optimal_map(), n_cells_max= n_cells_max, n_jobs=n_jobs, pca_max=pca_max, start_idx=start_idx)
    score_df = pd.DataFrame()
    for k, vals in celltype_scores.items():
        _df = pd.DataFrame.from_dict(vals)
        _df['celltype'] = k
        _df.set_index('id', inplace=True)
        score_df = pd.concat([score_df, _df])
    
    score_df.to_csv(PATH_DIR + f'PD/Kam/score_df_{start_idx}_to_{M}_{covariate}_pca{pca_max}_bs_execid.csv')
    return score_df


def run_sm(M=10, covariate='pmi', trn_frac = 0.8, n_cells_max=None, n_jobs=10, pca_max = 10000, start_idx=0):
    path_dir = PATH_DIR

    adata = sc.read_h5ad(ADATA_PRE+'adata_pp_Nov6_pmi.h5ad')
    adata.obs.rename(columns={'patient': 'donor_id'}, inplace=True)
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.index.isin(genes_ros)]

    celltypes = run.PD_Sm_optimal_map().keys()
    celltype_scores = doPLSRegression(adata, celltypes, M=M, optimalMapping = run.PD_Sm_optimal_map(), n_cells_max= n_cells_max, n_jobs=n_jobs, pca_max=pca_max, start_idx=start_idx)
    score_df = pd.DataFrame()
    for k, vals in celltype_scores.items():
        _df = pd.DataFrame.from_dict(vals)
        _df['celltype'] = k
        _df.set_index('id', inplace=True)
        score_df = pd.concat([score_df, _df])

    score_df.to_csv(PATH_DIR + f'PD/Sm/score_df_{start_idx}_to_{M}_{covariate}_pca{pca_max}_bs_execid.csv')


def run_sea(M=10, covariate='pmi', trn_frac = 0.8, n_cells_max=None, n_jobs=10, pca_max = 10000, start_idx=0):
    path_dir = PATH_DIR

    adata = sc.read_h5ad(ADATA_PRE+'adata_pp_Nov6_pmi.h5ad')

    celltypes = ['astro', 'endo', 'opc', 'micro', 'oligo', 'l4_it', 'l5_it', 'vip']
    score_df = pd.DataFrame()

    genesym_path = PATH_DIR+'ensemble2HGNC.csv'
    genesym_map = pd.read_csv(genesym_path, index_col=0)
    def get_hgnc_sym(x):
        res = genesym_map[genesym_map['Ensemble_ID']==x]['HGNC_symbol'].values
        if res.size>0:
            return res[0] if not res[0]=='nan' else x
        else:
            return x
        
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()

    for celltype in celltypes:
        print(f'reading {ADATA_PRE}local_{celltype}.h5ad')
        adata = sc.read_h5ad(ADATA_PRE+f'local_{celltype}.h5ad')

        # adata.var.reset_index(inplace=True)
        adata.var['gene_symbol'] = adata.var.index.map( lambda x: get_hgnc_sym(x))
        adata.var['gene_symbol'] = adata.var['gene_symbol'].astype(str)
        adata.var.set_index('gene_symbol', inplace=True)
        
        adata = adata[:, adata.var.index.isin(genes_ros)]
        # Below not done. DO it.
        sc.pp.filter_cells(adata, min_genes=int(1+adata.shape[0]/1000))
        sc.pp.filter_genes(adata, min_cells=int(1+adata.shape[1]/1000))
        # sc.pp.normalize_total(adata)
        # sc.pp.log1p(adata)
        adata.var_names_make_unique()
        diagnosis_map = {'Low':1, 'Reference':-1, 'Intermediate':1, 'High':1, 'Not AD':-1}
        adata.obs['diagnosis'] = adata.obs.ADNC.map(lambda x: diagnosis_map[x])
        adata.obs['cell_type'] = celltype
        adata = adata[~adata.obs['Age at death'].isin(['Less than 65 years old', '65 to 77 years old'])]
        pmi_map = {'3.2 to 5.9 hours': 3.5, '5.9 to 8.7 hours': 7.5, '8.7 to 11.4 hours': 10.5, 'Reference':float('nan')}
        adata.obs['pmi'] = adata.obs.PMI.map(lambda x: pmi_map[x]).astype('float32')
        adata.obs.drop(columns='PMI', inplace=True)
        celltype_score = doPLSRegression(adata, [celltype], M=M, optimalMapping = run.AD_Seattle_optimal_map(), n_cells_max= n_cells_max, n_jobs=n_jobs, pca_max=pca_max, start_idx=start_idx)
        _df = pd.DataFrame.from_dict(celltype_score[celltype])
        _df['celltype'] = celltype
        _df.set_index('id', inplace=True)
        score_df = pd.concat([score_df, _df])

        score_df.to_csv(PATH_DIR + f'AD/Sea/score_df_{start_idx}_to_{M}_{covariate}_pca{pca_max}_bs_execid_{celltype}.csv')
        del(adata)
        gc.collect()

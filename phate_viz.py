import gc
import sys
import scanpy as sc
import run
import matplotlib.pyplot as plt
import numpy as np
import importlib
import pandas as pd
import random
import phate
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import anndata as ad
from sklearn.utils import shuffle
from joblib import Parallel, delayed
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression, Ridge
from scipy import stats
import os
import zipfile
import logging as log
import pyarrow as pa
import pyarrow.parquet as pq
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'


PATH_DIR = run.PATH_DIR
ADATA_PRE = run.ADATA_PRE


def _do_lr_bootstrap(X, y, iter, lr_model, **model_params):
    model = lr_model(**model_params)
    X_bootstrap, y_bootstrap = resample(X, y, replace=True, random_state=iter)
    model.fit(y, X)
    return model.coef_

def _thresh_bs(bs_coefs, true_coef, threshold=95):
    high_per = (100 + threshold)/2
    low_per = 100-high_per
    permute_perc_high = stats.scoreatpercentile(bs_coefs, per = high_per, axis=0)
    permute_perc_low = stats.scoreatpercentile(bs_coefs, per = low_per, axis=0)
    # zero loadings that are <higher_percentile or >lower_percentile
    return ((permute_perc_high>0) & (permute_perc_low<0))
    # return np.where((true_coef<permute_perc_high) | (true_coef>permute_perc_low))[0]
    
def regress_out(X, y, n_bootstrap=1, threshold=95, n_jobs=20):
    model_params = {'fit_intercept': True}
    lr_model = Ridge
    model = lr_model(**model_params)
    print('X', X.shape)
    scaler = StandardScaler(copy=False)
    y = scaler.fit_transform(y)
    print('y:', y.shape)
    model.fit(y, X)
    bs_coefs = Parallel(n_jobs=n_jobs, verbose=10)(delayed(_do_lr_bootstrap)(X, y, iter, lr_model, **model_params) for iter in range(n_bootstrap))
    mask_idx = _thresh_bs(bs_coefs, model.coef_, threshold=threshold)[:,0]
    print(f'pmi unrelated to {mask_idx.shape} genes')
    X_res = X
    X_res[:, ~mask_idx] = X[:, ~mask_idx] -  model.predict(y)[:, ~mask_idx]
    return X_res

def fit_model(data, n_comp, target='diagnosis', covariate = None, n_bootstrap=100, n_jobs = 20):
    data_PD = data[data.obs[target] > 0]
    data_ctrl = data[data.obs[target] <= 0]
    n_PD = data_PD.shape[0]
    n_ctrl = data_ctrl.shape[0]
    del(data)
    if(n_PD > n_ctrl):
        sc.pp.subsample(data_PD, fraction=n_ctrl/n_PD, random_state=0)
    else:
        sc.pp.subsample(data_ctrl, fraction=n_PD/n_ctrl, random_state=0)
    data_sub = ad.concat([data_PD, data_ctrl])
    data_shuffled = data_sub #shuffle(data_sub, random_state=0)
    X = data_shuffled.X.toarray().astype('float32')
    # if covariates:
    #     X = np.concatenate([X, data_shuffled.obs[covariates].fillna(0).values], axis=1)
    obs = data_shuffled.obs
    y = data_shuffled.obs[target]

    scaler = StandardScaler(copy=False)
    X_reg = scaler.fit_transform(X)

    if covariate:
        y_cov = data_shuffled.obs[covariate].fillna(0).values.reshape(-1, 1)
        X_reg = regress_out(X, y_cov, n_bootstrap=n_bootstrap, threshold=95, n_jobs=n_jobs)
        scaler = StandardScaler(copy=False)
        X_reg = scaler.fit_transform(X_reg)
    del(X)
    pca_comps = min(500, X_reg.shape[0])
    pca = PCA(n_components=pca_comps, copy=True, random_state=0)
    X_ = pca.fit_transform(X_reg)

    scPLS_optimal = PLSRegression(n_components=n_comp, scale=False, copy=True)
    scPLS_optimal.fit(X_, y)
    return scPLS_optimal, obs, pca, scaler, data_shuffled


def assign_component_class(data_phate, model, cell_type, n_pca, knn, save_path):
    df_phate_1 = pd.DataFrame(data_phate)
    df_phate_1['PLS_comp'] = np.argmax(np.abs(model.x_scores_), axis = 1)+1 #np.apply_along_axis(lambda x: 1 if x[0]>x[1] else 2, 0, np.array([abs(model.x_scores_[:,0]), abs(model.x_scores_[:,1])]))
    df_phate_1['cell_type'] = cell_type
    if not save_path:
        print(f'Please provide path to save data')
    log.debug('Writing PHATE data...')
    file_path = save_path+f'PHATE_comp_npca{n_pca}_knn{knn}.zip'
    log.debug(f'Writing at {file_path}')
    if not os.path.isfile(file_path):
        writer = zipfile.ZipFile(file_path, 'w')
    else:
        writer = zipfile.ZipFile(file_path, 'a')

    parquet_filename_s = f'{cell_type}.parquet'
    comp_class_bytes = df_phate_1.to_parquet()
    ## Write the bytes to the zip file
    writer.writestr(parquet_filename_s, comp_class_bytes)


def do_phate(adata, optimalMapping, covariate = 'pmi', n_pca = 500, save_path = None, celltypes = None, n_jobs = 20, knn = 5, all_cts = False):
    celltypes = celltypes if celltypes else adata.obs['cell_type'].unique().tolist()
    print(f'CELLTYPES: {celltypes}')
    if all_cts:
        data = adata
        model, _, _, _ , data_shuffled = fit_model(data, n_comp, target='diagnosis', covariate = covariate, n_bootstrap=500, n_jobs=n_jobs)
        phate_op = phate.PHATE(n_pca=n_pca, knn=knn)
        data_phate = phate_op.fit_transform(data_shuffled.X)
        assign_component_class(data_phate, model, 'all_cells', n_pca, knn, save_path)
        print(f'Finished PHATE')
    else:
        for ct in celltypes:
            n_comp = optimalMapping[ct]
            print(f'Starting celltype {ct}')
            data = adata[adata.obs['cell_type']==ct]
            model, _, _, _ , data_shuffled = fit_model(data, n_comp, target='diagnosis', covariate = covariate, n_bootstrap=500, n_jobs=n_jobs)
            n_pca = min(n_pca, data.shape[0])
            phate_op = phate.PHATE(n_pca=n_pca, knn=knn)
            data_phate = phate_op.fit_transform(data_shuffled.X)
            assign_component_class(data_phate, model, ct, n_pca, knn, save_path)
            print(f'Finished celltype {ct}')
            gc.collect()


def phate_ros(covariate = 'pmi', n_pca = 500, knn=5, save_path = PATH_DIR + 'AD/Rosmap/', celltypes = None, n_jobs=20, all_cts = False):
    adata = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad')
    adata.obs.rename(columns={'id': 'donor_id'}, inplace=True)
    do_phate(adata, run.AD_optimal_map(), covariate, n_pca, save_path, celltypes, n_jobs, knn = knn, all_cts = all_cts)


def phate_kam(covariate = 'pmi', n_pca = 500, knn=5, save_path = PATH_DIR + 'PD/Kam/', celltypes = None, n_jobs=20, all_cts = False):
    adata = sc.read_h5ad(ADATA_PRE+'kadata_pp_agematch_Oct31.h5ad')
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.gene_ids.isin(genes_ros)]
    do_phate(adata, run.PD_optimal_map(), covariate, n_pca, save_path, celltypes, n_jobs, knn = knn, all_cts = all_cts)


def phate_sm(covariate = 'pmi', n_pca = 500, knn=5, save_path = PATH_DIR + 'PD/Sm/', celltypes = None, n_jobs=20, all_cts = False):
    adata = sc.read_h5ad(ADATA_PRE+'adata_pp_Nov6_pmi.h5ad')
    adata.obs.rename(columns={'patient': 'donor_id'}, inplace=True)
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.index.isin(genes_ros)]
    do_phate(adata, run.PD_Sm_optimal_map(), covariate, n_pca, save_path, celltypes, n_jobs, knn = knn, all_cts = all_cts)


def phate_sea(covariate = 'pmi', n_pca = 500, knn=5, save_path = PATH_DIR + 'AD/Sea/', celltypes = None, n_jobs=20, all_cts = False):
    celltypes = celltypes if celltypes else ['astro', 'endo', 'opc', 'micro', 'oligo', 'l4_it', 'l5_it', 'vip', 'sncg', 'sst', 'pvalb']

    for celltype in celltypes:
        print(f'reading {ADATA_PRE}local_{celltype}.h5ad')
        adata = sc.read_h5ad(ADATA_PRE+f'local_{celltype}.h5ad')

        genesym_path = PATH_DIR+'ensemble2HGNC.csv'
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
        genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
        adata = adata[:, adata.var.index.isin(genes_ros)]
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

        do_phate(adata, run.AD_Seattle_optimal_map(), covariate, n_pca, save_path, [celltype], n_jobs, knn = knn, all_cts = all_cts)
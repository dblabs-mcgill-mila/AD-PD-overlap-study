"""
Script to save fitted PLS models.
"""


import numpy as np
import pandas as pd
import scanpy as sc

from joblib import Parallel, delayed, dump
from numpy.random import default_rng

from sklearn.utils import shuffle
from itertools import chain
import gc
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import zipfile
import logging as log
import pyarrow as pa # type: ignore
import pyarrow.parquet as pq # type: ignore
import warnings
from utils import MakeDirs
from scipy.linalg import pinv as pinv2
import anndata as ad
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import Ridge
from sklearn.utils import resample
import sys
from end2endPLS import BasePLS
import run
from scipy.stats import scoreatpercentile




class PLS_Model(BasePLS):

    def _do_lr_bootstrap(self, X, y, iter, lr_model, **model_params):
        model = lr_model(**model_params)
        X_bootstrap, y_bootstrap = resample(X, y, replace=True, random_state=iter)
        model.fit(y, X)
        return model.coef_

    def _thresh_bs(self, bs_coefs, true_coef, threshold=95):
        high_per = (100 + threshold)/2
        low_per = 100-high_per
        permute_perc_high = scoreatpercentile(bs_coefs, per = high_per, axis=0)
        permute_perc_low = scoreatpercentile(bs_coefs, per = low_per, axis=0)
        return ((permute_perc_high>0) & (permute_perc_low<0))
        
    def regress_out(self, X, y, n_bootstrap=1, threshold=95):
        print('regressing...')
        scaler = StandardScaler(copy=False)
        y = scaler.fit_transform(y)
        model_params = {'fit_intercept': True}
        lr_model = Ridge
        model = lr_model(**model_params)
        X = X.astype('float32')
        model.fit(y, X)
        bs_coefs = Parallel(n_jobs=20, verbose=10)(delayed(self._do_lr_bootstrap)(X, y, iter, lr_model, **model_params) for iter in range(n_bootstrap))
        mask_idx = self._thresh_bs(bs_coefs, model.coef_, threshold=threshold)[:,0]
        print(f'pmi unrelated to {mask_idx.shape} genes, regressing the rest out')
        X_res = X
        X_res[:, ~mask_idx] = X[:, ~mask_idx] -  model.predict(y)[:, ~mask_idx]
        return X_res, scaler, model, mask_idx
    
    def _fit_model(self, scPLS_optimal, X, y):
        try:
            check_is_fitted(scPLS_optimal)
        except:
            scPLS_optimal.fit(X, y)
        # return scPLS_optimal

    def pls_topGenes(self, data_shuffled, target, num_components, covariate=None, n_bootstrap=1000, threshold=95):
        X = data_shuffled.X.toarray().astype('float32')
        y = data_shuffled.obs[target]
        scaler = StandardScaler(copy=False)
        X_reg = scaler.fit_transform(X)
        scaler_reg = None
        cov_scalar = None 
        lr_model = None 
        mask_idx = None

        if covariate:
            y_cov = data_shuffled.obs[covariate].fillna(0).values.reshape(-1, 1)
            X_reg, cov_scalar, lr_model, mask_idx = self.regress_out(X, y_cov, n_bootstrap=n_bootstrap, threshold=threshold)
            scaler_reg = StandardScaler(copy=False)
            X_reg = scaler_reg.fit_transform(X_reg)
        del(X)
        pca_comps = min(500, X_reg.shape[0])
        pca = PCA(n_components=pca_comps, copy=True, random_state=0)
        X_ = pca.fit_transform(X_reg)
        scPLS_optimal = PLSRegression(n_components=num_components, scale=False, copy=True)
        self._fit_model(scPLS_optimal, X_, y)
        return scPLS_optimal, pca, scaler_reg, scaler, cov_scalar, lr_model, mask_idx

    @classmethod
    def pls_model_path(cls, path_dir, disease, disease_name, n_bootstrap, seed):
        return path_dir+f'{disease}/{disease_name}/PLS_Model/model_bs{n_bootstrap}_seed{seed}'
    
    def endToEndPlsAndBootstrap(self,
                                data,
                                celltype, 
                                num_components,
                                n_cells_max = None, 
                                n_bootstrap = 500,
                                target = 'diagnosis',
                                model_path = None,
                                seed = 42,
                                covariate = None,
                                cov_threshold = 95,
                                ):
        print(celltype.upper())
        sc.pp.filter_genes(data, min_cells=int(1+data.shape[0]/1000))

        print(f'maximum cells being considered for {celltype} is {n_cells_max}')
        data.var.to_csv( model_path+f'_cell{celltype}_filtered_genes.csv')

        if n_cells_max and (data.shape[0]>n_cells_max):
            sc.pp.subsample(data, n_obs=n_cells_max, random_state=seed)

        data_PD = data[data.obs[target] > 0]
        data_ctrl = data[data.obs[target] <= 0 ]
        n_PD = data_PD.shape[0]
        n_ctrl = data_ctrl.shape[0]
        del(data)
        if(n_PD > n_ctrl):
            sc.pp.subsample(data_PD, fraction=n_ctrl/n_PD, random_state=0)
        else:
            sc.pp.subsample(data_ctrl, fraction=n_PD/n_ctrl, random_state=0)
        data_sub = ad.concat([data_PD, data_ctrl])
        # data_shuffled = shuffle(data_sub, random_state=0)

        del(data_PD)
        del(data_ctrl)
        scPLS_optimal, pca, scaler_reg, scaler, cov_scalar, lr_model, mask_idx = self.pls_topGenes( data_sub.copy(), 
                                                                                                    target, 
                                                                                                    num_components, 
                                                                                                    covariate=covariate, 
                                                                                                    n_bootstrap=n_bootstrap, 
                                                                                                    threshold=cov_threshold)
        
        if model_path:
            log.debug('Writing model...')
            dump(scPLS_optimal, model_path+f'_cell{celltype}_pls.pkl')
            dump(pca, model_path+f'_cell{celltype}_pca.pkl')
            dump(scaler_reg, model_path+f'_cell{celltype}_scalerReg.pkl')
            dump(scaler, model_path+f'_cell{celltype}_scaler.pkl')
            dump(cov_scalar, model_path+f'_cell{celltype}_cov_scalar.pkl')
            dump(lr_model, model_path+f'_cell{celltype}_lr_model.pkl')
            np.save(model_path+f'_cell{celltype}_mask_idx.npy', mask_idx)
            data_sub.write_h5ad(model_path+f'adata_cell{celltype}.h5ad')

    def run(self, 
            adata,
            n_cells_max = None,
            n_bootstrap = 500,
            n_jobs = 6, 
            seed = 42,
            celltypes = None,
            covariate = None):
        model_path = PLS_Model.pls_model_path(self.path_dir, 
                                                self.disease, 
                                                self.disease_name, 
                                                n_bootstrap, 
                                                seed) 
        MakeDirs.make_dirs(model_path, include_end=False)
        celltypes = celltypes if celltypes else self.optimalMapping.keys()
        n_jobs = n_jobs if n_jobs else len(celltypes)
        _ = Parallel(n_jobs = n_jobs, verbose=5)(delayed(self.endToEndPlsAndBootstrap)(adata[adata.obs.cell_type == celltype].copy(),
                                                                                               celltype,
                                                                                               self.optimalMapping[celltype],
                                                                                               n_cells_max=n_cells_max, 
                                                                                               n_bootstrap=n_bootstrap, 
                                                                                               model_path=model_path, 
                                                                                               seed=seed,
                                                                                               covariate=covariate) for celltype in celltypes)
            
        gc.collect()


def run_rosmap(covariate = 'pmi', n_jobs = 20):
    optimalMapping = run.AD_optimal_map()
    adata = sc.read_h5ad(run.ADATA_PRE+'mathys19_pp_filtered_June21.h5ad')
    path_dir = run.PATH_DIR
    print('running PLS Rosmap')
    gemtxad = PLS_Model('AD', optimalMapping, path_dir,  'Rosmap')
    gemtxad.run(adata, n_cells_max = 40000, n_bootstrap = 500, n_jobs=n_jobs, covariate=covariate)


def run_kamath(covariate = 'pmi', n_jobs = 20):
    adata = sc.read_h5ad(run.ADATA_PRE+'kadata_pp_agematch_Oct31.h5ad')
    genes_ros = sc.read_h5ad(run.ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.gene_ids.isin(genes_ros)]
    optimalMapping = run.PD_optimal_map()
    path_dir = run.PATH_DIR
    print('running PLS Kamath')
    gemtxad = PLS_Model('PD', optimalMapping, path_dir,  'Kamath')
    gemtxad.run(adata, n_cells_max = 40000, n_bootstrap = 500, n_jobs=n_jobs, covariate=covariate)


def run_smijac(covariate = 'pmi', n_jobs = 20):
    print('Running PLS PD Smijac')
    adata = sc.read_h5ad(run.ADATA_PRE+'adata_pp_Nov6_pmi.h5ad')
    genes_ros = sc.read_h5ad(run.ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.index.isin(genes_ros)]
    optimalMapping = run.PD_Sm_optimal_map()
    path_dir = run.PATH_DIR
    print('running PLS Smijac')
    gemtxad = PLS_Model('PD', optimalMapping, path_dir,  'Sm')
    gemtxad.run(adata, n_cells_max = 40000, n_bootstrap = 500, n_jobs=n_jobs, covariate=covariate)


def run_seattle(covariate = 'pmi', n_jobs = 20):
    optimalMapping = run.AD_Seattle_optimal_map()
    ADATA_PRE = run.ADATA_PRE
    celltypes = ['astro', 'opc', 'micro', 'oligo', 'l4_it', 'l5_it', 'vip', 'sncg', 'sst', 'pvalb'] #'endo', 
    for celltype in celltypes:
        print(f'reading {ADATA_PRE}local_{celltype}.h5ad')
        adata = sc.read_h5ad(ADATA_PRE+f'local_{celltype}.h5ad')

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
        genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
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

        gemtxad = PLS_Model('AD', optimalMapping, run.PATH_DIR,  'Sea')
        gemtxad.run(adata, n_cells_max = 40000, n_bootstrap = 500, n_jobs=n_jobs, covariate=covariate)
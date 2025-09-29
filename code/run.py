"""
MAIN RUN script

Contains examples to run end-to-end analysis. 
PLS -> Correlation / Gene set enrichment analysis.
"""


import end2endPLS as pls
import correlation_base as co
from collections import OrderedDict
import scanpy as sc
import gc
from joblib import Parallel, delayed
from utils import convert_excel_sheets_to_parquet_permute, convert_excel_sheets_to_parquet
from scipy.stats import kendalltau
import gseaAnalysis as ga
import pandas as pd

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import csv
from itertools import product
from utils import piecewise_weightedtau as pwt


PATH_DIR = '/host/socrates/pool4_z2/anweshab/OverlapAnalysis/GenePermute2/'
ADATA_PRE = '/host/socrates/pool3_z0/anwesha/OverlapAnalysis/adatas/'

def AD_optimal_map():
    return OrderedDict({ 
                'Ex': 2,
                'Oli': 2,
                'In': 2,
                'Mic': 2,
                'Opc': 2,
                'Ast': 2,
                'End': 2,
                'Per': 2,
                })


def PD_optimal_map():
    return OrderedDict({ 
                'SOX6': 2,
                'CALB1': 3,
                'Ependyma': 2,
                'Microglia': 2,
                'Macrophage': 3,
                'Astrocyte': 2,
                'Endothelial': 3,
                'OPC': 2,
                'Excitatory neuron': 2,
                'Inhibitory neuron': 2,
                'Oligodendrocyte': 2,
                })


def Lung_optimal_map():
    n_comp = 2
    return OrderedDict({
        'Myeloid': n_comp,
        'Lymphoid': n_comp,
        'Endothelial': n_comp,
        'Stromal': n_comp,
        'Epithelial': n_comp,
        'Multiplet': n_comp})

def PD_Sm_optimal_map():
    return OrderedDict({'Oligodendrocytes': 2,
                        'Excitatory neuron': 2,
                        'Microglia': 2,
                        'OPC': 2,
                        'Astrocyte': 3,
                        'Endothelial cells': 2,
                        'Pericytes': 2,
                        'Inhibitory neuron': 3,
                        'Ependymal': 2,
                        'GABA': 2})


def AD_Seattle_optimal_map():
    return {'astro':3, 'endo':2, 'opc':2, 'micro':2, 'oligo':2, 'l4_it':3, 'l5_it':3, 'vip':3, 'sncg':2, 'sst':2, 'pvalb':3}


def PD_Mart_optimal_map():
    return {'Micro':2, 'Neurons':2, 'Astro':2, 'OPC':2, 'Oligo':2, 'VC':2, 'T cells':2}

def run_ad(path_dir, num_perm=100, permute_factor = 'label', start_idx=0, covariate='pmi', do_permute=True, do_pls=True):
    optimalMapping = AD_optimal_map()
    n_jobs = 10
    adata = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad')
    if do_pls:
        print('running PLS Rosmap')
        gemtxad = pls.GeneListFromExpressionMtx('AD',optimalMapping, path_dir,  'Rosmap')
        gemtxad.run(adata, n_cells_max = 40000, n_bootstrap = 500, n_jobs=n_jobs, covariate=covariate)
    
    gc.collect()
    if do_permute:
        print('running permutation Rosmap')
        glPerm = pls.PermutationGeneList('AD', optimalMapping, path_dir,  'Rosmap')
        glPerm.run(adata, num_perm=num_perm, n_jobs=n_jobs, n_cells_max = 40000, start_idx=start_idx,  permute_factor=permute_factor, covariate=covariate)


def run_ad_2(path_dir, num_perm=100, permute_factor = 'label', start_idx=0, covariate='pmi', do_permute=True, do_pls=True):
    optimalMapping = AD_optimal_map()
    n_jobs = 10
    adata = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad')
    if do_pls:
        print('running PLS Rosmap 2')
        gemtxad = pls.GeneListFromExpressionMtx('AD',optimalMapping, path_dir,  'Rosmap_2')
        gemtxad.run(adata, n_cells_max = 40000, n_bootstrap = 500, n_jobs=n_jobs, covariate=covariate, seed = 0)
    
    gc.collect()
    if do_permute:
        print('running permutation Rosmap')
        glPerm = pls.PermutationGeneList('AD', optimalMapping, path_dir,  'Rosmap_2')
        glPerm.run(adata, num_perm=num_perm, n_jobs=n_jobs, n_cells_max = 40000, start_idx=start_idx,  permute_factor=permute_factor, covariate=covariate, seed=0)


def run_pd(path_dir, num_perm=100, permute_factor = 'label', start_idx=0, covariate = 'pmi', do_permute=True, do_pls=True):
    adata = sc.read_h5ad(ADATA_PRE+'kadata_pp_agematch_Oct31.h5ad')
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.gene_ids.isin(genes_ros)]
    optimalMapping = PD_optimal_map()
    celltypes = list(optimalMapping.keys())
    adata = adata[adata.obs.cell_type.isin(celltypes)]
    n_jobs =10
    if do_pls:
        print('running PLS KAM')
        gemtxpd = pls.GeneListFromExpressionMtx('PD',optimalMapping, path_dir, 'Kam')
        gemtxpd.run(adata, n_cells_max = 40000, n_bootstrap = 500, n_jobs=n_jobs, covariate=covariate, celltypes=celltypes)
    del(adata)
    gc.collect()
    if do_permute:
        print('running permutation KAM')
        adata = sc.read_h5ad(ADATA_PRE+'kadata_pp_agematch_Oct31.h5ad')
        adata = adata[:, adata.var.gene_ids.isin(genes_ros)]
        optimalMapping = PD_optimal_map()
        celltypes = list(optimalMapping.keys())[-1:]
        adata = adata[adata.obs.cell_type.isin(celltypes)].copy()
        glPerm = pls.PermutationGeneList('PD', optimalMapping, path_dir, 'Kam')
        glPerm.run(adata, num_perm=num_perm, n_jobs=20, n_cells_max = 80000, start_idx=start_idx, permute_factor=permute_factor, covariate=covariate, celltypes=celltypes)


def run_pd_2(path_dir, num_perm=100, permute_factor = 'label', start_idx=0, covariate = 'pmi', do_permute=True, do_pls=True):
    adata = sc.read_h5ad(ADATA_PRE+'kadata_pp_agematch_Oct31.h5ad')
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.gene_ids.isin(genes_ros)]
    optimalMapping = PD_optimal_map()
    celltypes = list(optimalMapping.keys())
    adata = adata[adata.obs.cell_type.isin(celltypes)]
    n_jobs =10
    if do_pls:
        print('running PLS KAM 2')
        gemtxpd = pls.GeneListFromExpressionMtx('PD',optimalMapping, path_dir, 'Kam_2')
        gemtxpd.run(adata, n_cells_max = 40000, n_bootstrap = 500, n_jobs=n_jobs, covariate=covariate, celltypes=celltypes, seed=0)
    del(adata)
    gc.collect()
    if do_permute:
        print('running permutation KAM 2')
        adata = sc.read_h5ad(ADATA_PRE+'kadata_pp_agematch_Oct31.h5ad')
        adata = adata[:, adata.var.gene_ids.isin(genes_ros)]
        optimalMapping = PD_optimal_map()
        celltypes = list(optimalMapping.keys())
        adata = adata[adata.obs.cell_type.isin(celltypes)].copy()
        glPerm = pls.PermutationGeneList('PD', optimalMapping, path_dir,  'Kam_2')
        glPerm.run(adata, num_perm=num_perm, n_jobs=20, n_cells_max = 80000, start_idx=start_idx, permute_factor=permute_factor, covariate=covariate, celltypes=celltypes, seed=0)


def run_lewy(path_dir, num_perm=100, permute_factor = 'label', start_idx=0, covariate = 'pmi', do_permute=True, do_pls=True):
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    n_jobs =20
    if do_pls:
        print('running PLS LEWY')
        adata = sc.read_h5ad(ADATA_PRE+'kadata_Lewy_pp_age_matched.h5ad')
        adata = adata[:, adata.var.gene_ids.isin(genes_ros)]
        optimalMapping = PD_optimal_map()
        celltypes = list(optimalMapping.keys())
        adata = adata[adata.obs.cell_type.isin(celltypes)]
        gemtxpd = pls.GeneListFromExpressionMtx('Lewy',optimalMapping, path_dir, 'Kam')
        gemtxpd.run(adata, n_cells_max = 40000, n_bootstrap = 500, n_jobs=n_jobs, covariate=covariate, celltypes=celltypes)
        del(adata)
    gc.collect()
    if do_permute:
        print('running permutation LEWY')
        adata = sc.read_h5ad(ADATA_PRE+'kadata_Lewy_pp_age_matched.h5ad')
        adata = adata[:, adata.var.gene_ids.isin(genes_ros)]
        optimalMapping = PD_optimal_map()
        celltypes = list(optimalMapping.keys())
        adata = adata[adata.obs.cell_type.isin(celltypes)].copy()
        glPerm = pls.PermutationGeneList('Lewy', optimalMapping, path_dir,  'Kam')
        glPerm.run(adata, num_perm=num_perm, n_jobs=n_jobs, n_cells_max = 80000, start_idx=start_idx, permute_factor=permute_factor, covariate=covariate, celltypes=celltypes)


def run_lung(path_dir,num_perm=100, permute_factor = 'label', start_idx=0, do_permute=True, do_pls = True):
    adata = sc.read_h5ad(ADATA_PRE+'adata_pp_Jun30.h5ad')
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.index.isin(genes_ros)]
    optimalMapping = Lung_optimal_map()
    if do_pls:
        gemtxpd = pls.GeneListFromExpressionMtx('Lung',optimalMapping, path_dir,  'Kaminski')
        gemtxpd.run(adata, n_cells_max = 40000, n_bootstrap = 500, n_jobs = 20)
    gc.collect()
    if do_permute:
        print('running kaminski')
        glPerm = pls.PermutationGeneList('Lung', optimalMapping, path_dir,  'Kaminski')
        glPerm.run(adata, num_perm=num_perm, n_jobs=20, n_cells_max = 80000, permute_factor=permute_factor, start_idx=start_idx)


def run_pd_sm(path_dir, num_perm=100, permute_factor = 'label', start_idx=0, covariate='pmi', do_permute=True, do_pls = True):
    print('Running PLS PD Smijac')
    adata = sc.read_h5ad(ADATA_PRE+'adata_pp_Nov6_pmi.h5ad')
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.index.isin(genes_ros)]
    optimalMapping = PD_Sm_optimal_map()
    n_jobs = 10
    
    if do_pls:
        print('running PLS SM')
        gemtxpd = pls.GeneListFromExpressionMtx('PD',optimalMapping, path_dir,  'Sm')
        gemtxpd.run(adata, n_cells_max = 40000, n_bootstrap = 500, n_jobs=n_jobs, covariate=covariate)
    gc.collect()
    if do_permute:
        print('running permutaitons SM')
        glPerm = pls.PermutationGeneList('PD', optimalMapping, path_dir,  'Sm')
        glPerm.run(adata, num_perm=num_perm,n_jobs=n_jobs, n_cells_max = None, permute_factor=permute_factor, start_idx=start_idx, covariate=covariate)


def run_pd_mart(path_dir, num_perm=100, permute_factor = 'label', start_idx=0, covariate='pmi', do_permute=True, do_pls = True):
    print('Running PLS PD Mart')
    adata = sc.read_h5ad(ADATA_PRE+'adata_mart.h5ad')
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.index.isin(genes_ros)]
    optimalMapping = PD_Mart_optimal_map()
    n_jobs = 10
    
    if do_pls:
        print('running PLS MART')
        gemtxpd = pls.GeneListFromExpressionMtx('PD',optimalMapping, path_dir,  'Mart')
        gemtxpd.run(adata, n_cells_max = None, n_bootstrap = 500, n_jobs=n_jobs, covariate=covariate)

    gc.collect()
    if do_permute:
        print('running permutaitons MART')
        glPerm = pls.PermutationGeneList('PD', optimalMapping, path_dir,  'Mart')
        glPerm.run(adata, num_perm=num_perm,n_jobs=n_jobs, n_cells_max = None, permute_factor=permute_factor, start_idx=start_idx, covariate=covariate)


def run_ad_sea(path_dir,num_perm=100, permute_factor = 'label', start_idx=0,covariate='pmi', do_permute=True, do_pls=True ):
    print('Running PLS AD SEA', do_pls)
    optimalMapping = AD_Seattle_optimal_map()
    n_jobs = 10

    celltypes = ['sncg', 'sst', 'pvalb', 'astro', 'endo', 'opc', 'micro', 'oligo', 'l4_it', 'l5_it', 'vip', 'sncg', 'sst', 'pvalb']
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
        diagnosis_map = {'normal':-1, 'dementia':1}
        adata.obs['diagnosis'] = adata.obs.disease.map(lambda x: diagnosis_map[x])
        adata.obs['diagnosis'] = adata.obs['diagnosis'].astype('int64')
        adata.obs['cell_type'] = celltype
        adata = adata[~adata.obs['Age at death'].isin(['Less than 65 years old', '65 to 77 years old'])]
        pmi_map = {'3.2 to 5.9 hours': 3.5, '5.9 to 8.7 hours': 7.5, '8.7 to 11.4 hours': 10.5, 'Reference':float('nan')}
        adata.obs['pmi'] = adata.obs.PMI.map(lambda x: pmi_map[x]).astype('float32')
        adata.obs.drop(columns='PMI', inplace=True)
        if do_pls:
            print('running pls SEA')
            gemtxad = pls.GeneListFromExpressionMtx('AD',optimalMapping, path_dir,  'Sea')
            gemtxad.run(adata, n_cells_max = 40000, n_bootstrap = 500, celltypes=[celltype], n_jobs = n_jobs, covariate=covariate)
        
        gc.collect()
        if do_permute:
            print('running permutation SEA')
            glPerm = pls.PermutationGeneList('AD', optimalMapping, path_dir,  'Sea')
            glPerm.run(adata, num_perm=num_perm, celltypes= [celltype], n_jobs=n_jobs, n_cells_max = 40000, permute_factor=permute_factor, start_idx=start_idx, covariate=covariate)

        del(adata)
        gc.collect()


def run_split_half_pd(path_dir, idx = 0, covariate='pmi'):
    print(f'Running PD {idx} first half')
    adata = sc.read_h5ad(ADATA_PRE+'kadata_pp_agematch_Oct31.h5ad')
    
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.gene_ids.isin(genes_ros)]

    sampled_data = sc.pp.subsample(adata, fraction=0.5, random_state=0+idx, copy=True)
    
    gemtxpd = pls.GeneListFromExpressionMtx('PD_split',PD_optimal_map(), path_dir,  f'Kam_{idx}_1')
    gemtxpd.run(sampled_data, n_cells_max = 40000, n_bootstrap = 1, covariate=covariate)

    sampled_other = adata[list(set(adata.obs.index.to_list()) - set(sampled_data.obs.index.to_list()))]
    del(sampled_data)
    del(adata)

    print(f'Running PD {idx} first half')
    gemtxpd = pls.GeneListFromExpressionMtx('PD_split', PD_optimal_map(), path_dir,  f'Kam_{idx}_2')
    gemtxpd.run(sampled_other, n_cells_max = 40000, n_bootstrap = 1, covariate=covariate)

    # glPerm = pls.PermutationGeneList('PD_split', PD_optimal_map(), path_dir,  f'Kam_2_{idx}')
    # glPerm.run(sampled_other, num_perm=100)


def run_split_half_ad(path_dir, idx = 0, covariate='pmi'):
    print(f'SPLIT HALF {idx}')
    n_jobs = 5
    optimalMapping = AD_optimal_map()
    adata = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad')
     
    sampled_data = sc.pp.subsample(adata, fraction=0.5, random_state=0+idx, copy=True)
    
    gemtxpd = pls.GeneListFromExpressionMtx('AD_split',optimalMapping, path_dir,  f'Rosmap_{idx}_1')
    gemtxpd.run(sampled_data, n_cells_max = 40000, n_bootstrap = 1,  n_jobs=n_jobs, covariate=covariate)

    # glPerm = pls.PermutationGeneList('AD_split', optimalMapping, path_dir,  f'Rosmap_{idx}_1')
    # glPerm.run(sampled_data, num_perm=100, n_jobs=n_jobs)

    sampled_other = adata[list(set(adata.obs.index.to_list()) - set(sampled_data.obs.index.to_list()))]
    del(sampled_data)
    del(adata)

    gemtxpd = pls.GeneListFromExpressionMtx('AD_split',optimalMapping, path_dir,  f'Rosmap_{idx}_2')
    gemtxpd.run(sampled_other, n_cells_max = 40000, n_bootstrap = 1, n_jobs=n_jobs, covariate=covariate)

    # glPerm = pls.PermutationGeneList('AD_split', optimalMapping, path_dir,  f'Rosmap_{idx}_2')
    # glPerm.run(sampled_other, num_perm=100, n_jobs=n_jobs)


def _run_pls_split(idx, run_ad = True, run_pd = True):
        print('RUNNING IDX ', idx)
        if run_ad:
            run_split_half_ad(PATH_DIR, idx)
        if run_pd:
            run_split_half_pd(PATH_DIR, idx)
        print('FINISHED IDX ', idx)


def run_pls_split(start_idx=0, end_idx=1000, n_jobs=10,run_ad = True, run_pd = True):
    Parallel(n_jobs=n_jobs)(delayed(_run_pls_split)(idx, run_ad, run_pd) for idx in range(start_idx, end_idx))


def run_pls(start_idx=0, end_idx=1000, permute_factor='label', do_permute=True, do_pls = True):
    '''
    start_idx: start index for split half analysis
    end_idx: end idx for split half analysis
    '''
    path_dir = PATH_DIR
    
    print('running ad rosmap')
    run_ad(path_dir, start_idx=start_idx, num_perm=end_idx, permute_factor=permute_factor, do_permute=do_permute, do_pls=do_pls)
    gc.collect()

    print('running pd kamath')
    run_pd(path_dir, start_idx=start_idx, num_perm=end_idx, permute_factor=permute_factor, do_permute=do_permute, do_pls=do_pls)
    gc.collect()


def run_pls2(start_idx=0, end_idx=1000, permute_factor='label', do_permute=True, do_pls=True):
    '''
    start_idx: start index for split half analysis
    end_idx: end idx for split half analysis
    '''
    path_dir = PATH_DIR
    
    print('running sea')
    run_ad_sea(path_dir, start_idx=start_idx, num_perm=end_idx, permute_factor=permute_factor, do_permute=do_permute, do_pls=do_pls)
    
    print('running sm')
    run_pd_sm(path_dir, start_idx=start_idx, num_perm=end_idx, permute_factor=permute_factor, do_permute=do_permute, do_pls=do_pls)

def run_corr2(start_idx=0, end_idx=1000, thresh_high_per=97.5, doOr = False, corr_colname='loading', corr_metric = 'kt'):
    '''
    start_idx: start index for split half analysis
    end_idx: end idx for split half analysis
    '''
    path_dir = PATH_DIR
    if corr_metric=='kt':
        corr_metric=kendalltau
    elif corr_metric == 'pwt':
        corr_metric = pwt
    
    print('PD AD')
    corr_obj = co.PLSCorrelation(corr_metric, 'PD', 'AD', 'Sm', 'Sea', path_dir, doOR = doOr, corr_colname=corr_colname)
    corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per)

def run_corr_lung(start_idx=0, end_idx=1000, thresh_high_per=97.5, doOr = False, corr_colname='loading', corr_metric = 'kt'):
    '''
    start_idx: start index for split half analysis
    end_idx: end idx for split half analysis
    '''
    path_dir = PATH_DIR
    if corr_metric=='kt':
        corr_metric=kendalltau
    elif corr_metric == 'pwt':
        corr_metric = pwt
    
    print('PD Lung')
    corr_obj = co.PLSCorrelation(corr_metric, 'Lung', 'PD', 'Kaminski', 'Kam', path_dir, doOR = doOr, corr_colname=corr_colname)
    corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per)

    print('AD Lung')
    corr_obj = co.PLSCorrelation(corr_metric, 'Lung', 'AD', 'Kaminski', 'Rosmap', path_dir, doOR = doOr, corr_colname=corr_colname)
    corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per)


def run_corr(start_idx=0, end_idx=1000, thresh_high_per=97.5, doOr = False, corr_colname='loading', corr_metric = 'kt'):
    '''
    start_idx: start index for split half analysis
    end_idx: end idx for split half analysis
    '''
    path_dir = PATH_DIR
    if corr_metric=='kt':
        corr_metric=kendalltau
    elif corr_metric == 'pwt':
        corr_metric = pwt
    
    # print('PD AD')
    # corr_obj = co.PLSCorrelation(corr_metric, 'PD', 'AD', 'Sm', 'Sea', path_dir, doOR = doOr, corr_colname=corr_colname)
    # corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per)

    print('PD AD')
    corr_obj = co.PLSCorrelation(corr_metric, 'PD', 'AD', 'Kam', 'Rosmap', path_dir, doOR = doOr, corr_colname=corr_colname)
    corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per)

    # print('Lung AD')
    # corr_obj = co.PLSCorrelation(corr_metric, 'Lung', 'AD', 'Kaminski', 'Rosmap', path_dir, doOR = doOr, corr_colname=corr_colname)
    # corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per)

    # print('Lung PD')
    # corr_obj = co.PLSCorrelation(corr_metric, 'Lung', 'PD', 'Kaminski', 'Kam', path_dir, doOR = doOr, corr_colname=corr_colname)
    # corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per)
           
    # print('Lung AD')
    # corr_obj = co.PLSCorrelation(corr_metric,  'Lung', 'AD', 'Kaminski', 'Sea', path_dir, doOR = doOr, corr_colname=corr_colname)
    # corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per)

    # print('Lung PD')
    # corr_obj = co.PLSCorrelation(corr_metric, 'Lung', 'PD', 'Kaminski', 'Sm', path_dir, doOR = doOr, corr_colname=corr_colname)
    # corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per)

def run_corr_ad(start_idx=0, end_idx=1000, thresh_high_per=97.5, doOr = False, corr_colname='loading', corr_metric = 'kt'):
    '''
    start_idx: start index for split half analysis
    end_idx: end idx for split half analysis
    '''
    path_dir = PATH_DIR
    if corr_metric=='kt':
        corr_metric=kendalltau
    elif corr_metric == 'pwt':
        corr_metric = pwt
    
    print('AD AD')
    corr_obj = co.PLSCorrelation(corr_metric, 'AD', 'AD', 'Rosmap', 'Rosmap_2', path_dir, doOR = doOr, corr_colname=corr_colname)
    corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per, seed2=0)


def run_corr_adad_pdpd(start_idx=0, end_idx=1000, thresh_high_per=97.5, doOr = False, corr_colname='loading', corr_metric = 'kt'):
    '''
    start_idx: start index for split half analysis
    end_idx: end idx for split half analysis
    '''
    path_dir = PATH_DIR
    if corr_metric=='kt':
        corr_metric=kendalltau
    elif corr_metric == 'pwt':
        corr_metric = pwt
    
    print('AD AD')
    corr_obj = co.PLSCorrelation(corr_metric, 'AD', 'AD', 'Sea', 'Rosmap', path_dir, doOR = doOr, corr_colname=corr_colname)
    corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per)

    print('PD PD')
    corr_obj = co.PLSCorrelation(corr_metric, 'PD', 'PD', 'Sm', 'Kam', path_dir, doOR = doOr, corr_colname=corr_colname)
    corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per)


def run_corr_pd(start_idx=0, end_idx=1000, thresh_high_per=97.5, doOr = False, corr_colname='loading', corr_metric = 'kt'):
    '''
    start_idx: start index for split half analysis
    end_idx: end idx for split half analysis
    '''
    path_dir = PATH_DIR
    if corr_metric=='kt':
        corr_metric=kendalltau
    elif corr_metric == 'pwt':
        corr_metric = pwt
    
    print('PD PD')
    corr_obj = co.PLSCorrelation(corr_metric, 'PD', 'PD', 'Kam', 'Kam_2', path_dir, doOR = doOr, corr_colname=corr_colname)
    corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per, seed2=0)


def run_corr_sea_mart(start_idx=0, end_idx=1000, thresh_high_per=97.5, doOr = False, corr_colname='loading', corr_metric = 'kt'):
    '''
    start_idx: start index for split half analysis
    end_idx: end idx for split half analysis
    '''
    path_dir = PATH_DIR
    if corr_metric=='kt':
        corr_metric=kendalltau
    elif corr_metric == 'pwt':
        corr_metric = pwt
    
    print('PD AD')
    corr_obj = co.PLSCorrelation(corr_metric, 'PD', 'AD', 'Mart', 'Sea', path_dir, doOR = doOr, corr_colname=corr_colname)
    corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per)


def run_corr_lewy(start_idx=0, end_idx=1000, thresh_high_per=97.5, doOr = False, corr_colname='loading', corr_metric = 'kt'):
    '''
    start_idx: start index for split half analysis
    end_idx: end idx for split half analysis
    '''
    path_dir = PATH_DIR
    if corr_metric=='kt':
        corr_metric=kendalltau
    elif corr_metric == 'pwt':
        corr_metric = pwt
    
    # print('Lewy AD')
    # corr_obj = co.PLSCorrelation(corr_metric, 'Lewy', 'AD', 'Kam', 'Rosmap', path_dir, doOR = doOr, corr_colname=corr_colname)
    # corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per)

    print('Lewy PD')
    corr_obj = co.PLSCorrelation(corr_metric, 'Lewy', 'PD', 'Kam', 'Kam', path_dir, doOR = doOr, corr_colname=corr_colname)
    corr_obj.run(num_perm=end_idx, n_bootstrap=500, n_jobs=20, thresh_high_per=thresh_high_per)


def run_split_corr(start_idx = 0, end_idx = 1000, doOR=False):
    print('corr PD_split AD_split')
    path_dir = PATH_DIR
    corr_obj = co.PLSInterSplitCorrelationUtils(kendalltau, 'PD', 'AD', f'Kam', f'Rosmap', path_dir, num_split = end_idx, doOR = doOR, loading_thresh=0)
    corr_obj.run(n_bootstrap=1, start_idx = start_idx)

    return corr_obj.do_splithalves_correlation()

# GENE_SET_LIST = ['GO_Biological_Process_2021', 'WikiPathway_2021_Human', 'Panther_2016', 'Reactome_2016', 'KEGG_2021_Human']

GENE_SET_LIST = ['GO_Biological_Process_2023','GO_Molecular_Function_2023', 'GO_Cellular_Component_2023']


def thresh_loadings_and_gsea_pd(high_percentile = 95, perm_num=1000, permute_factor='label', error_log=None, n_bootstrap=100, n_jobs=20, gene_sets = GENE_SET_LIST, do_thresh=True, do_gsea=True, gs_prefix = None):

    ########################################################################################################################
    ## IMPORTANT : using 1 cpu to run because otherwise the file system to store object gets messed up. 
    ########################################################################################################################
    
    if do_thresh:
        print('Thresholding PD...')
        pls_obj = pls.ThresholdPredictiveComponents('PD', PD_optimal_map(), PATH_DIR, 'Kam', high_percentile=high_percentile,permute_factor=permute_factor )
        pls_obj.run(perm_num=perm_num, n_bootstrap=n_bootstrap, n_jobs=n_jobs)
    
    if do_gsea:
        print('GSEA PD...')
        gsea_obj = ga.GSEA('PD', PD_optimal_map(), PATH_DIR, 'Kam')
        gsea_obj.run(n_jobs = 1, thresh_high_per = high_percentile,permute_factor=permute_factor, n_bootstrap=n_bootstrap, gene_set_list=gene_sets, gs_prefix=gs_prefix)


def thresh_loadings_and_gsea_pd_2(high_percentile = 95, perm_num=1000, permute_factor='label', error_log=None, n_bootstrap=100, n_jobs=1, gene_sets = GENE_SET_LIST, do_thresh=True, do_gsea=True, gs_prefix = None):

    ########################################################################################################################
    ## IMPORTANT : using 1 cpu to run because otherwise the file system to store object gets messed up. 
    ########################################################################################################################
    
    if do_thresh:
        print('Thresholding PD 2...')
        pls_obj = pls.ThresholdPredictiveComponents('PD', PD_optimal_map(), PATH_DIR, 'Kam_2', high_percentile=high_percentile,permute_factor=permute_factor )
        pls_obj.run(perm_num=perm_num, n_bootstrap=n_bootstrap, n_jobs=n_jobs, seed=0)
    
    if do_gsea:
        print('GSEA PD 2...')
        gsea_obj = ga.GSEA('PD', PD_optimal_map(), PATH_DIR, 'Kam_2')
        gsea_obj.run(n_jobs = 1, thresh_high_per = high_percentile,permute_factor=permute_factor, n_bootstrap=n_bootstrap, gene_set_list=gene_sets, gs_prefix=gs_prefix, seed=0)


def thresh_loadings_and_gsea_lewy(high_percentile = 95, perm_num=1000, permute_factor='label', error_log=None, n_bootstrap=100, n_jobs=20, gene_sets = GENE_SET_LIST, do_thresh=True, do_gsea=True, gs_prefix = None):

    ########################################################################################################################
    ## IMPORTANT : using 1 cpu to run because otherwise the file system to store object gets messed up. 
    ########################################################################################################################
    
    if do_thresh:
        print('Thresholding Lewy...')
        pls_obj = pls.ThresholdPredictiveComponents('Lewy', PD_optimal_map(), PATH_DIR, 'Kam', high_percentile=high_percentile,permute_factor=permute_factor )
        pls_obj.run(perm_num=perm_num, n_bootstrap=n_bootstrap, n_jobs=n_jobs)
    
    if do_gsea:
        print('GSEA Lewy...')
        gsea_obj = ga.GSEA('Lewy', PD_optimal_map(), PATH_DIR, 'Kam')
        gsea_obj.run(n_jobs = 1, thresh_high_per = high_percentile,permute_factor=permute_factor, n_bootstrap=n_bootstrap, gene_set_list=gene_sets, gs_prefix=gs_prefix)


def thresh_loadings_and_gsea_lung(high_percentile = 95, perm_num=1000, permute_factor='label', error_log=None, n_bootstrap=100, n_jobs=1, gene_sets = GENE_SET_LIST, do_thresh=True, do_gsea=True):
    try:
        if do_thresh:
            print('Thresholding Lung...')
            pls_obj = pls.ThresholdPredictiveComponents('Lung', Lung_optimal_map(), PATH_DIR, 'Kaminski', high_percentile=high_percentile,permute_factor=permute_factor )
            pls_obj.run(perm_num=perm_num, n_bootstrap=n_bootstrap, n_jobs=n_jobs)

        if do_gsea:     
            print('GSEA Lung...')
            gsea_obj = ga.GSEA('Lung', Lung_optimal_map(), PATH_DIR, 'Kaminski')
            gsea_obj.run(n_jobs = 1, thresh_high_per = high_percentile,permute_factor=permute_factor, n_bootstrap=n_bootstrap, gene_set_list=gene_sets)
    except Exception as e:
        print(e)


def thresh_loadings_and_gsea_ros(high_percentile = 95, perm_num=1000, permute_factor='label', error_log=None, n_bootstrap=100, n_jobs=1, gene_sets = GENE_SET_LIST, do_thresh=True, do_gsea=True, gs_prefix=None):
    try:
        if do_thresh:
            print('Thresholding AD...')
            pls_obj = pls.ThresholdPredictiveComponents('AD', AD_optimal_map(), PATH_DIR, 'Rosmap', high_percentile=high_percentile,permute_factor=permute_factor )
            pls_obj.run(perm_num=perm_num, n_bootstrap=n_bootstrap, n_jobs=n_jobs)
        if do_gsea:
            print('GSEA AD...')
            gsea_obj = ga.GSEA('AD', AD_optimal_map(), PATH_DIR, 'Rosmap')
            gsea_obj.run(n_jobs = 1, thresh_high_per = high_percentile, permute_factor=permute_factor, n_bootstrap=n_bootstrap, gene_set_list=gene_sets, gs_prefix=gs_prefix)
    except Exception as e:
        print(e)

def thresh_loadings_and_gsea_ros_2(high_percentile = 95, perm_num=1000, permute_factor='label', error_log=None, n_bootstrap=100, n_jobs=1, gene_sets = GENE_SET_LIST, do_thresh=True, do_gsea=True, gs_prefix=None):
    try:
        if do_thresh:
            print('Thresholding AD 2...')
            pls_obj = pls.ThresholdPredictiveComponents('AD', AD_optimal_map(), PATH_DIR, 'Rosmap_2', high_percentile=high_percentile,permute_factor=permute_factor )
            pls_obj.run(perm_num=perm_num, n_bootstrap=n_bootstrap, n_jobs=n_jobs, seed = 0 )
        if do_gsea:
            print('GSEA AD 2...')
            gsea_obj = ga.GSEA('AD', AD_optimal_map(), PATH_DIR, 'Rosmap_2')
            gsea_obj.run(n_jobs = 1, thresh_high_per = high_percentile, permute_factor=permute_factor, n_bootstrap=n_bootstrap, gene_set_list=gene_sets, gs_prefix=gs_prefix, seed=0)
    except Exception as e:
        print(e)

def thresh_loadings_and_gsea_sm(high_percentile = 95, perm_num=1000, permute_factor='label', error_log=None, n_bootstrap=100, n_jobs=20, gene_sets = GENE_SET_LIST, do_thresh=True, do_gsea=True, gs_prefix=None):
    try:
        if do_thresh:
            print('Thresholding Smijac...')
            pls_obj = pls.ThresholdPredictiveComponents('PD', PD_Sm_optimal_map(), PATH_DIR, 'Sm', high_percentile=high_percentile,permute_factor=permute_factor )
            pls_obj.run(perm_num=perm_num, n_bootstrap=n_bootstrap,  n_jobs=n_jobs)
        if do_gsea:
            print('GSEA PD...')
            gsea_obj = ga.GSEA('PD', PD_Sm_optimal_map(), PATH_DIR, 'Sm')
            gsea_obj.run(n_jobs = 1, thresh_high_per = high_percentile,permute_factor=permute_factor, n_bootstrap=n_bootstrap, gene_set_list=gene_sets, gs_prefix=gs_prefix)
    except Exception as e:
        print(e)
        #error_log['Smijac'] = e

        
def thresh_loadings_and_gsea_sea(high_percentile = 95, perm_num=1000, permute_factor='label', error_log=None, n_bootstrap=100, n_jobs=1, gene_sets = GENE_SET_LIST, do_thresh=True, do_gsea=True, gs_prefix=None):
    celltypes = ['sncg', 'sst', 'pvalb']
    try:
        if do_thresh:
            print('Thresholding Sea.')
            pls_obj = pls.ThresholdPredictiveComponents('AD', AD_Seattle_optimal_map(), PATH_DIR, 'Sea', high_percentile=high_percentile,permute_factor=permute_factor)
            pls_obj.run(perm_num=perm_num, n_bootstrap=n_bootstrap, celltypes = celltypes)
        if do_gsea:
            print('GSEA AD Sea...')
            gsea_obj = ga.GSEA('AD', AD_Seattle_optimal_map(), PATH_DIR, 'Sea')
            gsea_obj.run(n_jobs = 1, thresh_high_per = high_percentile,permute_factor=permute_factor, n_bootstrap=n_bootstrap, gene_set_list=gene_sets, gs_prefix=gs_prefix)
    except Exception as e:
        #error_log['Sea'] = e
        print(e)

def thresh_loadings_and_gsea_mart(high_percentile = 95, perm_num=1000, permute_factor='label', error_log=None, n_bootstrap=100, n_jobs=20, gene_sets = GENE_SET_LIST, do_thresh=True, do_gsea=True, gs_prefix=None):
    try:
        if do_thresh:
            print('Thresholding Mart...')
            pls_obj = pls.ThresholdPredictiveComponents('PD', PD_Mart_optimal_map(), PATH_DIR, 'Mart', high_percentile=high_percentile,permute_factor=permute_factor )
            pls_obj.run(perm_num=perm_num, n_bootstrap=n_bootstrap,  n_jobs=n_jobs)
        if do_gsea:
            print('GSEA Mart...')
            gsea_obj = ga.GSEA('PD', PD_Mart_optimal_map(), PATH_DIR, 'Mart')
            gsea_obj.run(n_jobs = 1, thresh_high_per = high_percentile,permute_factor=permute_factor, n_bootstrap=n_bootstrap, gene_set_list=gene_sets, gs_prefix=gs_prefix)
    except Exception as e:
        print(e)

def thresh_loadings_and_gsea(func, *args):
    error_logs = {}
    func(*args)
    #with open(PATH_DIR+f'error_logs_thresh_gsea_{func.__name__}}.csv', 'w') as csvfile:
     #   writer = csv.DictWriter(csvfile, fieldnames=error_log.keys())
      #  writer.writeheader()
       # writer.writerows([error_log])


def run_multi_thresh_and_gsea(threshs, num_perm=1000, permute_factor='label', n_bootstrap=500, n_jobs=1, gene_sets=GENE_SET_LIST, do_thresh=True, do_gsea=True, gs_prefix=None):
    func_list1 = [thresh_loadings_and_gsea_pd, thresh_loadings_and_gsea_ros]
    func_list2 = [thresh_loadings_and_gsea_sm, thresh_loadings_and_gsea_sea] #, thresh_loadings_and_gsea_lung]
    func_list3 = [thresh_loadings_and_gsea_mart]
    func_list = func_list1 + func_list2
    Parallel(n_jobs=5, verbose=10)(delayed(thresh_loadings_and_gsea)(func, *(thresh, num_perm, permute_factor, None, n_bootstrap, n_jobs, gene_sets, do_thresh, do_gsea, gs_prefix)) for thresh, func in product(threshs, func_list3))


def threshold_corr_heatmap(num_perm=1000, high_per = 97.5, do_save=True, corr_col = 'loading', corr_metric = 'kt', doOr=False, do_rk=False, do_ss=False, do_sm=False,
                           do_kk=False, do_lk=False, do_lr=False, do_kr=False, do_ls=False, do_lsm=False, do_rr=False, do_pp = False, do_sr =False, do_sk = False):

    path_dir = PATH_DIR
    if corr_metric=='kt':
        corr_metric=kendalltau
    elif corr_metric == 'pwt':
        corr_metric = pwt

    if do_rk:
        corr_util = co.PLSCorrelationUtils(corr_metric, 'PD', 'AD', 'Kam', 'Rosmap', PATH_DIR, doOR = doOr, corr_colname=corr_col)
        corr_util.threshold_corrs(high_per, do_save=do_save, num_perm=num_perm)
        gc.collect()

    if do_lk:
        corr_util = co.PLSCorrelationUtils(corr_metric, 'Lung', 'PD', 'Kaminski', 'Kam', PATH_DIR, doOR = doOr, corr_colname=corr_col)
        corr_util.threshold_corrs(high_per, do_save=do_save, num_perm=num_perm)
        gc.collect()

    if do_lr:
        corr_util = co.PLSCorrelationUtils(corr_metric, 'Lung', 'AD', 'Kaminski', 'Rosmap', PATH_DIR, doOR = doOr, corr_colname=corr_col)
        corr_util.threshold_corrs(high_per, do_save=do_save, num_perm=num_perm)
        gc.collect()

    if do_ls:
        corr_util = co.PLSCorrelationUtils(corr_metric, 'Lung', 'AD', 'Kaminski', 'Sea', PATH_DIR, doOR = doOr, corr_colname=corr_col)
        corr_util.threshold_corrs(high_per, do_save=do_save, num_perm=num_perm)
        gc.collect()

    if do_lsm:
        corr_util = co.PLSCorrelationUtils(corr_metric, 'Lung', 'PD', 'Kaminski', 'Sm', PATH_DIR, doOR = doOr, corr_colname=corr_col)
        corr_util.threshold_corrs(high_per, do_save=do_save, num_perm=num_perm)
        gc.collect()

    if do_ss:
        corr_util = co.PLSCorrelationUtils(corr_metric, 'PD', 'AD', 'Sm', 'Sea', PATH_DIR, doOR = doOr, corr_colname=corr_col)
        corr_util.threshold_corrs(high_per, do_save=do_save, num_perm=num_perm)
        gc.collect()

    if do_sm:
        corr_util = co.PLSCorrelationUtils(corr_metric, 'PD', 'AD', 'Mart', 'Sea', PATH_DIR, doOR = doOr, corr_colname=corr_col)
        corr_util.threshold_corrs(high_per, do_save=do_save, num_perm=num_perm)
        gc.collect()

    if do_kk:
        corr_util = co.PLSCorrelationUtils(corr_metric, 'Lewy', 'PD', 'Kam', 'Kam', PATH_DIR, doOR = doOr, corr_colname=corr_col)
        corr_util.threshold_corrs(high_per, do_save=do_save, num_perm=num_perm)
        gc.collect()

    if do_kr:
        corr_util = co.PLSCorrelationUtils(corr_metric, 'Lewy', 'AD', 'Kam', 'Rosmap', PATH_DIR, doOR = doOr, corr_colname=corr_col)
        corr_util.threshold_corrs(high_per, do_save=do_save, num_perm=num_perm)
        gc.collect()

    if do_rr:
        corr_util = co.PLSCorrelationUtils(corr_metric, 'AD', 'AD', 'Rosmap', 'Rosmap_2', PATH_DIR, doOR = doOr, corr_colname=corr_col)
        corr_util.threshold_corrs(high_per, do_save=do_save, num_perm=num_perm)
        gc.collect()

    if do_pp:
        corr_util = co.PLSCorrelationUtils(corr_metric, 'PD', 'PD', 'Kam', 'Kam_2', PATH_DIR, doOR = doOr, corr_colname=corr_col)
        corr_util.threshold_corrs(high_per, do_save=do_save, num_perm=num_perm)
        gc.collect()

    if do_sk:
        corr_util = co.PLSCorrelationUtils(corr_metric, 'PD', 'PD', 'Sm', 'Kam', PATH_DIR, doOR = doOr, corr_colname=corr_col)
        corr_util.threshold_corrs(high_per, do_save=do_save, num_perm=num_perm)
        gc.collect()

    if do_sr:
        corr_util = co.PLSCorrelationUtils(corr_metric, 'AD', 'AD', 'Sea', 'Rosmap', PATH_DIR, doOR = doOr, corr_colname=corr_col)
        corr_util.threshold_corrs(high_per, do_save=do_save, num_perm=num_perm)
        gc.collect()


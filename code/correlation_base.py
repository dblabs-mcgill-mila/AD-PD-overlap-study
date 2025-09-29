"""
Script to perform correlation analyses. Examples in run.py as run_corr_*(...)
"""


import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed

import scipy

from matplotlib.colors import TwoSlopeNorm
from end2endPLS import PermutationGeneList, GeneListFromExpressionMtx, ThresholdPredictiveComponents
from utils import MakeDirs
import zipfile
import pyarrow.parquet as pq
import gc
from gseaAnalysis import GSEA
import numpy as np
from heatmap_r import heatmap


plt.rcParams['svg.fonttype'] = 'none'


class CorrelationBase(object):
    """
    Base class for handling correlation analyses between two diseases.
    """
    def __init__(self, corr_metric, disease1, disease2, disease_name1, disease_name2, path_dir):
        """
        Initialize correlation analysis with metric, diseases, and paths.
        """
        self.corr_metric = corr_metric
        self.disease1 = disease1
        self.disease2 = disease2
        self.disease_name1 = disease_name1
        self.disease_name2 = disease_name2
        self.var1_name = self.disease1+self.disease_name1
        self.var2_name = self.disease2+self.disease_name2 
        self.path_dir = path_dir

    def get_info(self):
        print(f'corr_metric: {self.corr_metric.__name__}')
        print(f'disease1: {self.disease1}')
        print(f'disease2: {self.disease2}')
        print(f'disease_name1: {self.disease_name1}')
        print(f'disease_name2: {self.disease_name2}')

    def __repr__(self):
        return f'{self.corr_metric.__name__}_{self.var1_name}_{self.var2_name}'
    
    def __str__(self):
        return self.__repr__()

    @classmethod
    def read_zip_to_dict(cls, zip_filename):
        """
        Read parquet files from a zip archive into a dictionary of DataFrames.
        """
        dataframes_dict = {}
        with zipfile.ZipFile(zip_filename, 'r') as zip_file:
            for file_name in zip_file.namelist():
                if file_name.endswith('.parquet'):
                    parquet_file = zip_file.open(file_name)
                    table = pq.read_table(parquet_file)
                    df = table.to_pandas()
                    df_name = file_name.replace('.parquet', '')
                    dataframes_dict[df_name] = df.copy()
                    parquet_file.close()
        return dataframes_dict
    
    def get_merged_dfs(self, df_path_1, df_path_2):
        """
        Merge DataFrames from two zip archives into a combined DataFrame.
        """
        df_dict1 = self.__class__.read_zip_to_dict(df_path_1)
        df1_all = pd.DataFrame()
        for celltype, df in df_dict1.items():
            if ('_score' not in celltype) and ('_rocauc' not in celltype):
                df['celltype'] = celltype
                df1_all = pd.concat([df1_all, df])
        df1_all['cell_comp'] = df1_all['celltype'] + '_' + df1_all['component'].apply(str)
        df_dict2 = self.__class__.read_zip_to_dict(df_path_2)
        df2_all = pd.DataFrame()
        for celltype, df in df_dict2.items():
            if ('_score' not in celltype) and ('_rocauc' not in celltype):
                df['celltype'] = celltype
                df2_all = pd.concat([df2_all, df])
        df2_all['cell_comp'] = df2_all['celltype'] + '_' + df2_all['component'].apply(str)
        del(df_dict1)
        del(df_dict2)
        merged_all_bs = self.merge_datasets(df1_all, df2_all)
        return merged_all_bs


class PLSCorrelationBase(CorrelationBase):

    def __init__(self, corr_metric, disease1, disease2, disease_name1, disease_name2, path_dir, doOR = True, loading_thresh=0, is_split = False, corr_colname = 'loading'):
        '''
        Parameters:
        is_split : bool (True/False) determines whether the analysis is a splithalf analysis or not. If True, some column names are modified to include the iteration number information
        corr_colname : choose between 'loading' / 'abs_loading'. The abs_loading loadings aren't zeroed out.
        '''
        self.is_split = is_split
        self.doOR = doOR
        self.loading_thresh = loading_thresh
        self.corr_colname = corr_colname
        super().__init__(corr_metric, disease1, disease2, disease_name1, disease_name2, path_dir)

    def get_info(self):
        super().get_info()
        print(f'is_split: {self.is_split}')
        print(f'doOR: {self.doOR}')
        print(f'loading_thresh: {self.loading_thresh}')
        print(f'corr_colname: {self.corr_colname}')

    def merge_datasets(self, df1, df2):
        col_list = [self.corr_colname, 'component', 'gene', 'celltype', 'cell_comp']
        if 'true_loading' in df1.columns and self.corr_colname!='true_loading':
            col_list.append('true_loading')
        merged_df = df1[col_list]
        merged_df = merged_df.merge(df2[col_list], 
                                      on = 'gene', 
                                      how = 'inner', 
                                      suffixes = (f'_{self.var1_name}', f'_{self.var2_name}'))
        return merged_df
        
    def _correlations(self, pls_path_1, pls_path_2, corr_path, min_overlaps_genes=15):
        merged_all_bs = self.get_merged_dfs(pls_path_1, pls_path_2)
        merged_all_bs[f'cell_comp_{self.var1_name}_{self.var2_name}'] = merged_all_bs[f'cell_comp_{self.var1_name}'].astype(str)+'%' + merged_all_bs[f'cell_comp_{self.var2_name}'].astype(str)
        
        #Note: if doOR is None, do nothing.
        if self.doOR == True:
            merged_all_bs.drop(merged_all_bs[( abs(merged_all_bs[f'{self.corr_colname}_{self.var1_name}']) <= self.loading_thresh) & (abs(merged_all_bs[f'{self.corr_colname}_{self.var2_name}'])<= self.loading_thresh)].index, inplace = True)
        elif self.doOR == False:
            merged_all_bs.drop(merged_all_bs[( abs(merged_all_bs[f'{self.corr_colname}_{self.var1_name}']) <= self.loading_thresh) | (abs(merged_all_bs[f'{self.corr_colname}_{self.var2_name}'])<= self.loading_thresh)].index, inplace = True)
        gc.collect()
        def _f_corr(_ser, metric, colname):
            if (colname == 'loading') | (colname == 'true_loading'):
                # We always correlate with the true loadings (plsr model assigned loadings)
                colname = 'true_loading' if f'true_loading_{self.var1_name}' in _ser.columns else colname
                return ( metric(_ser[f'{colname}_{self.var1_name}'], _ser[f'{colname}_{self.var2_name}']), _ser.shape[0])
            elif colname == 'abs_loading':
                return ( metric(_ser[f'{colname}_{self.var1_name}'].values * np.sign(_ser[f'true_loading_{self.var1_name}'].values),\
                                _ser[f'{colname}_{self.var2_name}'].values * np.sign(_ser[f'true_loading_{self.var2_name}'].values)),
                         _ser.shape[0] )
            else:
                raise AssertionError('Using some weird colname. Not present in the pls results')
            
        raw_corr_df = pd.DataFrame()
        raw_corr_df = merged_all_bs.groupby(f'cell_comp_{self.var1_name}_{self.var2_name}').apply(_f_corr, self.corr_metric, self.corr_colname)
        del(merged_all_bs)
        raw_corr_df = raw_corr_df.to_frame()
        raw_corr_df.reset_index(inplace=True)
        raw_corr_df[f'cell_comp_{self.var1_name}'] = raw_corr_df[f'cell_comp_{self.var1_name}_{self.var2_name}'].apply(lambda x: x.rsplit('%',1)[0])
        raw_corr_df[f'cell_comp_{self.var2_name}'] = raw_corr_df[f'cell_comp_{self.var1_name}_{self.var2_name}'].apply(lambda x: x.rsplit('%',1)[1])
        raw_corr_df[f'{self.corr_metric.__name__}_corr'] = raw_corr_df[0].apply(lambda x: x[0][0])
        raw_corr_df['p_value'] = raw_corr_df[0].apply(lambda x: x[0][1])
        raw_corr_df['gene_overlap'] = raw_corr_df[0].apply(lambda x: x[1])
        raw_corr_df.drop(columns = [f'cell_comp_{self.var1_name}_{self.var2_name}', 0], inplace = True)
        
        MakeDirs.make_dirs(corr_path, include_end=False)
        raw_corr_df.to_csv(corr_path)
        del(raw_corr_df)
        gc.collect()
        


class PLSCorrelationTrue(PLSCorrelationBase):
    """
    Extension of CorrelationBase for PLS loadings-based correlations.
    """

    @classmethod
    def correlation_path(cls, path_dir, disease1, disease2, disease_name1, disease_name2, corr_metric, seed, doOR, corr_colname):
        '''TODO: scary - hardcoding loading here. Change it to match the default from CorrelationBase init'''
        if corr_colname != 'loading':
            return path_dir+f'{disease1}_{disease2}/{disease_name1}_{disease_name2}/doOR_{doOR}/correlation_{disease1}_{disease2}_{corr_metric.__name__}_{seed}seed_{doOR}doOR_{corr_colname}.csv'
        return path_dir+f'{disease1}_{disease2}/{disease_name1}_{disease_name2}/doOR_{doOR}/correlation_{disease1}_{disease2}_{corr_metric.__name__}_{seed}seed_{doOR}doOR.csv'

    def do_correlations(self, 
                        n_bootstrap=500, 
                        zero_threshold=5, 
                        seed=42, 
                        thresh_high_per = None,
                        thresh_low_per = None,
                        permute_factor='label',
                        use_permute_res = True,
                        seed2 = None):
        seed2 = seed2 if seed2 is not None else seed
        if use_permute_res:
            pls_GE_path1 = ThresholdPredictiveComponents.pls_bs_thresh_path(self.path_dir, 
                                                                            self.disease1, 
                                                                            self.disease_name1, 
                                                                            n_bootstrap, 
                                                                            zero_threshold, 
                                                                            seed,
                                                                            thresh_high_per,
                                                                            thresh_low_per,
                                                                            permute_factor)+'.zip'
            pls_GE_path2 = ThresholdPredictiveComponents.pls_bs_thresh_path(self.path_dir, 
                                                                        self.disease2, 
                                                                        self.disease_name2, 
                                                                        n_bootstrap, 
                                                                        zero_threshold, 
                                                                        seed2,
                                                                        thresh_high_per,
                                                                        thresh_low_per,
                                                                        permute_factor)+'.zip'
        else:
            pls_GE_path1 = GeneListFromExpressionMtx.pls_bs_GE_path(self.path_dir, 
                                                                    self.disease1, 
                                                                    self.disease_name1, 
                                                                    n_bootstrap, 
                                                                    zero_threshold, 
                                                                    seed) +'.zip'
            pls_GE_path2 = GeneListFromExpressionMtx.pls_bs_GE_path(self.path_dir, 
                                                                    self.disease2, 
                                                                    self.disease_name2, 
                                                                    n_bootstrap, 
                                                                    zero_threshold, 
                                                                    seed2) +'.zip'
        corr_path = PLSCorrelationTrue.correlation_path(self.path_dir, 
                                                        self.disease1, 
                                                        self.disease2, 
                                                        self.disease_name1, 
                                                        self.disease_name2, 
                                                        self.corr_metric,
                                                        seed,
                                                        self.doOR,
                                                        self.corr_colname,
                                                        )
        self._correlations(pls_GE_path1, pls_GE_path2, corr_path)
        
    def run(self, n_bootstrap=500, zero_threshold=5, seed=42, thresh_high_per = None, thresh_low_per = None, permute_factor='label', use_permute_res = True, seed2 = None):
        self.do_correlations(n_bootstrap=n_bootstrap, zero_threshold=zero_threshold, seed=seed, thresh_high_per=thresh_high_per, thresh_low_per = thresh_low_per, permute_factor=permute_factor, use_permute_res=use_permute_res, seed2 = seed2, )
    

class PLSCorrelationPermute(PLSCorrelationBase):
        """
        Extension of PLSCorrelationBase to run correlation on permutation based results- permutated correlations.
        """

    @classmethod
    def correlation_path(cls, path_dir, disease1, disease2, disease_name1, disease_name2, perm_idx, seed, corr_metric, doOR, corr_colname):
        path_prefix = path_dir+f'{disease1}_{disease2}/{disease_name1}_{disease_name2}/doOR_{doOR}/Permute/correlation_{disease1}_{disease2}_{corr_metric.__name__}_perms_{seed}seed_{doOR}doOR'
        if corr_colname != 'loading':
            return path_prefix+f'_{corr_colname}_{str(perm_idx)}.csv'
        return path_prefix+'_'+str(perm_idx)+'.csv'

    def do_correlations(self, perm_idx, seed=42, permute_factor='label', seed2 = None):
        seed2 = seed2 if seed2 is not None else seed
        print(f'iteration #:{perm_idx:.0f}')
        pls_perm_GE_path1 = PermutationGeneList.pls_perm_GE_path(self.path_dir,
                                                           self.disease1, 
                                                           self.disease_name1, 
                                                           seed,
                                                           permute_factor)+'_'+str(perm_idx)+'.zip'
        pls_perm_GE_path2 = PermutationGeneList.pls_perm_GE_path(self.path_dir,
                                                           self.disease2, 
                                                           self.disease_name2, 
                                                           seed2,
                                                           permute_factor)+'_'+str(perm_idx)+'.zip'
        corr_path = PLSCorrelationPermute.correlation_path(self.path_dir, 
                                                    self.disease1, 
                                                    self.disease2, 
                                                    self.disease_name1, 
                                                    self.disease_name2, 
                                                    perm_idx,
                                                    seed,
                                                    self.corr_metric,
                                                    self.doOR,
                                                    self.corr_colname)
        self._correlations(pls_perm_GE_path1, pls_perm_GE_path2, corr_path)

    def run(self, n_jobs = 10, num_perm =100, start_perm_idx=0, seed=42, permute_factor = 'label', seed2 = None):
        Parallel(n_jobs=n_jobs, verbose = 10)(delayed(self.do_correlations)(perm_idx, seed=seed, permute_factor=permute_factor, seed2=seed2) for perm_idx in range(start_perm_idx, num_perm))


class PLSCorrelation(PLSCorrelationBase):

    def __init__(self, 
                 corr_metric, 
                 disease1, 
                 disease2, 
                 disease_name1, 
                 disease_name2, 
                 path_dir,
                 doOR = True, 
                 loading_thresh=0,
                 is_split=False,
                 corr_colname = 'true_loading',
                 ):
        super().__init__(corr_metric, disease1, disease2, disease_name1, disease_name2, path_dir, is_split=is_split, doOR = doOR, loading_thresh=loading_thresh, corr_colname=corr_colname)
        self.perm_corr = PLSCorrelationPermute(corr_metric, disease1, disease2, disease_name1, disease_name2, path_dir, doOR = doOR, loading_thresh=loading_thresh, corr_colname='loading')
        self.true_corr = PLSCorrelationTrue(corr_metric, disease1, disease2, disease_name1, disease_name2, path_dir, doOR = doOR, loading_thresh=loading_thresh, corr_colname=corr_colname)

    def run(self, 
            num_perm, 
            start_perm_idx=0, 
            n_bootstrap=500, 
            zero_threshold=5, 
            seed=42, n_jobs=10, 
            permute_factor='label',
            thresh_high_per = None,
            thresh_low_per = None,
            use_permute_res = True,
            seed2 = None):
        #running true correlations
        print(f'running pls true correlation...')
        self.true_corr.run(n_bootstrap=n_bootstrap, 
                           zero_threshold=zero_threshold, 
                           seed=seed,
                           thresh_high_per=thresh_high_per,
                           thresh_low_per=thresh_low_per,
                           permute_factor=permute_factor,
                           use_permute_res=use_permute_res,
                           seed2 = seed2,
                           )
        #running permutation correlations
        if num_perm:
            #num_perm is None for inter split correlations. Just work with true correlations in case of split half analysis
            print('running pls perm correlations')
            self.perm_corr.run(num_perm = num_perm, start_perm_idx = start_perm_idx, seed=42, n_jobs=n_jobs, permute_factor=permute_factor, seed2 = seed2)


class CorrelationUtilsBase(CorrelationBase):
    '''utils for thresholding and plotting pls correlation heatmaps'''

    def cellcomp_name1(self):
        return f'cell_comp_{self.var1_name}'

    def cellcomp_name2(self):
        return f'cell_comp_{self.var2_name}'
 
    def cellcomp_comb(self):
        return f'cell_comp_{self.var1_name}_{self.var2_name}'
    
    def _heatmap_mtx_reshape(self,
                        corr_matrix, 
                        dropna = True, 
                        fill_na=None, 
                        rem_celltypes_1 = None, 
                        rem_celltypes_2 = None):

        celltype_map = {'ast': 'Ast', 'ex':'Ex', 'in': 'In', 'mic':'Mic', 'macro':'Macro', 'oligo':'Oli', 'opc':'Opc', \
            'endo':'End', 'per':'Per', 'sox6':'Sox6', 'calb1':'Calb1', 'epen':'Epen', 'l4_it':'L4_it', 'l5_it':'L5_it', 'vip':'Vip'}
        def label_map(ct):
            for k,v in celltype_map.items():
                if k in ct.lower():
                    return v
            return ct

        _corr_df = corr_matrix.astype({f'{self.corr_metric.__name__}_corr':'float64'})
        if dropna:
            _corr_df.dropna(inplace=True)
        elif fill_na:
            _corr_df.fillna(fill_na)
        if rem_celltypes_1:
            _corr_df.drop(index = _corr_df[_corr_df[self.cellcomp_name1()].str.contains('|'.join(rem_celltypes_1))].index, inplace=True)
        if rem_celltypes_2:
            _corr_df.drop(index = _corr_df[_corr_df[self.cellcomp_name2()].str.contains('|'.join(rem_celltypes_2))].index, inplace=True)
        
        _corr_df[self.cellcomp_name1()] = _corr_df[self.cellcomp_name1()].apply(lambda x: label_map(x.rsplit('_',1)[0])+'_'+str(int(x.rsplit('_',1)[1])+1))
        _corr_df[self.cellcomp_name2()] = _corr_df[self.cellcomp_name2()].apply(lambda x: label_map(x.rsplit('_',1)[0])+'_'+str(int(x.rsplit('_',1)[1])+1))
        
        return _corr_df


    def _heatmap( self,
                 mtx, 
                 norm = None, 
                 do_transpose=False, 
                 annot = False, 
                 annot_thresh=0.3, 
                 fontsize=8, 
                 size_labels = [5,30,200,10000],
                 size_scale = 7000,
                 figsize = None,
                 palette = None,
                 facecolor = None,
                 ):

        norm = norm if norm else (0,-1, 1)
        xlabel = mtx[self.cellcomp_name1()]
        ylabel = mtx[self.cellcomp_name2()]

        transposed = 0
        if xlabel.unique().shape[0] < ylabel.unique().shape[0]: 
            xlabel = mtx[self.cellcomp_name2()]
            ylabel = mtx[self.cellcomp_name1()]
            transposed = ~transposed
        if do_transpose:
            temp = xlabel
            xlabel = ylabel
            ylabel = temp
            del(temp)
            transposed = ~transposed

        figsize = figsize if figsize else (xlabel.shape[0], ylabel.shape[0]) 
        f = plt.figure(figsize=figsize)
        size_labels = size_labels
        palette = palette if palette else sns.diverging_palette(140, 350, s=80, n=128, sep=10)
        ax, ax1, ax2, ss = heatmap(
            x=xlabel, # Column to use as horizontal dimension 
            y=ylabel, # Column to use as vertical dimension
            size_scale=size_scale, # Change this to see how it affects the plot
            size=np.log(mtx['gene_overlap']),
            size_range = (np.log(1), np.log(10000)),
            color=mtx[f'{self.corr_metric.__name__}_corr'].fillna(0),
            color_range=norm,
            palette=palette,
            fontsize = fontsize,
            x_tick_rotation = 90,
            num_size_label=6,
            m_color='g',
            x_order = xlabel.sort_values().unique(),
            y_order = ylabel.sort_values().unique()[::-1],
            size_labels = np.log(size_labels).tolist(),
            facecolor = facecolor,
        )

        if ~transposed:
            ax.set_xlabel(f'{self.var1_name}', fontsize=fontsize)
            ax.set_ylabel(f'{self.var2_name}', fontsize=fontsize)
        else:
            ax.set_ylabel(f'{self.var1_name}', fontsize=fontsize)
            ax.set_xlabel(f'{self.var2_name}', fontsize=fontsize)

        ax.grid(linewidth=1.5, which='minor')
        ax1.set_ylabel(f'{self.corr_metric.__name__} corr', fontsize = fontsize, rotation=-90)
        ax1.yaxis.set_label_position("right")
        _ = ax2.set_yticklabels(size_labels, rotation=0, fontsize = fontsize)
        _ = ax2.set_xlabel('# genes', fontsize=20)
        ax2.patch.set_alpha(0)
        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()): label.set_fontsize(fontsize)

        return f, ax


    def _plot_mtx(self, mtx, norm = None, do_transpose=False, annot = False, annot_thresh=0.3, fontsize=8, size_labels = [5,30,200,10000], size_scale = 7000, figsize = None, palette = None, facecolor = None):
        
        fig, ax = self._heatmap(mtx, 
                                norm = norm, 
                                do_transpose=do_transpose, 
                                annot = annot, 
                                annot_thresh = annot_thresh, 
                                fontsize = fontsize, 
                                size_labels = size_labels,
                                size_scale = size_scale,
                                figsize = figsize,
                                palette = palette,
                                facecolor = facecolor, )
        if annot:
            for i in range(mtx.shape[0]):
                for j in range(mtx.shape[1]):
                    if abs(mtx.iloc[i, j]) >= min(annot_thresh, mtx.abs().max().max()):
                        ax.text(j + 0.5, i + 0.5, f'{mtx.iloc[i, j]:.1f}', 
                                horizontalalignment='center', 
                                verticalalignment='center', 
                                color='white' if mtx.iloc[i, j] > 70 else 'black')
        return fig, ax
    
    def plot_corrs(self, 
                   corr_matrix=None, 
                   norm=None, 
                   dropna = True, 
                   fill_na=None, 
                   rem_celltypes_1 = None, 
                   rem_celltypes_2 = None,
                   seed=42,
                   num_perm=1000,
                   high_percentile=95,
                   low_percentile=None,
                   do_transpose=False,
                   annot=False,
                   annot_thresh=0.3,
                   fontsize=14, 
                   size_labels = [5,30,200,10000],
                   size_scale = 7000,
                   figsize = None,
                   palette = None,
                   facecolor = None,
                ):
        '''Plot function for correlation heatmap (example Figure. 1a)'''
        if corr_matrix is None:
            corr_path_pre = self.true_corr.correlation_path(self.path_dir, 
                                                            self.disease1, 
                                                            self.disease2, 
                                                            self.disease_name1, 
                                                            self.disease_name2, 
                                                            self.corr_metric,
                                                            seed,
                                                            self.doOR,
                                                            self.corr_colname)
            corr_path = PLSCorrelationUtils.threshold_path(corr_path_pre, num_perm, high_percentile, low_percentile)
            corr_matrix = pd.read_csv(corr_path)
        mtx = self._heatmap_mtx_reshape(corr_matrix, 
                                       dropna = dropna, 
                                       fill_na = fill_na,
                                       rem_celltypes_1 = rem_celltypes_1,
                                       rem_celltypes_2 = rem_celltypes_2 )
        fig, ax = self._plot_mtx(mtx, norm=norm, do_transpose=do_transpose, annot=annot, annot_thresh=annot_thresh, fontsize=fontsize, size_labels=size_labels, size_scale = size_scale, figsize = figsize, palette = palette, facecolor = facecolor)
        ax.set_title(f'pls predictive loadings correlation {self.corr_metric.__name__} heatmap, disease: {self.var1_name} and {self.var2_name}, doOR: {self.doOR}, thresh: {high_percentile}/{low_percentile if low_percentile else 100-high_percentile}', fontsize=fontsize)
        return fig

    def colorbar_norm(self, vmax = 1, vmin = -1, vcenter = 0):
        return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    

class PLSCorrelationUtils(PLSCorrelation, CorrelationUtilsBase):
    '''
    Utils for correlation modules
    '''

    def cellcomp_name1(self):
        if self.is_split:
            return 'cell_comp_'+'_'.join([self.var1_name.rsplit('_',2)[0], self.var1_name.rsplit('_',1)[-1]])
        else:
            return f'cell_comp_{self.var1_name}'

    def cellcomp_name2(self):
        if self.is_split:
            return 'cell_comp_'+'_'.join([self.var2_name.rsplit('_',2)[0], self.var2_name.rsplit('_',1)[-1]])
        else:
            return f'cell_comp_{self.var2_name}'
 
    def _get_true_corrs(self, seed):
        corr_path = self.true_corr.correlation_path(self.path_dir, 
                                                    self.disease1, 
                                                    self.disease2, 
                                                    self.disease_name1, 
                                                    self.disease_name2, 
                                                    self.corr_metric,
                                                    seed,
                                                    self.doOR,
                                                    self.corr_colname)
        return pd.read_csv(corr_path)

    def rename_cols(self):
        return {f'cell_comp_{self.var1_name}': self.cellcomp_name1(), 
                f'cell_comp_{self.var2_name}': self.cellcomp_name2()}
        
    def get_true_corrs(self, seed):
        true_corrs = self._get_true_corrs(seed)
        true_corrs.rename(columns = self.rename_cols(), inplace=True)
        true_corrs[self.cellcomp_comb()] = true_corrs[self.cellcomp_name1()].astype(str)\
                                                                        +'_' + true_corrs[self.cellcomp_name2()].astype(str)
        if 'Unnamed: 0' in true_corrs.columns:
            true_corrs.drop(columns = 'Unnamed: 0', inplace=True)
        return true_corrs
    
    def _get_permuted_corrs(self, num_perm, seed):
        corr_path = lambda perm_idx, seed: self.perm_corr.correlation_path(self.path_dir, 
                                                                        self.disease1, 
                                                                        self.disease2, 
                                                                        self.disease_name1, 
                                                                        self.disease_name2, 
                                                                        perm_idx,
                                                                        seed,
                                                                        self.corr_metric,
                                                                        self.doOR,
                                                                        self.perm_corr.corr_colname)
        raw_corr_dfs = []
        for perm_idx in range(num_perm):
            try:
                raw_corr_dfs.append(pd.read_csv(corr_path(perm_idx, seed)))
            except:
                print(f'skipping perm idx {perm_idx}')
        # [pd.read_csv(corr_path(perm_idx, seed)) for perm_idx in range(num_perm)]
        all_corr_dfs = pd.DataFrame()
        for _df in raw_corr_dfs:
            all_corr_dfs = pd.concat([_df, all_corr_dfs])
        if 'Unnamed: 0' in all_corr_dfs.columns:
            all_corr_dfs.drop(columns = 'Unnamed: 0', inplace=True)
        all_corr_dfs.reset_index(inplace=True, drop = True)
        return all_corr_dfs
    
    def get_permuted_corrs(self, num_perm, seed):
        all_corr_dfs = self._get_permuted_corrs(num_perm, seed)
        all_corr_dfs.rename(columns = self.rename_cols(), inplace=True)
        all_corr_dfs[self.cellcomp_comb()] = all_corr_dfs[self.cellcomp_name1()].astype(str) \
                                                                        + '_' + all_corr_dfs[self.cellcomp_name2()].astype(str)
        return all_corr_dfs
    
    @classmethod
    def threshold_path(cls, corr_path_pre, num_perm, high_per , low_per):
        low_per = low_per if low_per else 100-high_per
        return corr_path_pre + f'_{num_perm}thresholded_{high_per}highper_{low_per}lowper.csv'

    def threshold_corrs(self, high_percentile, true_corrs=None, all_corr_dfs=None, low_percentile = None, num_perm=100, seed=42, do_save = False):
        print(f'doing correlations {self.disease_name1}, {self.disease_name2}...')
        if not true_corrs:
            true_corrs = self.get_true_corrs(seed)
        print('finished true correlations')
        if not all_corr_dfs:
            all_corr_dfs = self.get_permuted_corrs(num_perm=num_perm, seed=seed)
        print('retrieved all permutations')

        low_percentile = low_percentile if low_percentile else 100-high_percentile
        for cell_comp in all_corr_dfs[self.cellcomp_comb()].unique():
            permute_perc_high = scipy.stats.scoreatpercentile(all_corr_dfs[all_corr_dfs[self.cellcomp_comb()]==cell_comp][f'{self.corr_metric.__name__}_corr'].values, 
                                                              per = high_percentile, axis =0)
            permute_perc_low = scipy.stats.scoreatpercentile(all_corr_dfs[all_corr_dfs[self.cellcomp_comb()]==cell_comp][f'{self.corr_metric.__name__}_corr'].values, 
                                                             per = low_percentile, axis =0)
            # zero loadings that are <higher_percentile or >lower_percentile
            true_val = true_corrs[true_corrs[self.cellcomp_comb()]==cell_comp][f'{self.corr_metric.__name__}_corr'].values
            # if true_val<max(permute_perc_high, permute_perc_low) and true_val>min(permute_perc_high, permute_perc_low):
            #     true_corrs.loc[true_corrs[self.cellcomp_comb()]==cell_comp, f'{self.corr_metric.__name__}_corr']=0
            if true_val.size > 0:
                if np.all((true_val < max(permute_perc_high, permute_perc_low)) & 
                        (true_val > min(permute_perc_high, permute_perc_low))):
                    true_corrs.loc[true_corrs[self.cellcomp_comb()] == cell_comp, f'{self.corr_metric.__name__}_corr'] = 0
        true_corrs.drop(columns=[self.cellcomp_comb()], inplace=True)
        print('finished computation. Saving...')
        if do_save:
            corr_path_pre = self.true_corr.correlation_path(self.path_dir, 
                                                    self.disease1, 
                                                    self.disease2, 
                                                    self.disease_name1, 
                                                    self.disease_name2, 
                                                    self.corr_metric,
                                                    seed,
                                                    self.doOR,
                                                    self.corr_colname)
            true_corrs.to_csv(PLSCorrelationUtils.threshold_path(corr_path_pre, num_perm, high_percentile, low_percentile))
        else:
            return true_corrs
    

class PLSIntraSplitCorrelationUtils(PLSCorrelation, CorrelationUtilsBase):
    '''Class to calculate correlations between two runs for the same dataset'''

    def __init__(self, corr_metric, disease, disease_name, path_dir, num_split = 10, doOR = True, loading_thresh=0):
        '''
        The same disease (and dataset) is analyzed for intra-correlation. Split by non-overlapping donors.
        '''
        self.doOR = doOR
        self.loading_thresh = loading_thresh
        self.corr_utils = [PLSCorrelationUtils(corr_metric, 
                                                f'{disease}_split', 
                                                f'{disease}_split', 
                                                f'{disease_name}_{idx}_1', 
                                                f'{disease_name}_{idx}_2', 
                                                path_dir, 
                                                doOR = doOR, 
                                                loading_thresh=loading_thresh, 
                                                is_split = True) for idx in range(num_split)]
        disease1 = f'{disease}_split'
        disease_name1 = f'{disease_name}_1'
        disease2 = f'{disease}_split'
        disease_name2 = f'{disease_name}_2'
        super().__init__(corr_metric, disease1, disease2, disease_name1, disease_name2, path_dir, doOR=doOR, loading_thresh=loading_thresh, is_split=False)
        
    def threshold_corrs(self, high_percentile, low_percentile = None, num_perm=100, seed=42):
        thresh_mtxs = [corr_util.threshold_corrs(high_percentile, 
                                                low_percentile = low_percentile, 
                                                num_perm=num_perm, 
                                                seed=seed)
                      for corr_util in self.corr_utils]
        # return thresh_mtxs
        thresh_mtx = pd.concat(thresh_mtxs)
        return thresh_mtx.groupby([self.cellcomp_name1(), self.cellcomp_name2()]).mean().reset_index()
    

class PLSInterSplitCorrelationUtils(CorrelationUtilsBase):
    '''
    Perform correlation on two halves of disease1-disease2 dataset pair. 
    Analyze correlation strength robustness using 50% subsampled datasets.
    '''
    def __init__(self, corr_metric, disease1, disease2, disease_name1, disease_name2, path_dir, num_split = 10, doOR = True, loading_thresh=0):
        self.doOR = doOR
        self.loading_thresh = loading_thresh
        self.corr_metric = corr_metric
        self.corr_utils1 = [PLSCorrelationUtils(corr_metric, 
                                                f'{disease1}_split', 
                                                f'{disease2}_split', 
                                                f'{disease_name1}_{idx}_1', 
                                                f'{disease_name2}_{idx}_1', 
                                                path_dir, 
                                                doOR = doOR, 
                                                loading_thresh=loading_thresh, 
                                                is_split = True) for idx in range(num_split)]
        self.corr_utils2 = [PLSCorrelationUtils(corr_metric, 
                                                f'{disease1}_split', 
                                                f'{disease2}_split', 
                                                f'{disease_name1}_{idx}_2', 
                                                f'{disease_name2}_{idx}_2', 
                                                path_dir, 
                                                doOR = doOR, 
                                                loading_thresh=loading_thresh, 
                                                is_split = True) for idx in range(num_split)]
        self.corr_obj1 = PLSCorrelation(corr_metric, f'{disease1}_split', f'{disease2}_split', f'{disease_name1}_1', f'{disease_name2}_1', path_dir, doOR=doOR, loading_thresh=loading_thresh, is_split=False)
        self.corr_obj2 = PLSCorrelation(corr_metric, f'{disease1}_split', f'{disease2}_split', f'{disease_name1}_2', f'{disease_name2}_2', path_dir, doOR=doOR, loading_thresh=loading_thresh, is_split=False)
         
    def threshold_corrs(self, high_percentile, low_percentile = None, num_perm=100, seed=42):
        print('No thresholding performed. Keeping true correlations')
        return None
    
    def run(self, n_bootstrap=500, zero_threshold=5, seed=42, start_idx=0):
        print('Correlating first splits')
        [corr_util.run(num_perm=None, n_bootstrap=n_bootstrap, zero_threshold=zero_threshold, seed=seed, use_permute_res=False) \
         for corr_util in self.corr_utils1[start_idx:]]
        print('Correlating second splits')
        [corr_util.run(num_perm=None, n_bootstrap=n_bootstrap, zero_threshold=zero_threshold, seed=seed, use_permute_res=False) \
         for corr_util in self.corr_utils2[start_idx:]]
        
    def do_splithalves_correlation(self, seed=42, corr_metric = None):
        corr_metric = corr_metric if corr_metric else self.corr_metric
        corr_diff_df = pd.DataFrame()
        for idx, (cu1, cu2) in enumerate(zip(self.corr_utils1, self.corr_utils2)):
            tc1 = cu1.get_true_corrs(seed).fillna(0)
            tc1.rename(columns = {cu1.cellcomp_comb():'cell_comp'}, inplace=True)
            tc2 = cu2.get_true_corrs(seed).fillna(0)
            tc2.rename(columns = {cu2.cellcomp_comb():'cell_comp'}, inplace=True)
            _df = tc1.merge(tc2, how='inner',
                            on = 'cell_comp',
                            suffixes=('_1', '_2'))
            _df['diff'] = _df[f'{cu1.corr_metric.__name__}_corr_1'].values - _df[f'{cu2.corr_metric.__name__}_corr_2'].values
            corr_diff_df = _df[['diff', 'cell_comp']] if corr_diff_df.empty else corr_diff_df.merge(_df[['diff', 'cell_comp']], on = 'cell_comp', suffixes=['', f'_{idx}'])
        return corr_diff_df


class GSEACorrelation(CorrelationBase):
    '''
    DEPRECATED
    Correlation between GSEA terms. Correlation done on basis of presence or absence of term.
    '''
    def merge_datasets(self, df1, df2):
        components = df1[['Term', 'NES', 'celltype', 'component', 'gene_set_source']]
        components = components.merge(df2[['Term', 'NES', 'celltype', 'component', 'gene_set_source']], 
                                      on = ['Term', 'gene_set_source'], 
                                      how = 'inner', 
                                      suffixes = (f'_{self.var1_name}', f'_{self.var2_name}'))
        return components
        
    def _correlations(self, gsea_path_1, gsea_path_2, corr_path, gene_sets = None):
        try:
        # print('reading 1')
            merged_all_bs = self.get_merged_dfs(gsea_path_1, gsea_path_2)
            if gene_sets:
                merged_all_bs = merged_all_bs[merged_all_bs.gene_set_source.isin(gene_sets)]
            merged_all_bs[self.cellcomp_comb()] = merged_all_bs[f'celltype_{self.var1_name}'].astype(str)+'_'+merged_all_bs[f'component_{self.var1_name}'].astype(int).astype(str)+\
                            '%' + merged_all_bs[f'celltype_{self.var2_name}'].astype(str)+'_'+merged_all_bs[f'component_{self.var1_name}'].astype(int).astype(str)
            gc.collect()
            # print('correlating')
            def _f(_ser, metric):
                return metric(_ser[f'NES_{self.var1_name}'], _ser[f'NES_{self.var2_name}'])
            raw_corr_df = pd.DataFrame()
            raw_corr_df = merged_all_bs.groupby(self.cellcomp_comb()).apply(_f, self.corr_metric)
            del(merged_all_bs)
            raw_corr_df = raw_corr_df.to_frame()
            raw_corr_df.reset_index(inplace=True)
            raw_corr_df[f'cell_comp_{self.var1_name}'] = raw_corr_df[self.cellcomp_comb()].apply(lambda x: x.rsplit('%',1)[0])
            raw_corr_df[f'cell_comp_{self.var2_name}'] = raw_corr_df[self.cellcomp_comb()].apply(lambda x: x.rsplit('%',1)[1])
            raw_corr_df[f'{self.corr_metric.__name__}_corr'] = raw_corr_df[0].apply(lambda x: x[0])
            raw_corr_df['p_value'] = raw_corr_df[0].apply(lambda x: x[1])
            raw_corr_df.drop(columns = [self.cellcomp_comb(), 0], inplace = True)
            MakeDirs.make_dirs(corr_path, include_end=False)
            raw_corr_df.to_csv(corr_path)
            del(raw_corr_df)
            print(f'finished correlation...{corr_path}')
            gc.collect()
        except:
            print(f'skipping {gsea_path_1} {gsea_path_2}')

    @classmethod
    def correlation_path(cls, path_dir, disease1, disease2, disease_name1, disease_name2, corr_metric, seed):
        return path_dir+f'{disease1}_{disease2}/{disease_name1}_{disease_name2}/gsea_correlation_{disease1}_{disease2}_{corr_metric.__name__}_{seed}seed'+'.csv'

    def do_correlations(self, n_bootstrap=500, zero_threshold=5, seed=42):
        gsea_path1 = GSEA.gsea_path(self.path_dir, 
                                    self.disease1, 
                                    self.disease_name1, 
                                    zero_threshold, 
                                    n_bootstrap, 
                                    seed)
        gsea_path2 = GSEA.gsea_path(self.path_dir, 
                                    self.disease2, 
                                    self.disease_name2, 
                                    zero_threshold, 
                                    n_bootstrap, 
                                    seed)
        corr_path = GSEACorrelation.correlation_path(self.path_dir, 
                                                    self.disease1, 
                                                    self.disease2, 
                                                    self.disease_name1, 
                                                    self.disease_name2, 
                                                    self.corr_metric,
                                                    seed,
                                                    )
        self._correlations(gsea_path1, gsea_path2, corr_path)
        
    def run(self, n_bootstrap=500, zero_threshold=5, seed=42):
        self.do_correlations(n_bootstrap=n_bootstrap, zero_threshold=zero_threshold, seed=seed)
    

class GSEACorrelationUtils(CorrelationUtilsBase):

    def get_correlations(self, seed):
        corr_path = GSEACorrelation.correlation_path(self.path_dir, 
                                                    self.disease1, 
                                                    self.disease2, 
                                                    self.disease_name1, 
                                                    self.disease_name2, 
                                                    self.corr_metric,
                                                    seed,
                                                    )
        return pd.read_csv(corr_path)

    def plot_corrs(self, 
                   corr_matrix=None, 
                   norm=None, 
                   dropna = True, 
                   fill_na=None, 
                   rem_celltypes_1 = None, 
                   rem_celltypes_2 = None,):
        mtx = self._heatmap_mtx_reshape(corr_matrix, 
                                       dropna = dropna, 
                                       fill_na = fill_na,
                                       rem_celltypes_1 = rem_celltypes_1,
                                       rem_celltypes_2 = rem_celltypes_2 )
        fig, ax = self._plot_mtx(mtx, norm=norm)
        ax.set_title(f'GSEA NES correlation {self.corr_metric.__name__} heatmap, disease: {self.var1_name} and {self.var2_name}')
        return fig


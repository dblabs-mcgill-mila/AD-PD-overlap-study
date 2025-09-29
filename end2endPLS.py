import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from numpy.random import default_rng

from scipy.stats import pearsonr, kendalltau, scoreatpercentile
from scipy import sparse
from sklearn.utils import shuffle
from itertools import chain
import gc
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import zipfile
import logging as log
import pyarrow as pa
import pyarrow.parquet as pq
import warnings
from utils import MakeDirs
from scipy.linalg import pinv as pinv2
import anndata as ad
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from sklearn.utils import resample
import sys


class BasePLS(object):

    def __init__(self, disease, celltype_component_optimal, path_dir, disease_name):
        self.disease = disease
        self.optimalMapping = celltype_component_optimal
        self.path_dir = path_dir
        self.disease_name = disease_name

    def get_info(self):
        print(f'disease: {self.disease}')
        print(f'disease_name: {self.disease_name}')

    def __repr__(self):
        return f'{self.disease}_{self.disease_name}'
    
    def __str__(self):
        return self.__repr__()

    def convert_dict_to_zip(self, dataframes_dict, zip_filename):
        with zipfile.ZipFile(zip_filename, 'w') as zip_file:
            for name, df in dataframes_dict.items():
                # Convert the DataFrame to bytes using parquet
                # table = pa.Table.from_pandas(df)
                parquet_filename = f'{name}.parquet'
                # pq.write_table(table, parquet_filename)
                data_bytes = df.to_parquet()
                zip_file.writestr(parquet_filename, data_bytes)

    def read_zip_to_dict(self, zip_filename):
        dataframes_dict = {}
        try:
            with zipfile.ZipFile(zip_filename, 'r') as zip_file:
                for file_name in zip_file.namelist():
                    if file_name.endswith('.parquet'):
                        parquet_file = zip_file.open(file_name)
                        table = pq.read_table(parquet_file)
                        df = table.to_pandas()
                        df_name = file_name.replace('.parquet', '')
                        dataframes_dict[df_name] = df
            return dataframes_dict
        except Exception as e:
            print(e)
            return None


class GeneListFromExpressionMtx(BasePLS):

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
        # zero loadings that are <higher_percentile or >lower_percentile
        return ((permute_perc_high>0) & (permute_perc_low<0))
        # return np.where((true_coef<permute_perc_high) | (true_coef>permute_perc_low))[0]
        
    def regress_out(self, X, y, n_bootstrap=1, threshold=95):
        print('regressing...')
        scaler = StandardScaler(copy=False)
        y = scaler.fit_transform(y)
        model_params = {'fit_intercept': True}
        lr_model = Ridge
        model = lr_model(**model_params)
        X = X.astype('float32')
        model.fit(y, X)
        bs_coefs = Parallel(n_jobs=10, verbose=10)(delayed(self._do_lr_bootstrap)(X, y, iter, lr_model, **model_params) for iter in range(n_bootstrap))
        mask_idx = self._thresh_bs(bs_coefs, model.coef_, threshold=threshold)[:,0]
        print(f'pmi unrelated to {mask_idx.shape} genes, regressing the rest out')
        X_res = X
        X_res[:, ~mask_idx] = X[:, ~mask_idx] -  model.predict(y)[:, ~mask_idx]
        return X_res

    def _fit_model_and_score(self, scPLS_optimal, X, y):
        try:
            check_is_fitted(scPLS_optimal)
        except:
            scPLS_optimal.fit(X, y)

        arr = np.array([pearsonr(x, y).statistic for x, y in zip(scPLS_optimal.x_scores_.T, scPLS_optimal.y_scores_.T)])
        return np.append(arr, roc_auc_score(y, scPLS_optimal.predict(X)))
    
    def _fit_model_and_loadings(self, scPLS_optimal, pca, X, y, num_components, X_reg = None, true_loadings=None):
        try:
            check_is_fitted(scPLS_optimal)
        except:
            scPLS_optimal.fit(X, y)

        decomp = []
        for i in range(num_components):
            loadings = scPLS_optimal.x_loadings_[:, i]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # k = i+1
                # x_rotations_ = np.dot(
                #     scPLS_optimal.x_weights_[:, :k],
                #     pinv2(np.dot(scPLS_optimal.x_loadings_[:, :k].T, scPLS_optimal.x_weights_[:, :k]), check_finite=False),
                # )
                # coef_ = np.dot(x_rotations_, scPLS_optimal.y_loadings_[:, :k].T)
                # y_pred = X.dot(coef_).squeeze()
                # y_pred_par = X.dot(loadings)
                # stat = pearsonr(y_pred, y_pred_par).statistic 
                # if stat<0:
                #     loadings = loadings * -1.0
                decomp.append(loadings)
        del(X)
        loadings = np.array(np.squeeze(decomp))
        loadings = pca.inverse_transform(loadings)

        if X_reg is  not None:
            # An attempt to identify gene deviation direction for each gene module
            y_pred_pca = X_reg.dot(loadings.T)
            for comp in range(num_components):
                # y_pred_pca = X_reg.dot(loadings.T)
                stat = pearsonr(y, y_pred_pca[:,comp]).statistic
                if stat<0:
                    loadings[comp, :] = -1.0*loadings[comp, :]

        if true_loadings is not None:
            # To align bootstrap compoentns. 
            for comp in range(num_components):
                stat = np.dot(loadings[comp, :],true_loadings[:, comp])/(np.linalg.norm(loadings[comp, :])*np.linalg.norm(true_loadings[:, comp])) #pearsonr(loadings[comp, :], true_loadings[:, comp])
                if stat<0: #stat.statistic<0
                    loadings[comp, :] = -1.0 * loadings[comp, :]

        return loadings.T

    def pls_topGenes(self, data_shuffled, target, num_components, covariate=None, n_bootstrap=1000, threshold=95):
        X = data_shuffled.X.toarray().astype('float32')
        y = data_shuffled.obs[target]
        
        scaler = StandardScaler(copy=False)
        X_reg = scaler.fit_transform(X)

        if covariate:
            y_cov = data_shuffled.obs[covariate].fillna(0).values.reshape(-1, 1)
            X_reg = self.regress_out(X, y_cov, n_bootstrap=n_bootstrap, threshold=threshold)
            scaler = StandardScaler(copy=False)
            X_reg = scaler.fit_transform(X_reg)
        del(X)
        pca_comps = min(500, X_reg.shape[0])
        pca = PCA(n_components=pca_comps, copy=True, random_state=0)
        X_ = pca.fit_transform(X_reg)
        
        scPLS_optimal = PLSRegression(n_components=num_components, scale=False, copy=True)
        loadings = self._fit_model_and_loadings(scPLS_optimal, pca, X_, y, num_components, X_reg)
        return loadings, self._fit_model_and_score(scPLS_optimal, X_, y), X_reg

    def paralledBootstrap(self,
                        ge_csr_array,
                        ge_obs,
                        n_components,
                        target, 
                        rng, 
                        true_loadings,
                        ):
        # select bootstrap data
        idx_bootstrap = []
        for i in set(ge_obs):
            idx_class = np.where(ge_obs == i)[0]
            n_class = idx_class.shape[0]
            idx_class_bootstrap = rng.integers(0, n_class, n_class)
            idx_bootstrap.extend(idx_class[idx_class_bootstrap])
        X_bootstrap = ge_csr_array[idx_bootstrap, :]
        del(ge_csr_array) 
        y_bootstrap = ge_obs[idx_bootstrap]
        # calculate model loadings of null model 
        import sys
        print(sys.getsizeof(X_bootstrap))
        
        # del(X_bootstrap)
        pca_comps = min(500, X_bootstrap.shape[0])
        pca = PCA(n_components=pca_comps, copy=True, random_state=0)
        X_bootstrap_ = pca.fit_transform(X_bootstrap)
      
        del(X_bootstrap)

        scPLS_optimal = PLSRegression(n_components=n_components, scale=False, copy=True)
        bootstrap_loadings = self._fit_model_and_loadings(scPLS_optimal, pca, X_bootstrap_, y_bootstrap, n_components, true_loadings = true_loadings)
        bootstrap_score = self._fit_model_and_score(scPLS_optimal, X_bootstrap_, y_bootstrap)

        del(X_bootstrap_)
        del(y_bootstrap)
        return bootstrap_loadings, bootstrap_score[-1]

    def do_bootstrap(   
            self,
            ge_csr_array,
            ge_obs,
            n_components,
            bootstrap_idx,
            n_bootstrap = 500,                    
            target = 'diagnosis',
            seed = None,
            true_loadings = None,
            ):
        np.random.seed(seed)
        if((100*bootstrap_idx/n_bootstrap)%10==0):
            print(f'{100*bootstrap_idx/n_bootstrap:.0f}% complete')
        rng = default_rng(bootstrap_idx+seed)

        bootstrap_loadings, bootstrap_score = self.paralledBootstrap(ge_csr_array, ge_obs, n_components, target, rng, true_loadings,)
        del(ge_csr_array)
        del(ge_obs)
        bootstrap_loadings_csr = sparse.csr_matrix(bootstrap_loadings)
        gc.collect()
        return bootstrap_loadings_csr, bootstrap_score

    def do_zeroed_loadings(self, loadings, loading_median, zero_threshold = 5):
        # zero_threshold: fraction of bootstrap loadings that need to cross zero to zero out feature
        loadings_median_zeroed = loading_median.copy()
        loadings_zeroed = loadings.copy()
        
        limits = np.percentile(loadings_zeroed, q=[zero_threshold,100-zero_threshold], axis=0)

        # boolean mask for features where the sign of one of the limits is of opposite sign from the median
        # True indicates feature should be dropped
        zero_mask = np.abs(np.sign(limits).sum(axis=0))<2

        loadings_median_zeroed[zero_mask] = 0
        loadings_zeroed[:,zero_mask] = 0
        for i in range(zero_mask.shape[1]):
            print(f"component {i}: {zero_mask[:,i].sum()} features zeroed")
        return loadings_median_zeroed, loadings_zeroed

    def plot_top_genes(self, celltype, num_components, gene_symbols, loadings_median_zeroed, loadings_zeroed, NN, N_TOP, pls_bs_GE_path = None):
        optimal_n_comp = num_components

        f, axs = plt.subplots(nrows=optimal_n_comp, figsize=(25,optimal_n_comp*12))
        if(optimal_n_comp==1):
            axs = [axs]

        # plot the significant genes for each component
        for comp, ax in enumerate(axs):
            sort_idx = np.argsort(np.abs(loadings_median_zeroed[:,comp]))[::-1]
            # plot top genes
            ax.violinplot(loadings_zeroed[:,:,comp][:,sort_idx][:,NN:(NN+N_TOP)], showextrema=False, widths=0.8, showmedians=True, points=200)
            ax.set_xticks(np.arange(0,N_TOP))
            ax.set_xticklabels(labels=gene_symbols[sort_idx][NN:(NN+N_TOP)], rotation=65, fontsize=14)
            ax.hlines(y=0, xmin=0, xmax=N_TOP+1, colors='k')
            ax.set_xlim([0,N_TOP+1])
            ax.set_title('cellType_{}_component_{}.png'.format(celltype, comp))
        if pls_bs_GE_path:
            f.savefig(pls_bs_GE_path+'cellType_{}.png'.format(celltype))
        plt.close(f)

    def endToEndPlsAndBootstrap(self,
                                data,
                                celltype, 
                                num_components,
                                n_cells_max = None, 
                                n_bootstrap = 500,
                                target = 'diagnosis',
                                NN = 0,
                                N_TOP = 100,
                                pls_bs_GE_path = None,
                                n_jobs = 5,
                                zero_threshold = 5,
                                seed = 42,
                                save_fig = False,
                                covariate = None,
                                cov_threshold = 95,
                                ):
        print(celltype.upper())
        
        if n_cells_max and (data.shape[0]>n_cells_max):
            sc.pp.subsample(data, n_obs=n_cells_max, random_state=seed)
            
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

        del(data_PD)
        del(data_ctrl)
        true_loadings, true_scores, X_reg = self.pls_topGenes(data_shuffled.copy(), 
                                                              target, 
                                                              num_components, 
                                                              covariate=covariate, 
                                                              n_bootstrap=n_bootstrap, 
                                                              threshold=cov_threshold)
        ge_csr_array = X_reg #data_shuffled.X
        ge_obs = data_shuffled.obs['diagnosis'].values  # need to send in the difference group as well if true
        data_vars = data_shuffled.var.index.tolist()
        del(data_shuffled)
        bootstrap_loadings_scores = Parallel(n_jobs=n_jobs, verbose = 10)(delayed(self.do_bootstrap) \
                                   (ge_csr_array, ge_obs, num_components, idx, n_bootstrap, target, seed, true_loadings) \
                                   for idx in range(n_bootstrap))

        bootstrap_loadings_csr = [bsc for bsc, _ in bootstrap_loadings_scores]
        bootstrap_scores = [[s] for _, s in bootstrap_loadings_scores]
        bs_score_q1, bs_score_q2, bs_score_q3 = np.percentile(bootstrap_scores, [zero_threshold, 50, 100-zero_threshold], axis=0)
        print('finished bootstrapping')
        bootstrap_loadings = [bsl.todense() for bsl in bootstrap_loadings_csr]
        del(bootstrap_loadings_csr)
        bootstrap_loadings = np.array(bootstrap_loadings)
        bs_q1, bootstrap_median, bs_q3 = np.percentile(bootstrap_loadings, [zero_threshold, 50, 100-zero_threshold], axis=0)
        loadings_median_zeroed, _ = self.do_zeroed_loadings(bootstrap_loadings, bootstrap_median, zero_threshold)
        gene_symbols = np.array(data_vars)
        
        if pls_bs_GE_path:
            log.debug('Writing pls data...')
            file_path = pls_bs_GE_path+'.zip'
            log.debug(f'Writing at {file_path}')
            if not os.path.isfile(file_path):
                writer = zipfile.ZipFile(file_path, 'w')
            else:
                writer = zipfile.ZipFile(file_path, 'a')

            parquet_filename_s = f'{celltype}_score.parquet'
            score_bytes = pd.DataFrame(true_scores, columns = ['score']).to_parquet()
            ## Write the bytes to the zip file
            writer.writestr(parquet_filename_s, score_bytes)

            parquet_filename_s = f'{celltype}_bs_rocauc.parquet'
            score_bytes = pd.DataFrame(np.array([bs_score_q1, bs_score_q2, bs_score_q3, [true_scores[-1]]]).T, columns = ['q1', 'median', 'q3','true_sc']).to_parquet()
            ## Write the bytes to the zip file
            writer.writestr(parquet_filename_s, score_bytes)

            topGeneDf = pd.DataFrame()
            for n_comp in range(num_components):
               sort_idx = np.argsort(np.abs(loadings_median_zeroed[:,n_comp]))[::-1]
               genes = gene_symbols[sort_idx]
               _temp_df = pd.DataFrame(list(chain(zip([n_comp]*len(genes), 
                                                       genes, 
                                                       loadings_median_zeroed[sort_idx,n_comp], 
                                                       true_loadings[sort_idx,n_comp],
                                                       bootstrap_median[sort_idx,n_comp],
                                                       bs_q1[sort_idx,n_comp],
                                                       bs_q3[sort_idx,n_comp],
                                                       ))), 
                                                       columns = ['component', 'gene', 'loading', 'true_loading', 'nz_loading', 'bs_q1', 'bs_q3'], 
                                                       index = [celltype]*len(genes))
               topGeneDf = pd.concat([topGeneDf, _temp_df])
               
            parquet_filename = f'{celltype}.parquet'
            df_bytes = topGeneDf.to_parquet()
            # Write the bytes to the zip file
            writer.writestr(parquet_filename, df_bytes)

            writer.close()
        # if save_fig:
        #     self.plot_top_genes(celltype, num_components, gene_symbols, loadings_median_zeroed, loadings_zeroed, NN, N_TOP, pls_bs_GE_path)
    
    @classmethod
    def pls_bs_GE_path(cls, path_dir, disease, disease_name, n_bootstrap, zero_threshold, seed):
        return path_dir+f'{disease}/{disease_name}/DEGenes/pls_{n_bootstrap}bs_{zero_threshold}thresh_{seed}seed'
    
    def run(self, 
            adata,
            n_cells_max = None,
            n_bootstrap = 500,
            n_jobs = 6, 
            zero_threshold = 5, 
            N_TOP = 100, 
            seed = 42,
            save_fig = False,
            celltypes = None,
            covariate = None):
        pls_bs_GE_path = GeneListFromExpressionMtx.pls_bs_GE_path(self.path_dir, 
                                                                  self.disease, 
                                                                  self.disease_name, 
                                                                  n_bootstrap, 
                                                                  zero_threshold, 
                                                                  seed)
        log.debug(f'PLS BS GE PATH {pls_bs_GE_path}')
        MakeDirs.make_dirs(pls_bs_GE_path, include_end=False)
        celltypes = celltypes if celltypes else self.optimalMapping.keys()
        for celltype in celltypes:
            data = adata[adata.obs.cell_type == celltype].copy()
            sc.pp.filter_genes(data, min_cells=int(1+data.shape[0]/1000))

            # n_cells_max = min(n_cells_max,data.shape[1]+5000) if n_cells_max else data.shape[1]+5000
            print(f'maximum cells being considered for {celltype} is {n_cells_max}')
                            
            n_comp = self.optimalMapping[celltype]
            _ = self.endToEndPlsAndBootstrap(data, 
                                             celltype, 
                                             n_comp, 
                                             n_cells_max=n_cells_max, 
                                             n_bootstrap=n_bootstrap, 
                                             pls_bs_GE_path=pls_bs_GE_path, 
                                             n_jobs=n_jobs, 
                                             zero_threshold=zero_threshold, 
                                             N_TOP=N_TOP, 
                                             seed=seed,
                                             save_fig=save_fig,
                                             covariate=covariate)
            del(data)
            gc.collect()


class PermutationGeneList(GeneListFromExpressionMtx):

    def do_permutations(self,
                        ge_csr_array,
                        ge_obs,
                        n_components,
                        permutation_idx,
                        target = 'diagnosis',
                        seed = 42,
                        frac_permute=1,
                        y_cov = None,
                        n_bootstrap=1000,
                        cov_threshold=95,
                        ): 
        ge_csr_array = ge_csr_array.copy() 
        scaler = StandardScaler(copy=False)
        X = scaler.fit_transform(ge_csr_array)
        # del(ge_csr_array)

        y_perm = ge_obs
        np.random.seed(seed+permutation_idx)
        np.random.shuffle(y_perm)

        pca_comps = min(500, X.shape[0])
        pca = PCA(n_components=pca_comps, copy=False)
        X_ = pca.fit_transform(X)

        scPLS_optimal = PLSRegression(n_components=n_components, scale=False, copy=True)

        permutation_loadings = self._fit_model_and_loadings(scPLS_optimal, pca, X_, y_perm, n_components, X)
        permutation_scores = self._fit_model_and_score(scPLS_optimal, X_, y_perm)
        return permutation_scores, permutation_loadings
    
    def _shuffle_row(self, row):
        np.random.shuffle(row)
        return row

    def do_permutations_columnwise( self,
                                    ge_csr_array,
                                    ge_obs,
                                    n_components,
                                    permutation_idx,
                                    seed = 42,
                                    ):
        X = ge_csr_array #.toarray()
        del(ge_csr_array)
        np.random.seed(seed+permutation_idx)
        X = np.apply_along_axis(self._shuffle_row, 1, X)
        scPLS_optimal = PLSRegression(n_components=n_components, scale=False, copy=True)
        permutation_loadings = self._fit_model_and_loadings(scPLS_optimal, X, ge_obs, n_components)
        return permutation_loadings

    def write_permLoadings(self, pls_perm_writer, gene_symbols, p_loading, num_components, celltype):
        topGeneDf = pd.DataFrame()
        for n_comp in range(num_components):
            _temp_df = pd.DataFrame( list(chain(zip([n_comp]*len(gene_symbols), 
                                                    gene_symbols, 
                                                    p_loading[:,n_comp], 
                                                    abs(p_loading[:,n_comp])))), 
                                                    columns = ['component', 'gene', 'loading', 'abs_loading'], 
                                                    index = [celltype]*len(gene_symbols))
            topGeneDf = pd.concat([topGeneDf, _temp_df])

        parquet_filename = f'{celltype}.parquet'
        data_bytes = topGeneDf.to_parquet()
        pls_perm_writer.writestr(parquet_filename, data_bytes)
        
    def endToEndPlsAndPermute(self,
                                ge_array, 
                                ge_obs, 
                                gene_symbols,
                                celltype, 
                                num_components,
                                target = 'diagnosis',
                                pls_perm_writer = None,
                                seed = 42,
                                perm_idx = None,
                                columnwise = False,
                                y_cov = None,
                                n_bootstrap=1000,
                                cov_threshold=95,
                                ):
        print(celltype.upper())
        
        if not columnwise:
            # perm_loadings = self.do_permutations(ge_array, ge_obs, num_components, perm_idx, target, seed )
            perm_scores, perm_loadings = self.do_permutations(ge_array, 
                                                              ge_obs, 
                                                              num_components, 
                                                              perm_idx, 
                                                              target, 
                                                              seed, 
                                                              y_cov = y_cov, 
                                                              n_bootstrap = n_bootstrap, 
                                                              cov_threshold = cov_threshold )
        else:
            perm_loadings = self.do_permutations_columnwise(ge_array, ge_obs, num_components, perm_idx, seed)
        del(ge_array)
        del(ge_obs)
        perm_scores = np.array(perm_scores)
        if pls_perm_writer:
            perm_score_df = pd.DataFrame(perm_scores, columns=['score'])
            parquet_filename = f'{celltype}_score.parquet'
            data_bytes = perm_score_df.to_parquet()
            pls_perm_writer.writestr(parquet_filename, data_bytes)
            self.write_permLoadings(pls_perm_writer, gene_symbols, perm_loadings, num_components, celltype)

    def permutation_analysis(self, ge_array, ge_obs, gene_symbols, idx, num_perm, pls_perm_GE_path, seed=42, 
                            #  n_cells_max=None, 
                            #  is_data_dict=False, 
                             celltype = None, columnwise = False):
        if((100*idx/num_perm)%2==0):
            print(f'{100*idx/num_perm:.0f}% of {celltype} complete')
        file_path = pls_perm_GE_path+'_'+str(idx)+'.zip'
        
        # for celltype in celltypes:
        if not os.path.isfile(file_path):
            writer = zipfile.ZipFile(file_path, 'w')
        else:
            writer = zipfile.ZipFile(file_path, 'a')
        
        # data = adata[celltype]

        n_comp = self.optimalMapping[celltype]
        _ = self.endToEndPlsAndPermute(ge_array, 
                                       ge_obs, 
                                       gene_symbols,
                                        celltype, 
                                        n_comp, 
                                        # n_cells_max=n_cells_max, 
                                        pls_perm_writer = writer, 
                                        seed = seed,
                                        perm_idx = idx,
                                        columnwise = columnwise )
        # del(data)
        writer.close()
        gc.collect()

    @classmethod
    def pls_perm_GE_path(cls, path_dir, disease, disease_name, seed, permute_factor):
        columnwise = cls.isColumnwise(permute_factor)
        if not columnwise:
            return path_dir + f'{disease}/{disease_name}/DEGenes_permute/pls_{seed}seed_permute'
        else:
            return path_dir + f'{disease}/{disease_name}/DEGenes_permutecolumn/pls_{seed}seed_permute'

    @staticmethod
    def isColumnwise(permute_factor):
        return True if permute_factor=='gene' else False
    
    def run(self,
            adata,
            num_perm = 100,
            seed=42,
            n_cells_max = None,
            n_jobs = 5,
            #is_data_dict = False,
            celltypes = None,
            permute_factor = 'label',
            start_idx = 0,
            covariate = None,
            n_bootstrap = 1000, 
            cov_threshold = 95,
            ):
        '''permute_factor: choose from gene/label. Gene permutation permutes the X columnwise. Label shuffles y'''
        columnwise = self.isColumnwise(permute_factor)
        pls_perm_GE_path = self.__class__.pls_perm_GE_path(self.path_dir, self.disease, self.disease_name, seed, permute_factor)
        MakeDirs.make_dirs(pls_perm_GE_path, include_end=False)
        data_dict = {}
        celltypes = celltypes or list(self.optimalMapping.keys())

        for celltype in celltypes:
            data = adata[adata.obs.cell_type == celltype].copy()
            sc.pp.filter_genes(data, min_cells=int(1+data.shape[0]/1000))
            
            print(f'maximum cells being considered for {celltype} is {n_cells_max}')
            if n_cells_max and (data.shape[0]>n_cells_max):
                sc.pp.subsample(data, n_obs=n_cells_max, random_state=seed)
            data_dict[celltype] = data

        del(adata)
        for celltype in celltypes:
            data = data_dict[celltype]
            data_PD = data[data.obs['diagnosis'] > 0]
            data_ctrl = data[data.obs['diagnosis'] <= 0 ]
            n_PD = data_PD.shape[0]
            n_ctrl = data_ctrl.shape[0]
            del(data)
            if(n_PD > n_ctrl):
                sc.pp.subsample(data_PD, fraction=n_ctrl/n_PD, random_state=0)
            else:
                sc.pp.subsample(data_ctrl, fraction=n_PD/n_ctrl, random_state=0)
            data_sub = ad.concat([data_PD, data_ctrl])
            data_shuffled = data_sub #shuffle(data_sub, random_state=0)
            del(data_PD)
            del(data_ctrl)

            ge_array = data_shuffled.X.toarray().astype('float32')
            ge_obs = data_shuffled.obs['diagnosis'].values.astype('int32')  # need to send in the difference group as well if true
            data_vars = data_shuffled.var.index.tolist()
            gene_symbols = np.array(data_vars)
            if covariate:
                y_cov = data_shuffled.obs[covariate].fillna(0).values.reshape(-1, 1) if covariate else None
                ge_array = self.regress_out(ge_array, y_cov, n_bootstrap=n_bootstrap, threshold=cov_threshold)
                
            del(data_shuffled)
            Parallel(n_jobs=n_jobs, verbose=10)(delayed(self.permutation_analysis)(ge_array, ge_obs, gene_symbols, idx, num_perm, pls_perm_GE_path, seed, celltype = celltype, columnwise=columnwise) \
                    for idx in range(start_idx, num_perm))


class ThresholdPredictiveComponents(BasePLS):
    HIGH_PERCENTILE = 95

    def __init__(self, disease, celltype_component_optimal, path_dir, disease_name, high_percentile=HIGH_PERCENTILE, low_percentile = None, permute_factor='label'):
        super().__init__(disease, celltype_component_optimal, path_dir, disease_name)
        self.high_percentile = high_percentile
        self.low_percentile = low_percentile if low_percentile else 100-self.high_percentile
        self.permute_factor = permute_factor

    @classmethod
    def pls_bs_thresh_path(cls, path_dir, disease, disease_name, n_bootstrap, zero_threshold, seed, high_per=None, low_per=None, permute_factor='label'):
        high_per = high_per if high_per else ThresholdPredictiveComponents.HIGH_PERCENTILE
        low_per = low_per if low_per else (100 - high_per)
        high_per = str(high_per).replace('.', '_')
        low_per = str(low_per).replace('.', '_')
        pls_bs_GE_path = GeneListFromExpressionMtx.pls_bs_GE_path(path_dir, disease, disease_name, n_bootstrap, zero_threshold, seed)
        return pls_bs_GE_path + f'_thresholded_{high_per}highper_{low_per}lowper_{permute_factor}permutefactor'
    
    def pls_paths(self, n_bootstrap, zero_threshold, seed, loading_path = None):
        path_args = (self.path_dir, 
                    self.disease, 
                    self.disease_name, 
                    n_bootstrap, 
                    zero_threshold, 
                    seed)
        loading_path = loading_path if loading_path else self.path_dir
        pls_bs_GE_path = GeneListFromExpressionMtx.pls_bs_GE_path(loading_path, 
                                                                  self.disease, 
                                                                  self.disease_name, 
                                                                  n_bootstrap, 
                                                                  zero_threshold, 
                                                                  seed)    # should be same as score path. TODO
        pls_score_path = GeneListFromExpressionMtx.pls_bs_GE_path(*path_args)
        pls_bs_thresh_path = ThresholdPredictiveComponents.pls_bs_thresh_path(*path_args, self.high_percentile, self.low_percentile, self.permute_factor)
        return pls_bs_GE_path, pls_score_path, pls_bs_thresh_path

    def threshold_loadings(self, celltype, perm_num=100, n_bootstrap = 500, zero_threshold = 5, seed=42, loading_path = None):
        print(celltype.upper())
        # try:
        pls_bs_GE_path, pls_score_path, pls_bs_thresh_path = self.pls_paths(n_bootstrap, zero_threshold, seed, loading_path)
        zip_file = zipfile.ZipFile(pls_score_path+'.zip', 'r')
        parquet_file = zip_file.open(f'{celltype}_score.parquet')
        pls_df_raw = pq.read_table(parquet_file).to_pandas()
        true_pls_df = pls_df_raw.rename(columns={0:'score'})
        true_pls_df['component'] = range(true_pls_df.shape[0])
        parquet_file.close()
        zip_file.close()

        perm_pls_df = pd.DataFrame()
        perm_path_pre = PermutationGeneList.pls_perm_GE_path(self.path_dir, self.disease, self.disease_name, seed, self.permute_factor)
        for i in range(perm_num):
            try:
                print(f'Iteration #: {i}, {celltype.upper()}')
                perm_path_pre_ = perm_path_pre + f'_{i}.zip'
                zip_file = zipfile.ZipFile(perm_path_pre_, 'r')
                parquet_file = zip_file.open(f'{celltype}_score.parquet')
                pls_df_raw = pq.read_table(parquet_file).to_pandas()
                pls_df_raw = pls_df_raw.rename(columns={0:'score'})
                pls_df_raw['component'] = range(pls_df_raw.shape[0])
                pls_df_raw['iter'] = i
                perm_pls_df = pd.concat([perm_pls_df, pls_df_raw.pivot_table(columns = 'component', values = 'score', index = 'iter')])
                parquet_file.close()
                zip_file.close()
            except:
                print(f'skipping iter #{i}')


        scores_shifted = perm_pls_df.abs().values
        true_scores = true_pls_df.score.abs().values.copy()
        permute_perc_high = scoreatpercentile(scores_shifted, per = self.high_percentile, axis =0)
        permute_perc_low = scoreatpercentile(scores_shifted, per = self.low_percentile, axis =0)

        drop_comp = np.where((true_scores<np.maximum(permute_perc_high, permute_perc_low) )& (true_scores>np.minimum(permute_perc_high, permute_perc_low)))[0]
        # drop_comp = np.where((true_scores<permute_perc_high))[0]

        if len(true_scores)-1 in drop_comp:
            print(drop_comp)
            print(true_scores, len(true_scores))
            print(f'{celltype} is being dropped because auroc does not pass the threshold')
            return

        zip_file = zipfile.ZipFile(pls_bs_GE_path+'.zip', 'r')
        parquet_file = zip_file.open(f'{celltype}.parquet')
        table = pq.read_table(parquet_file)
        zip_file.close()

        pls_bs_df = table.to_pandas()
        pls_bs_df.reset_index(inplace=True)
        pls_bs_df.drop(pls_bs_df[pls_bs_df.component.isin(drop_comp)].index, inplace=True)
        pls_bs_df.set_index('index', inplace=True)
        if pls_bs_thresh_path:
            file_path = pls_bs_thresh_path+'.zip'
            with zipfile.ZipFile(file_path, 'a') as writer:
                parquet_filename = f'{celltype}.parquet'
                data_bytes = pls_bs_df.to_parquet()
                writer.writestr(parquet_filename, data_bytes)
        # except:
        #     print('Did not work. Likely celltype not present')

    def run(self, celltypes=None, perm_num=100, seed=42, n_jobs=1, n_bootstrap=500, zero_threshold=5, loading_path=None):
        celltypes = celltypes if celltypes else self.optimalMapping.keys()
        _, _, pls_bs_thresh_path = self.pls_paths(n_bootstrap, zero_threshold, seed)
        writer = zipfile.ZipFile(pls_bs_thresh_path+'.zip', 'a')
        writer.close()
        _ = Parallel(n_jobs=n_jobs, verbose = 5)(delayed(self.threshold_loadings)(
                                                                        celltype, 
                                                                        perm_num = perm_num,
                                                                        seed=seed,
                                                                        loading_path=loading_path,
                                                                        n_bootstrap=n_bootstrap,
                                                                        ) for celltype in celltypes)


class ThresholdPredictiveWeights(BasePLS):

    HIGH_PERCENTILE = 95

    def __init__(self, disease, celltype_component_optimal, path_dir, disease_name, high_percentile=HIGH_PERCENTILE, low_percentile = None, permute_factor='label'):
        super().__init__(disease, celltype_component_optimal, path_dir, disease_name)
        self.high_percentile = high_percentile
        self.low_percentile = low_percentile if low_percentile else 100-self.high_percentile
        self.permute_factor = permute_factor

    def _read_loadings(self, perm_path, perm_idx, component, celltype, gene_list):
        try:
            zip_file = zipfile.ZipFile(perm_path+'_'+str(perm_idx)+'.zip', 'r')
            parquet_file = zip_file.open(f'{celltype}.parquet')
            table = pq.read_table(parquet_file)
            bs_all = table.to_pandas()
            bs_all = bs_all[bs_all.component==component]
            bs_all = bs_all.reset_index().set_index('gene').loc[gene_list]
            loadings = bs_all.loading.values
            zip_file.close()
            return loadings
        except:
            print(f'Bad zip file, could not read {perm_path} {perm_idx}')
            return None

    def _threshold_pls_loadings(self, _series, celltype, gene_list, perm_num = 100, seed=42):
        perm_path = PermutationGeneList.pls_perm_GE_path(self.path_dir, self.disease, self.disease_name, seed, self.permute_factor)
        loadings_permute = []
        component = _series.name
        loadings = _series.values
        print('reading ', celltype, component)
        print('path ', perm_path)
        loadings_permute = Parallel(n_jobs=1)(delayed(self._read_loadings)(perm_path, perm_idx, component, celltype, gene_list) \
                                               for perm_idx in range(perm_num))
        loadings_permute = list(filter(lambda x: x is not None, loadings_permute))
        print('thresholding...')
        # print('loadings mean ', np.mean(loadings_permute))
        loadings_shifted = loadings_permute # - np.mean(loadings_permute)
        permute_perc_high = scoreatpercentile(loadings_shifted, per = self.high_percentile, axis =0)
        permute_perc_low = scoreatpercentile(loadings_shifted, per = self.low_percentile, axis =0)
        # zero loadings that are <90 percentile or >10 percentile
        loadings[(loadings<np.maximum(permute_perc_high, permute_perc_low) )& (loadings>np.minimum(permute_perc_high, permute_perc_low))]=0
        return loadings

    @classmethod
    def pls_bs_thresh_path(cls, path_dir, disease, disease_name, n_bootstrap, zero_threshold, seed, high_per=None, low_per=None, permute_factor='label'):
        high_per = high_per if high_per else ThresholdPredictiveWeights.HIGH_PERCENTILE
        low_per = low_per if low_per else (100 - high_per)
        high_per = str(high_per).replace('.', '_')
        low_per = str(low_per).replace('.', '_')
        pls_bs_GE_path = GeneListFromExpressionMtx.pls_bs_GE_path(path_dir, disease, disease_name, n_bootstrap, zero_threshold, seed)
        return pls_bs_GE_path + f'_thresholded_{high_per}highper_{low_per}lowper_{permute_factor}permutefactor'
    
    def pls_paths(self, n_bootstrap, zero_threshold, seed):
        path_args = (self.path_dir, 
                    self.disease, 
                    self.disease_name, 
                    n_bootstrap, 
                    zero_threshold, 
                    seed)
        pls_bs_GE_path = GeneListFromExpressionMtx.pls_bs_GE_path(*path_args)
        pls_bs_thresh_path = ThresholdPredictiveWeights.pls_bs_thresh_path(*path_args, self.high_percentile, self.low_percentile, self.permute_factor)
        return pls_bs_GE_path, pls_bs_thresh_path

    def threshold_loadings(self, celltype, perm_num=100, n_bootstrap = 500, zero_threshold = 5, seed=42):
        print(celltype.upper())
        # try:
        pls_bs_GE_path, pls_bs_thresh_path = self.pls_paths(n_bootstrap, zero_threshold, seed)
        zip_file = zipfile.ZipFile(pls_bs_GE_path+'.zip', 'r')
        parquet_file = zip_file.open(f'{celltype}.parquet')
        table = pq.read_table(parquet_file)
        pls_bs_df = table.to_pandas()
        pls_bs_df['true_loading_thresh']  = pls_bs_df.groupby('component').true_loading.transform( 
                                                                                lambda x: self._threshold_pls_loadings(x, 
                                                                                **{'celltype': celltype, 
                                                                                'gene_list': pls_bs_df[pls_bs_df.component==x.name].gene.values,
                                                                                'perm_num':perm_num, 
                                                                                'seed':seed,
                                                                                    }))
        # print('res', res)
        # pls_bs_df['true_loading_thresh']  = pls_bs_df['component'].map(res)
        if pls_bs_thresh_path:
            file_path = pls_bs_thresh_path+'.zip'
            with zipfile.ZipFile(file_path, 'a') as writer:
                parquet_filename = f'{celltype}.parquet'
                data_bytes = pls_bs_df.to_parquet()
                writer.writestr(parquet_filename, data_bytes)
        # except:
        #     print('Did not work. Likely celltype not present')

    def run(self, celltypes=None, perm_num=100, seed=42, n_jobs=1, n_bootstrap=500, zero_threshold=5):
        celltypes = celltypes if celltypes else self.optimalMapping.keys()
        _, pls_bs_thresh_path = self.pls_paths(n_bootstrap, zero_threshold, seed)
        writer = zipfile.ZipFile(pls_bs_thresh_path+'.zip', 'a')
        writer.close()
        _ = Parallel(n_jobs=n_jobs, verbose = 5)(delayed(self.threshold_loadings)(
                                                                        celltype, 
                                                                        perm_num = perm_num,
                                                                        seed=seed,
                                                                        ) for celltype in celltypes)
        

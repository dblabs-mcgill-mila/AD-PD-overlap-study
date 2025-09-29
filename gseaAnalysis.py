import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import gseapy as gp
import os
from end2endPLS import BasePLS, ThresholdPredictiveWeights, ThresholdPredictiveComponents
import zipfile
import pyarrow.parquet as pq
import random


class GSEA(BasePLS):
    
    def rnd(self, exp = -20):
        significand = random.random()-0.5
        return significand * 10**exp

    def component_gsea( self, 
                        _ser, 
                        gene_set_list,
                        seed=42,
                        processes=4,
                        min_size=5, 
                        max_size=2000,
                        permutation_num=1000,
                        outdir=None,
                        min_nonzero_genes=1):
        comp = _ser['component'].values[0]
        gsea_terms_tmp_comp = pd.DataFrame()
        for gene_set in gene_set_list:
            loadings = _ser['true_loading'].values
            if not loadings[np.where(loadings!=0)].shape[0]>min_nonzero_genes:
                continue
            #loadings = np.array([x if x!=0 else self.rnd(np.floor(np.log10(loadings[np.where(loadings!=0)].__abs__().min()))) for x in loadings])
            gene_symbol = _ser['gene'].values
            # if not sum(loadings>0)>min_nonzero_genes:
            #     continue
            ranked_genes = pd.DataFrame(data={'genes':gene_symbol, #np.extract(loadings != 0.0, gene_symbol).tolist(), 
                                            'PLS_weights':loadings}) #np.extract(loadings != 0.0, loadings)})
            try:
                print('gsea prerank...', gene_set)
                pre_res = gp.prerank(rnk=ranked_genes,
                                gene_sets=gene_set,
                                processes=processes,
                                min_size=min_size, 
                                max_size=max_size,
                                permutation_num=permutation_num,
                                outdir=outdir,
                                seed=seed)
            
                df = pre_res.res2d
                df['component'] = comp
                df['gene_set_source'] = gene_set
                df = df[df['FDR q-val']<0.05]
                df = df[~df['NOM p-val'].isna()]
                df = df.sort_values('NES', ascending=False)
                if not df.empty:
                    gsea_terms_tmp_comp = pd.concat((gsea_terms_tmp_comp, df), axis=0)
            except Exception:
                print(f'GSEA Error: No enriched gene modules found in component {comp}')
        return gsea_terms_tmp_comp
                
    def do_gsea(self, 
                celltype, 
                gene_set_list, 
                gsea_path=None, 
                pls_bs_GE_path = None,
                processes=4,
                min_size=5, 
                max_size=2000,
                permutation_num=1000,
                outdir=None,
                seed=42,
                genesym_map = None,
                min_nonzero_genes = 1):
        print(celltype.upper())
        try:
            zip_file = zipfile.ZipFile(pls_bs_GE_path+'.zip', 'r')
            parquet_file = zip_file.open(celltype+'.parquet')
            table = pq.read_table(parquet_file)
            gsea_df = table.to_pandas()
            parquet_file.close()
            zip_file.close()
        except:
            print(f'{celltype} not present')
            return
        loadings_df = gsea_df[['component', 'true_loading', 'gene']]
        if genesym_map is not None:
            loadings_df['gene'].replace(genesym_map, inplace=True)
        gsea_terms_tmp_comp = loadings_df.groupby('component').apply(self.component_gsea, 
                                                                     **{'gene_set_list': gene_set_list, 
                                                                        'processes':processes,
                                                                        'min_size':min_size, 
                                                                        'max_size':max_size,
                                                                        'permutation_num':permutation_num,
                                                                        'outdir':outdir,
                                                                        'seed':seed,
                                                                        'min_nonzero_genes':min_nonzero_genes}).reset_index(drop=True)
        print(f'Writing at {gsea_path}')
        if not os.path.isfile(gsea_path):
            writer = zipfile.ZipFile(gsea_path, 'w')
        else:
            writer = zipfile.ZipFile(gsea_path, 'a')
        parquet_filename = f'{celltype}.parquet'
        data_bytes = gsea_terms_tmp_comp.to_parquet()
        writer.writestr(parquet_filename, data_bytes)
        writer.close()

    @classmethod
    def gsea_path(cls, path_dir, disease, disease_name, zero_threshold, n_bootstrap, seed, permute_factor, highper, lowper=None, gs_prefix = None):
        lowper = lowper if lowper else 100-highper
        highper = str(highper).replace('.', '_')
        lowper = str(lowper).replace('.', '_')
        if gs_prefix:
            return path_dir+f'{disease}/{disease_name}/DEGenes/gsea_results_{gs_prefix}_{n_bootstrap}bs_{zero_threshold}thresh_{seed}seed_{highper}_highper_{lowper}lowper_{permute_factor}permutefactor.zip'    
        return path_dir+f'{disease}/{disease_name}/DEGenes/gsea_results_{n_bootstrap}bs_{zero_threshold}thresh_{seed}seed_{highper}_highper_{lowper}lowper_{permute_factor}permutefactor.zip'
    
    def run(
            self, 
            celltypes = None,
            n_bootstrap = 500, 
            zero_threshold = 5,
            seed = 42,
            n_jobs = -1,
            gene_set_list = ['GO_Biological_Process_2021', 'WikiPathway_2021_Human', 'Panther_2016', 'Reactome_2016', 'KEGG_2021_Human'],
            #gsea params...
            processes=4,
            min_size=5, 
            max_size=2000,
            permutation_num=1000,
            outdir=None,
            genesym_map = None,
            min_nonzero_genes = 1,
            thresh_high_per = None,
            thresh_low_per = None,
            permute_factor='label',
            gs_prefix = None
            ):
        print(gene_set_list)
        thresh_high_per = thresh_high_per if thresh_high_per else ThresholdPredictiveComponents.HIGH_PERCENTILE
        celltypes = celltypes if celltypes else self.optimalMapping.keys()
        gsea_path = GSEA.gsea_path( self.path_dir, 
                                    self.disease, 
                                    self.disease_name, 
                                    zero_threshold, 
                                    n_bootstrap,
                                    seed,
                                    permute_factor,
                                    thresh_high_per,
                                    thresh_low_per,
                                    gs_prefix = gs_prefix )
        pls_bs_thresh_path = ThresholdPredictiveComponents.pls_bs_thresh_path(self.path_dir, 
                                                                           self.disease, 
                                                                           self.disease_name, 
                                                                           n_bootstrap, 
                                                                           zero_threshold, 
                                                                           seed,
                                                                           thresh_high_per,
                                                                           thresh_low_per,
                                                                           permute_factor)
        _ = Parallel(n_jobs=n_jobs, verbose = 5)(delayed(self.do_gsea)(celltype, 
                                                                   gene_set_list, 
                                                                   gsea_path = gsea_path, 
                                                                   pls_bs_GE_path = pls_bs_thresh_path,
                                                                   processes=processes,
                                                                   min_size=min_size, 
                                                                   max_size=max_size,
                                                                   permutation_num=permutation_num,
                                                                   outdir=outdir,
                                                                   seed=seed,
                                                                   genesym_map = genesym_map,
                                                                   min_nonzero_genes = min_nonzero_genes) for celltype in celltypes)
        
        


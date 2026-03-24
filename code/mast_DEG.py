import scanpy as sc
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri, Formula
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from sklearn.utils import shuffle
import gseapy as gp
import gc


def run_mast_analysis(adata, condition_col, covariates=[]):
    """
    Runs MAST DEG analysis on an AnnData object with modern rpy2 conversion.
    
    Parameters:
    -----------
    adata : sc.AnnData
        Log-normalized snRNA-seq data.
    condition_col : str
        The primary variable for comparison (e.g., 'diagnosis').
    covariates : list
        Other variables to control for (e.g., ['age', 'sex', 'pmi']).
    """
    
    # 1. Load the R libraries
    mast = importr('MAST')
    base = importr('base')
    
    # 2. Pre-calculate Cellular Detection Rate (CDR)
    # This is standard practice for high-impact snRNA-seq papers to account for dropout.
    print("Calculating Cellular Detection Rate (CDR)...")
    adata.obs['cdr'] = np.asarray((adata.X > 0).sum(axis=1)).flatten() / adata.shape[1]

    # Center and scale CDR to help model convergence
    adata.obs['cdr'] = (adata.obs['cdr'] - adata.obs['cdr'].mean()) / adata.obs['cdr'].std()
    
    # 3. Create a combined converter for Numpy and Pandas
    # This fixes the NotImplementedError for ndarray and the DeprecationWarning
    combined_converter = robjects.default_converter + pandas2ri.converter + numpy2ri.converter
    
    with localconverter(combined_converter):
        # 4. Prepare the Expression Matrix (MUST be Genes x Cells)
        # Transpose AnnData.X (Cells x Genes) to (Genes x Cells)
        if hasattr(adata.X, "toarray"):
            expr_mat = adata.X.T.toarray()
        else:
            expr_mat = adata.X.T
            
        # 5. Define Metadata Objects
        # c_data rows must match expr_mat columns (Cells)
        c_data = adata.obs
        # f_data rows must match expr_mat rows (Genes)
        f_data = pd.DataFrame(index=adata.var_names)
        f_data["primerid"] = adata.var_names
        
        print(f"Creating SingleCellAssay: {expr_mat.shape[0]} genes, {expr_mat.shape[1]} cells")
        
        # 6. Initialize the MAST object
        sca = mast.FromMatrix(expr_mat, c_data, f_data)
        
        # 7. Build the R Formula
        formula_str = f"~ {condition_col} + cdr"
        if covariates:
            formula_str += " + " + " + ".join(covariates)
        
        print(f"Fitting Hurdle Model: {formula_str}")
        f = Formula(formula_str)
        
        # 8. Run the Zero-Inflated Regression (zlm) and Likelihood Ratio Test
        zlm_fit = mast.zlm(f, sca)
    
    with localconverter(robjects.default_converter):
        summary_r = robjects.r['summary'](zlm_fit, doLRT=condition_col)
        # summary_r is an R list-like object (ListVector)
        

    with localconverter(robjects.default_converter + pandas2ri.converter):
        # -----------------------------
        # 4. Extract hurdle p-values
        # -----------------------------
        results = robjects.conversion.rpy2py(summary_r.rx2("datatable"))  # R data.table
        hurdle = results[results["component"] == "H"].copy()

        hurdle = hurdle[["primerid", "Pr(>Chisq)"]]
        hurdle = hurdle.rename(columns={
            "primerid": "gene",
            "Pr(>Chisq)": "pval"
        })

        # -----------------------------
        # 5. Extract effect sizes
        # -----------------------------
        coef_res = results[
            (results["component"] == "C") &
            (results["contrast"].str.contains(condition_col))
        ].copy()

        coef_res = coef_res[["primerid", "coef"]]

        coef_res = coef_res.rename(columns={
            "primerid": "gene",
            "coef": "logFC"
        })

        # -----------------------------
        # 6. Merge results
        # -----------------------------
        final = hurdle.merge(coef_res, on="gene", how="left")

        # -----------------------------
        # 7. Multiple testing correction
        # -----------------------------
        _, fdr, _, _ = multipletests(final["pval"], method="fdr_bh")
        final["fdr"] = fdr

        final = final[["gene", "logFC", "pval", "fdr"]]

        final = final.sort_values("fdr")

    return final.sort_values('fdr')


def _permutation_MAST(adata, condition_col, perm_idx, ros_ct, path_prefix, covariates=[],):
    np.random.seed(perm_idx)
    adata = adata[adata.obs.sample(frac=1).index]
    _res = run_mast_analysis(
        adata, 
        condition_col=condition_col, 
        covariates=covariates,
    )
    _res['cell_type'] = ros_ct
    _res['perm_idx'] = perm_idx
    _res.to_csv(DEG_PATH + f'Permute/{path_prefix}_{ros_ct}_{perm_idx}.csv')


# perform GSEA for DEGs from each cell type
def do_gsea_MAST(celltype, gene_set_list, weight_df, seed = None, do_shuffle=False):
    print(celltype.upper())
    weight_df = weight_df.dropna(axis = 0)
    weight_df = weight_df[~weight_df.logFC.isna()]
    weight_df['sign_fdr'] = np.sign(weight_df.logFC)*-np.sqrt(weight_df.fdr)

    loadings_df = weight_df[(weight_df.cell_type == celltype)&(weight_df.fdr<=1)][['sign_fdr', 'gene']]
    loadings = loadings_df['sign_fdr'].values
    gene_symbol = loadings_df['gene'].values
    gene_symbol = gene_symbol[loadings!=0]
    loadings = loadings[loadings!=0]
    if do_shuffle:
        loadings = shuffle(loadings, random_state=seed)
    ranked_genes = pd.DataFrame(data={'genes':gene_symbol, 
                                    'sign_fdr': loadings})

    print(f'cell type {celltype} has {ranked_genes.genes.unique().shape[0]} genes')
    gsea_terms_tmp_comp = pd.DataFrame()
    # perform GSEA using multiple gene set databases
    for gene_set in gene_set_list:
        
        pre_res = None
        try:
            pre_res = gp.prerank(rnk=ranked_genes,
                            gene_sets=gene_set,
                            threads=10,
                            min_size=5, 
                            max_size=1000,
                            no_plot=True,
                            permutation_num=1000,
                            outdir=None,
                            seed=1)
        except Exception:
            print(f'GSEA Error: No enriched gene modules found in celltype {celltype}')
        
        if pre_res:
            df = pre_res.res2d
            df['gene_set_source'] = gene_set
            # df = df[df['FDR q-val']<0.05]
            df = df[~df['NOM p-val'].isna()]
            df['cell_type'] = celltype
            df = df.sort_values('NES', ascending=False)
            if not df.empty:
                gsea_terms_tmp_comp = pd.concat((gsea_terms_tmp_comp, df), axis=0)
    return gsea_terms_tmp_comp



# region RUN DGE
    
PATH_DIR = '/host/socrates/pool4_z2/anweshab/OverlapAnalysis/GenePermute2/'
ADATA_PRE = '/host/socrates/pool3_z0/anwesha/OverlapAnalysis/adatas/'
DEG_PATH = PATH_DIR+'DGE/'


def run_ROSMAP():
    adata = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad')
    adata.obs['age_death'] = adata.obs.age_death.replace({'90+':'90'}).astype(float)
    adata = adata[~adata.obs.pmi.isna()]
    
    res_df = []
    for ros_ct in adata.obs.cell_type.unique():
        print('Doing ', ros_ct)
        adata_sub = adata[adata.obs.cell_type==ros_ct].copy()
        sc.pp.filter_genes(adata_sub, min_cells=3)
        _res = run_mast_analysis(
            adata_sub, 
            condition_col='diagnosis', 
            covariates=['age_death', 'pmi', 'sex'],
        )
        _res['cell_type'] = ros_ct
        res_df.append(_res)
    
    res_df = pd.concat(res_df)
    res_df.to_csv(DEG_PATH + 'AD_Rosmap.csv')


def run_ROSMAP_permutation(n_jobs = 10, num_perm = 1000, subsample_frac = 0.5, ):
    adata = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad')
    adata.obs['age_death'] = adata.obs.age_death.replace({'90+':'90'}).astype(float)
    adata = adata[~adata.obs.pmi.isna()]
    
    for ros_ct in adata.obs.cell_type.unique():
        print('Doing ', ros_ct)
        adata_sub = adata[adata.obs.cell_type==ros_ct].copy()
        sc.pp.filter_genes(adata_sub, min_cells=3)
        np.random.seed(0)

        sc.pp.subsample(adata_sub, fraction=subsample_frac)
        _ = Parallel(n_jobs = n_jobs, verbose=5)(delayed(_permutation_MAST)
                                                    (adata_sub, 
                                                     'diagnosis', 
                                                     perm_idx, 
                                                     ros_ct, 
                                                     'AD_Rosmap',
                                                     covariates=['age_death', 'pmi', 'sex'])
                                                    for perm_idx in range(num_perm))
       


def run_Kamath():
    adata = sc.read_h5ad(ADATA_PRE+'kadata_pp_agematch_Oct31.h5ad')
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.gene_ids.isin(genes_ros)]

    sample_age = pd.read_csv('Kamath_data.csv', sep='\t')
    age_dict = sample_age.set_index('Sample ID')['Age (yrs)'].to_dict()

    adata.obs['age'] = adata.obs.donor_id.map(age_dict)
    adata.obs['msex'] = adata.obs.sex.map({'male': 1, 'female':0})
    
    res_df = []
    for ctoi in adata.obs.cell_type.unique():
        print('Doing ', ctoi)
        adata_sub = adata[adata.obs.cell_type==ctoi].copy()
        sc.pp.filter_genes(adata_sub, min_cells=3)
        _res = run_mast_analysis(
            adata_sub, 
            condition_col='diagnosis', 
            covariates=['age', 'pmi', 'msex']
        )
        _res['cell_type'] = ctoi
        res_df.append(_res)
    
    res_df = pd.concat(res_df)
    res_df.to_csv(DEG_PATH + 'PD_Kamath.csv')

def run_Kamath_permutation(n_jobs = 10, num_perm = 1000, subsample_frac = 0.5,):
    adata = sc.read_h5ad(ADATA_PRE+'kadata_pp_agematch_Oct31.h5ad')
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.gene_ids.isin(genes_ros)]

    sample_age = pd.read_csv('Kamath_data.csv', sep='\t')
    age_dict = sample_age.set_index('Sample ID')['Age (yrs)'].to_dict()

    adata.obs['age'] = adata.obs.donor_id.map(age_dict)
    adata.obs['msex'] = adata.obs.sex.map({'male': 1, 'female':0})
    
    for ctoi in adata.obs.cell_type.unique():
        print('Doing ', ctoi)
        adata_sub = adata[adata.obs.cell_type==ctoi].copy()
        sc.pp.filter_genes(adata_sub, min_cells=3)
        np.random.seed(0)

        sc.pp.subsample(adata, fraction=subsample_frac)
        _ = Parallel(n_jobs = n_jobs, verbose=5)(delayed(_permutation_MAST)
                                                    (adata_sub, 
                                                     'diagnosis', 
                                                     perm_idx, 
                                                     ctoi, 
                                                     'PD_Kamath',
                                                     covariates=['age', 'pmi', 'msex'])
                                                    for perm_idx in range(num_perm))


def run_ADSEA():

    celltypes = ['sncg', 'sst', 'pvalb', 'astro', 'endo', 'opc', 'micro', 'oligo', 'l4_it', 'l5_it', 'vip']
    res_df = []
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

        sea_age_map = {'90+ years old' : 90, '78 to 89 years old': 83.5, '65 to 77 years old': 71.0, 'Less than 65 years old': 65}
        adata.obs['age'] = adata.obs['Age at death'].map(sea_age_map)

        pmi_map = {'3.2 to 5.9 hours': 3.5, '5.9 to 8.7 hours': 7.5, '8.7 to 11.4 hours': 10.5, 'Reference':float('nan')}
        adata.obs['pmi'] = adata.obs.PMI.map(lambda x: pmi_map[x]).astype('float32')
        adata.obs.drop(columns='PMI', inplace=True)

        adata.obs['msex'] = adata.obs.sex.map({'male': 1, 'female':0})
        adata = adata[~adata.obs.pmi.isna()]

        _res = run_mast_analysis(
            adata, 
            condition_col='diagnosis', 
            covariates=['age', 'pmi', 'msex']
        )
        _res['cell_type'] = celltype
        res_df.append(_res)
    
    res_df = pd.concat(res_df)
    res_df.to_csv(DEG_PATH + 'AD_SEAAD.csv')


def run_PDSm():
    adata = sc.read_h5ad(ADATA_PRE+'adata_pp_Nov6_pmi.h5ad')
    genes_ros = sc.read_h5ad(ADATA_PRE+'mathys19_pp_filtered_June21.h5ad').var.index.unique()
    adata = adata[:, adata.var.index.isin(genes_ros)]

    sm_age_dict = {'PD1':84, 'PD2':66, 'PD3':77, 'PD4':81, 'PD5':79, 'C1':93, 'C2':66, 'C3':77, 'C4':84, 'C5':88, 'C6':90}
    sm_sex_dict = {'PD1':0, 'PD2':1, 'PD3':1, 'PD4':1, 'PD5':1, 'C1':0, 'C2':1, 'C3':1, 'C4':1, 'C5':1, 'C6':1} # 0: female, 1: male
    
    adata.obs['age'] = adata.obs.patient.map(sm_age_dict)
    adata.obs['sex'] = adata.obs.patient.map(sm_sex_dict)

    res_df = []
    for ctoi in adata.obs.cell_type.unique():
        print('Doing ', ctoi)
        adata_sub = adata[adata.obs.cell_type==ctoi].copy()
        sc.pp.filter_genes(adata_sub, min_cells=3)
        _res = run_mast_analysis(
            adata_sub, 
            condition_col='diagnosis', 
            covariates=['age', 'pmi', 'sex']
        )
        _res['cell_type'] = ctoi
        res_df.append(_res)
    
    res_df = pd.concat(res_df)
    res_df.to_csv(DEG_PATH + 'PD_Smajic.csv')



# region RUN GSEA

def run_rosmap_kamath_gsea():
    gene_set_list = ['GO_Biological_Process_2023', 'GO_Molecular_Function_2023', 'GO_Cellular_Component_2023']
    deg_ros = pd.read_csv(DEG_PATH + 'AD_Rosmap.csv', index_col = 0)
    # deg_ros = deg_ros[deg_ros.fdr<0.05].dropna(subset='logFC')
    deg_kam = pd.read_csv(DEG_PATH + 'PD_Kamath.csv', index_col = 0)
    # deg_kam = deg_kam[deg_kam.fdr<0.05].dropna(subset='logFC')

    gsea_df = []
    for celltype in deg_ros.cell_type.unique():
        gsea_df.append(do_gsea_MAST(celltype, gene_set_list, deg_ros, seed = None, do_shuffle=False))
    gsea_df = pd.concat(gsea_df)

    gsea_df.to_csv(DEG_PATH + 'AD_Rosmap_GSEA.csv')

    del(gsea_df)
    gc.collect()
    gsea_df = []

    for celltype in deg_kam.cell_type.unique():
        gsea_df.append(do_gsea_MAST(celltype, gene_set_list, deg_kam, seed = None, do_shuffle=False))
    gsea_df = pd.concat(gsea_df)

    gsea_df.to_csv(DEG_PATH + 'PD_Kamath_GSEA.csv')
    


def run_seattle_smajic_gsea():
    gene_set_list = ['GO_Biological_Process_2023', 'GO_Molecular_Function_2023', 'GO_Cellular_Component_2023']
    deg_ros = pd.read_csv(DEG_PATH + 'AD_SEAAD.csv', index_col = 0)
    # deg_ros = deg_ros[deg_ros.fdr<0.05].dropna(subset='logFC')
    deg_kam = pd.read_csv(DEG_PATH + 'PD_Smajic.csv', index_col = 0)
    # deg_kam = deg_kam[deg_kam.fdr<0.05].dropna(subset='logFC')

    gsea_df = []
    for celltype in deg_ros.cell_type.unique():
        gsea_df.append(do_gsea_MAST(celltype, gene_set_list, deg_ros, seed = None, do_shuffle=False))
    gsea_df = pd.concat(gsea_df)

    gsea_df.to_csv(DEG_PATH + 'AD_SEAAD_GSEA.csv')

    del(gsea_df)
    gc.collect()
    gsea_df = []
    
    for celltype in deg_kam.cell_type.unique():
        gsea_df.append(do_gsea_MAST(celltype, gene_set_list, deg_kam, seed = None, do_shuffle=False))
    gsea_df = pd.concat(gsea_df)

    gsea_df.to_csv(DEG_PATH + 'PD_Smajic_GSEA.csv')
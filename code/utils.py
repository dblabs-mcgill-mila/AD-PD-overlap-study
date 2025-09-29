import pandas as pd
import zipfile
import os
from joblib import Parallel, delayed
from scipy._lib._bunch import _make_tuple_bunch
import numpy as np
import math
from scipy.stats import weightedtau
import gc 

SignificanceResult = _make_tuple_bunch('SignificanceResult', ['statistic', 'pvalue'], [])


class MakeDirs(object):

    @classmethod
    def _create_directory(cls, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)

    @classmethod
    def make_dirs(cls, path, include_end = True):
        if include_end:
            cls._create_directory(path)
        else:
            cls._create_directory(path.rsplit('/', 1)[0])


def convert_excel_sheets_to_parquet(excel_file_path, zip_file_path, drop_all_zeros = False):
    try:
        ad_bs_excel = pd.ExcelFile(excel_file_path+'.xlsx')
        zip_file_path = zip_file_path+'.zip'
        MakeDirs.make_dirs(zip_file_path, include_end=False)
        zip_file = zipfile.ZipFile(zip_file_path, 'w')
        for sheet in ad_bs_excel.sheet_names:
            if sheet != 'Blank':
                print(sheet)
                ad_bs_sheet = ad_bs_excel.parse(sheet)
                for comp in ad_bs_sheet.component.unique():
                    if drop_all_zeros:
                        if sum(abs(ad_bs_sheet[ad_bs_sheet.component==comp].loading)) == 0:
                            ad_bs_sheet.drop(ad_bs_sheet[ad_bs_sheet.component==comp].index, inplace=True)
                ad_bs_sheet['celltype'] = sheet
                parquet_filename = f'{sheet}.parquet'
                data_bytes = ad_bs_sheet.to_parquet()
                zip_file.writestr(parquet_filename, data_bytes)
        zip_file.close()
    except:
        print(f'file couldnot be opened')


def _convert_excel_sheets_to_parquet(excel_file_path, zip_file_path, idx, num_perm, drop_all_zeros = False):
    if((100*idx/num_perm)%2==0):
            print(f'{100*idx/num_perm:.0f}% complete')
    try:
        convert_excel_sheets_to_parquet(excel_file_path+f'_{idx}', zip_file_path+f'_{idx}', drop_all_zeros = drop_all_zeros)
    except:
        print(f'{idx} file couldnot be opened')


def convert_excel_sheets_to_parquet_permute(excel_file_path, zip_file_path, drop_all_zeros = False, num_perm=100, start_idx=0, n_jobs=-1):
    Parallel(n_jobs=n_jobs, verbose=5)(delayed(_convert_excel_sheets_to_parquet)(excel_file_path, zip_file_path, idx, num_perm=num_perm, drop_all_zeros = drop_all_zeros) \
               for idx in range(start_idx, num_perm))


def piecewise_weightedtau(x, y, metric = weightedtau, **kwargs):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("All inputs to `piecewise weighted tau` must be of the same "
                         f"size, found x-size {x.size} and y-size {y.size}")
    elif not x.size or not y.size:
        # Return NaN if arrays are empty
        res = SignificanceResult(np.nan, np.nan)
        res.correlation = np.nan
        return res
    
    tau_pos = metric(x[x>=0], y[x>=0], **kwargs).statistic 
    tau_pos = tau_pos if not math.isnan(tau_pos) else 0.0
    tau_neg = metric(x[x<0], y[x<0], **kwargs).statistic 
    tau_neg = tau_neg if not math.isnan(tau_neg) else 0.0
    num_pos = len(x[x>=0])
    num_neg = len(x[x<0])
    del(x)
    del(y)
    gc.collect()
    tau = (tau_pos * num_pos + tau_neg * num_neg)/(num_pos + num_neg)
    # Limit range to fix computational errors
    tau = min(1., max(-1., tau))
    res = SignificanceResult(tau, float('nan'))
    return res



# Number of optimal components for PLS determined by CV

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


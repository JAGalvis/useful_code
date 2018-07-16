####
from urllib.request import urlretrieve
def load_file_from_web(file_url, file_name):
	'''
	Downloads a file from a URL and stores it on the working directory. Usefull when working on cloud environments.
	'''
	url = file_url
	filename = file_name
	urlretrieve(url, filename)

####
import zipfile
import pandas as pd
def load_csv_from_zip(zipped_file_name,csv_name):
	'''
	Extracts CSVs from Zipped files and returns the Pandas DataFrame.
	required_libraries = ['zipfile', 'pandas']
	'''
	archive = zipfile.ZipFile(zipped_file_name, 'r')
	df =  pd.read_csv(archive.open(csv_name))
	return df

####
def check_cloud_or_premise():
    '''
    Check if the notebook is being run on Datalab or on a local machine.
    Returns True if Cloud, False if local
    '''
    import os
    is_cloud = 'DATALAB_ENV' in list(os.environ)
    return is_cloud

####
def how_many_cores():
	import multiprocessing
	cores = multiprocessing.cpu_count()
	return cores

####
from os import walk
def files_in_path(path_to_check, extension = ''):
    '''
    Returns the files in the path (excludes files in lower level directories and those directory names).
    If want to include only files with a given extension use, for instance, extension = 'csv'
    '''
    _ = walk('processed_data/')
    files = list(_)[0][-1]
    if extension == '':
        return files
    else:
        sel_files = []
        for fl in files:
            if '.'+extension in fl:
                sel_files.append(fl)
        return sel_files

####
def factorize_binary_categories(df):
    '''
    This function will factorize every binary "object" within a dataframe. (Null values will be returned as -1)
    '''
    cols =[]
    for col in df.columns:
        data_type = df[col].dtype
        col_unique_len = len(df[col].value_counts())
        if col_unique_len <3: #data_type not in []
            #print(col, col_unique_len, data_type)
            if (data_type == 'object'):
                cols.append(col)
        for col in cols:
            df[col] = pd.factorize(df[col])[0]
    # print('Factorized Columns:',cols)
    return df

####
def clean_str(str_to_clean):
    '''
    Clean some un-wanted characters from a string
    '''
    str_to_clean = str_to_clean.replace('-','_').replace(' ','_').replace('(','').replace(')','').replace(':','').replace(',','').replace('/','')
    return str_to_clean

####
def duplicate_columns(frame):
    '''
    Finds duplicate columns in a dataset and returns their names.
    WARNING. Very slow on wide datasets
    '''
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []
    for t, v in groups.items():
        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)
        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if array_equivalent(ia, ja):
                    dups.append(cs[i])
                    break
    return dups

####
def remove_zero_variance(df, min_var = 0):
    '''
    Function that receives a DF, drops all columns with 0 variance or the minimum variance threshold defined by the user 
    and returns the cleaned DataFrame
    '''
    zero_var = df.std()
    zero_var_cols = zero_var[zero_var.values <= min_var].index.values
    df = df.drop(columns = zero_var_cols)
    return df

####

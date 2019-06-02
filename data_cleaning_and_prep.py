def load_df_json_cols(csv_path, json_cols=[], data_type = {}, nrows=None):
    '''
        Function to load a DF with data in json format.
        Original idea in https://www.kaggle.com/scirpus/to-overcome-the-terrible-data 
    '''
    from pandas.io.json import json_normalize
    import json
    import os
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in json_cols}, 
                     dtype=data_type, # Important!!
                     nrows=nrows)
    
    for column in json_cols:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

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
def remove_zero_variance(df, min_var = 0, print_drop = False, fill_na=True, return_df = True):
    '''
        Function that receives a DF, finds all columns with 0 variance or the minimum variance threshold defined by the user 
        and returns either a copy of the cleaned DataFrame or the columns to drop
    '''
    df = df.copy()
    if fill_na:
        df.fillna(value=-1234567, inplace=True)
    zero_var = df.std()
    zero_var_cols = zero_var[zero_var.values <= min_var].index.values
    if print_drop:
        print('Dropping ',len(zero_var_cols), 'columns:')
        print(zero_var_cols)
    if return_df:
        return df.drop(columns = zero_var_cols)
    else:
        return zero_var_cols
####
# Function to calculate missing values by column
def missing_values_table(df):
    '''
        Function that calculates the number of missing values and the percentage of the total values that are missing for each column.
    '''
    # Total missing values
    missing_values = df.isnull().sum()
    
    # Percentage of missing values
    missing_values_percent = 100 * df.isnull().sum() / len(df)

    # Make a dataframe with the results
    mis_val_table = pd.DataFrame({'missing_values':missing_values, 
                                  'percent_of_total' :missing_values_percent}).sort_values(by = 'missing_values', 
                                                                                           ascending = False)
    mis_val_table = mis_val_table.loc[mis_val_table.missing_values > 0]

    # Print results
    print ("The selected dataframe has " + str(df.shape[1]) + " columns.\n" +
           "The following " + str(mis_val_table.shape[0]) + " columns have missing values.")

    return mis_val_table

def na_impute_means(df, target_col):
    '''
        Receives the processed dataframe and imputes missing numeric values using mean
    '''
#     from sklearn.preprocessing import Imputer
    from sklearn.impute import SimpleImputer

    
    impute_numeric = SimpleImputer(missing_values=np.NaN, strategy='mean', verbose=0, copy=True)
    
    y = df[[target_col]].copy()
    df = df.drop(columns = target_col).copy()
    
    impute_numeric.fit_transform(df)
    df = pd.DataFrame(impute_numeric.transform(df), columns = df.columns)
    df = pd.concat([df, y], axis = 1)

    return df

def data_cardinality(df, sort = True, only_object = False):
    '''
        Function that receives a DF and returns a new DF with the cardinality of each column.
    '''
    if only_object:
        df = df.select_dtypes(include=['O'])
        
    cardinality_df = pd.DataFrame(df.apply(pd.Series.nunique)).reset_index()    
    cardinality_df.columns = ['feature','cardinality']
    if sort:
        cardinality_df = cardinality_df.sort_values(by = 'cardinality').reset_index(drop=True)
    
    return cardinality_df

def remove_extreme_outliers(df, columns, drop_outliers = True):
    '''
        Function to remove or change to NAN the extreme outliers on a given column in the dataframe.
        "Extreme outliers are any data values which lie more than 3.0 times the interquartile range below the first quartile 
        or above the third quartile"
        https://people.richland.edu/james/lecture/m170/ch03-pos.html
    '''
    df = df.copy()
    if type(columns) == str:
        columns = [columns]
    for column in columns:
        # Calculate first and third quartile
        first_quartile = df[column].describe()['25%']
        third_quartile = df[column].describe()['75%']

        # Interquartile range
        iqr = third_quartile - first_quartile

        # Remove outliers
        if drop_outliers == True:
            df = df[(df[column] > (first_quartile - 3 * iqr)) &
                    (df[column] < (third_quartile + 3 * iqr))]
        else:
            df.loc[(df[column] < (first_quartile - 3 * iqr)) |
                   (df[column] > (third_quartile + 3 * iqr)), column] = np.nan
    return df

def remove_collinear_features(x, threshold, target_col):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.
    Inputs: 
        threshold: any features with correlations greater than this value are removed
    Output: 
        dataframe that contains only the non-highly-collinear features
        
    Adapted from:
    * https://github.com/WillKoehrsen/machine-learning-project-walkthrough/blob/master/Machine%20Learning%20Project%20Part%201.ipynb
    * https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on/43104383#43104383
    '''
    
    # Dont want to remove correlations between Energy Star Score
    y = x[target_col]
    x = x.drop(columns = [target_col])
    
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns = drops)
    
    # Add the score back in to the data
    x[target_col] = y
               
    return x

####
# from tqdm import tqdm_notebook
# import pickle
def duplicate_columns(df, show_progress = False, store_duplicates = False):
    '''
    Finds duplicate columns in a dataset and returns their names.
    REQUIRED:
        * import pandas as pd
        * from tqdm import tqdm_notebook
        * import cPickle as pickle
    WARNING: 
        * Can be slow on wide datasets.
        * As it's so slow, it pays to store the duplicate columns.
        * Show progress doesn't work on Google's Colab.
    '''
    dup_cols = {}
    train_enc =  pd.DataFrame(index = df.index)
    
    for col in df.columns:
        train_enc[col] = df[col].factorize()[0]
        
    if show_progress:
        columns = tqdm_notebook(df.columns)
    else:
        columns = df.columns
        
    for i, c1 in enumerate(columns):
        for c2 in train_enc.columns[i + 1:]:
            if c2 not in dup_cols and np.all(train_enc[c1] == train_enc[c2]):
                dup_cols[c2] = c1
    
    if store_duplicates:
        try:
            pickle.dump(dup_cols, open('dup_cols.p', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
        except:
            print("Columns couldn't be pickled.")
    return dup_cols


def mean_encoding_kfold(df, folds, col_to_encode, target_col):
    '''
        Very prone to target leakage. Do proper validation.
    '''
    df = df.copy()
    from sklearn.model_selection import KFold
    folds = KFold(n_splits=folds, shuffle=False, random_state=1)
    
    for tr_ind, val_ind in folds.split(df):
        X_tr, X_val = df.iloc[tr_ind], df.iloc[val_ind]
        df.loc[df.index[val_ind], col_to_encode+'_mean_enc'] = X_val[col_to_encode].map(X_tr.groupby(col_to_encode)[target_col].mean())
    
    df[col_to_encode+'_mean_enc'].fillna(df[target_col].mean(), inplace=True)
    
    return df

def mean_encoding_loo(df, col_to_encode, target_col):
    '''
        
    '''
    df = df.copy()
    leave_one_out_sum = df[col_to_encode].map(df.groupby(col_to_encode)[target_col].sum())
    leave_one_out_count = df[col_to_encode].map(df.groupby(col_to_encode)[target_col].count())
    
    df[col_to_encode+'_mean_enc_loo'] = ((leave_one_out_sum - df[target_col]))/(leave_one_out_count-1)
    df[col_to_encode+'_mean_enc_loo'].fillna(df[target_col].mean(), inplace=True)

    return df

def mean_encoding_smooth(df, col_to_encode, target_col, alpha):
    '''
        
    '''
    df = df.copy()
    globalmean = df[target_col].mean()
    nrows = df.groupby(col_to_encode).size()
    means = df.groupby(col_to_encode)[target_col].agg('mean')
    
    score = (np.multiply(means,nrows)  + globalmean*alpha) / (nrows+alpha)
    
    df[col_to_encode+'_smooth'] = df[col_to_encode].map(score)
    df[col_to_encode+'_smooth'].fillna(globalmean, inplace=True)

    return df

def mean_encoding_expanding(df, col_to_encode, target_col):
    '''
        
    '''
    df = df.copy()
    cumsum = df.groupby(col_to_encode)[target_col].cumsum() - df[target_col]
    cumcnt = df.groupby(col_to_encode).cumcount()

    df[col_to_encode+'_mean_expanded'] = cumsum/cumcnt
    df[col_to_encode+'_mean_expanded'].fillna(df[target_col].mean(), inplace=True)

    return df

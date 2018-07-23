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
# Function to calculate missing values by column
def missing_values_table(df):
    '''
    Function that calculates the number of missing values and the percentage of the total values that are missing for each column.
    Credits: https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-a-column-in-pandas-dataframe/39734251#39734251
    '''
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

def remove_extreme_outliers(df, column, drop_outliers = True):
    '''
    Function to remove or change to NAN the extreme outliers on a given column in the dataframe.
    "Extreme outliers are any data values which lie more than 3.0 times the interquartile range below the first quartile or above the third quartile"
    https://people.richland.edu/james/lecture/m170/ch03-pos.html
    '''
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
    x = x.drop(columns = ['Weather Normalized Site EUI (kBtu/ft²)', 
                          'Water Use (All Water Sources) (kgal)',
                          'log_Water Use (All Water Sources) (kgal)',
                          'Largest Property Use Type - Gross Floor Area (ft²)'])
    
    # Add the score back in to the data
    x[target_col] = y
               
    return x



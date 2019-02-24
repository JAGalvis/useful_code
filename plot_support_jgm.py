def plot_categorical_bars(df, column, hue=None, colors = None, normalized = True, figsize=(10,4), display_val = True):
    '''
        Creates a column plot of categorical data using seaborn
    '''
    norm = len(df) if normalized else 1
    y = 'percent' if normalized else 'count_values'
    columns = [col for col in [column, hue] if col != None]
    df = df[columns].copy()
    df['count_values'] = 1
    df = df.groupby(by = columns).count().reset_index()
    df['percent'] = df['count_values']/norm * 100
    fig, ax = plt.subplots(figsize=figsize); sns.set(style="darkgrid")
    ax = sns.barplot(x=column, y=y, data=df, hue=hue, palette=colors, ax=ax)
    if display_val:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}'.format(height),
                    ha="center") 
    plt.show()
    
def plot_faceted_categorical_bar(df, bars, group_cols, columns, rows, hue = None, colors = None): 
    '''
        Plots faceted bar chart
    '''
    df = df.copy()
    df = df[group_cols]
    df['count_values'] = 1
    df = df.groupby(by= group_cols, as_index = False).count()
    
    g = sns.FacetGrid(data=df, col=columns, row=rows, hue=hue, palette=colors)
    g = g.map(plt.bar, bars, 'count_values')
    g.set_xticklabels(rotation=80)
    plt.show()
    

def plot_unique_values_integer_dist(df):
    '''
        Plots the distribution of unique values in the integer columns.
    '''
    
    df.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'blue', 
                                                                              figsize = (8, 6),
                                                                              edgecolor = 'k', linewidth = 2);
    plt.xlabel('Number of Unique Values'); plt.ylabel('Count');
    plt.title('Count of Unique Values in Integer Columns');
    plt.show()
    
def plot_float_values_dist(df, target_col, colors_dict, category_dict):
    '''
        Plots distribution of all float columns
        https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough
    '''
    from collections import OrderedDict
    plt.figure(figsize = (20, 16))
    plt.style.use('fivethirtyeight')

    # Color mapping
    colors = OrderedDict(colors_dict) #{1: 'red', 2: 'orange', 3: 'blue', 4: 'green'}
    category_mapping = OrderedDict(category_dict) # {1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'}

    # Iterate through the float columns
    for i, col in enumerate(df.select_dtypes('float')):
        ax = plt.subplot(4, 2, i + 1)
        # Iterate through the poverty levels
        for category, color in colors.items():
            # Plot each poverty level as a separate line
            sns.kdeplot(df.loc[df[target_col] == category, col].dropna(), 
                        ax = ax, color = color, label = category_mapping[category])

        plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')

    plt.subplots_adjust(top = 2)
    
    plt.show()


def plot_bubble_categoricals(x, y, data, annotate = True):
    """
        Plot counts of two categoricals.
        Size is raw count for each grouping.
        Percentages are for a given value of y.
        To read the plot, choose a given y-value and then read across the row.  
        https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough
    """
    
    # Raw counts 
    raw_counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = False))
    raw_counts = raw_counts.rename(columns = {x: 'raw_count'})
    
    # Calculate counts for each group of x and y
    counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = True))
    
    # Rename the column and reset the index
    counts = counts.rename(columns = {x: 'normalized_count'}).reset_index()
    counts['percent'] = 100 * counts['normalized_count']
    
    # Add the raw count
    counts['raw_count'] = list(raw_counts['raw_count'])
    
    plt.figure(figsize = (14, 10))
    # Scatter plot sized by percent
    plt.scatter(counts[x], counts[y], edgecolor = 'k', color = 'lightgreen',
                s = 100 * np.sqrt(counts['raw_count']), marker = 'o',
                alpha = 0.6, linewidth = 1.5)
    
    if annotate:
        # Annotate the plot with text
        for i, row in counts.iterrows():
            # Put text with appropriate offsets
            plt.annotate(xy = (row[x] - (1 / counts[x].nunique()), 
                               row[y] - (0.15 / counts[y].nunique())),
                         color = 'navy',
                         s = f"{round(row['percent'], 1)}%")
        
    # Set tick marks
    plt.yticks(counts[y].unique())
    plt.xticks(counts[x].unique())
    
    # Transform min and max to evenly space in square root domain
    sqr_min = int(np.sqrt(raw_counts['raw_count'].min()))
    sqr_max = int(np.sqrt(raw_counts['raw_count'].max()))
    
    # 5 sizes for legend
    msizes = list(range(sqr_min, sqr_max,
                        int(( sqr_max - sqr_min) / 5)))
    markers = []
    
    # Markers for legend
    for size in msizes:
        markers.append(plt.scatter([], [], s = 100 * size, 
                                   label = f'{int(round(np.square(size) / 100) * 100)}', 
                                   color = 'lightgreen',
                                   alpha = 0.6, edgecolor = 'k', linewidth = 1.5))
        
    # Legend and formatting
    plt.legend(handles = markers, title = 'Counts',
               labelspacing = 3, handletextpad = 2,
               fontsize = 16,
               loc = (1.10, 0.19))
    
    plt.annotate(f'* Size represents raw count while % is for a given y value.',
                 xy = (0, 1), xycoords = 'figure points', size = 10)
    
    # Adjust axes limits
    plt.xlim((counts[x].min() - (6 / counts[x].nunique()), 
              counts[x].max() + (6 / counts[x].nunique())))
    plt.ylim((counts[y].min() - (4 / counts[y].nunique()), 
              counts[y].max() + (4 / counts[y].nunique())))
    plt.grid(None)
    plt.xlabel(f"{x}"); plt.ylabel(f"{y}"); plt.title(f"{y} vs {x}");
    
    
def plot_correlation_heatmap(df, variables):
    '''
    '''
    
    # variables = ['Target', 'dependency', 'warning', 'walls+roof+floor', 'meaneduc',
    #              'floor', 'r4m1', 'overcrowding']

    # Calculate the correlations
    corr_mat = df[variables].corr().round(2)

    # Draw a correlation heatmap
    # plt.rcParams['font.size'] = 18
    plt.figure(figsize = (12, 12))
    sns.heatmap(corr_mat, vmin = -0.5, vmax = 0.8, center = 0, 
                cmap = plt.cm.RdYlGn_r, annot = True);
    
    
# def plot_featuresplot(df, features):
    
#     #import warnings
#     #warnings.filterwarnings('ignore')

#     # Copy the data for plotting
#     plot_data = df[features]

#     # Create the pairgrid object
#     grid = sns.PairGrid(data = plot_data, size = 4, diag_sharey=False,
#                         hue = 'Target', hue_order = [4, 3, 2, 1], 
#                         vars = [x for x in list(plot_data.columns) if x != 'Target'])

#     # Upper is a scatter plot
#     grid.map_upper(plt.scatter, alpha = 0.8, s = 20)

#     # Diagonal is a histogram
#     grid.map_diag(sns.kdeplot)

#     # Bottom is density plot
#     grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r);
#     grid = grid.add_legend()
#     plt.suptitle('Feature Plots Colored By Target', size = 32, y = 1.05);
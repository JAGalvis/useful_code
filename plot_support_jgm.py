def plot_categorical_bars(df, column,hue=None, colors = None, normalized = True, figsize=(10,4), display_val = True):
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
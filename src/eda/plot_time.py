import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_time(df, x, y = "consumption", markersize=5, groupby=None, ax=None, title=None, marker=None):
    
    palette=None
    errorbar=('ci', 95)
    if groupby is not None:
        errorbar = None
        palette = "tab20"
    
    sns.lineplot(data=df,
                    x = x,
                    y=y,
                    hue=groupby,
                    ax=ax,
                    marker=marker,
                    markersize=markersize,
                    palette=palette,
                    errorbar=errorbar)
    
    ax.set_title(label=title)
    
    if x == "hour":
        ax.set_xticks(range(0,25))
    elif x == "week":
        ax.set_xticks(range(0,54,6))
    elif x == "month":
        ax.set_xticks(range(1,13,1))
    elif x == "day":
        ax.set_xticks(range(1,32,6))
        
    return ax
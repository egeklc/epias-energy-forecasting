import pandas as pd
import seaborn as sns
import numpy as np
from datetime import date

def plot_trend(df, y, label, ax, title):
    
    temp_df = df.reset_index("date")
    temp_df["date_ordinal"] = temp_df["date"].apply(lambda x: x.toordinal())
    
    
    sns.lineplot(data=temp_df, x="date_ordinal", y=y, label=label, ax=ax)
    
    sns.regplot(
        data=temp_df,
        x="date_ordinal",
        y=y,
        scatter=False,
        lowess=True,
        line_kws={"color":"red"},
        label="Trend",
        ax=ax
    )
    
    
    ax.set_xlim(temp_df['date_ordinal'].min() - 1, temp_df['date_ordinal'].max() + 1)
    
    
    tick_dates = pd.to_datetime([
        "2018", "2019", "2020", "2021","2022",
        "2023", "2024", "2025", "2026"
    ])
    
    tick_positions = tick_dates.map(pd.Timestamp.toordinal)
    ax.set_xticks(tick_positions)
    
    tick_labels = [date.fromordinal(int(item)).strftime("%Y") for item in tick_positions]
    ax.set_xticklabels(tick_labels)
    
    ax.set_xlabel("Date")
    ax.set_ylabel(y.capitalize())
    ax.set_title(title)
    ax.legend(loc="upper left")
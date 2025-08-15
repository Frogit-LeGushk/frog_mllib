from typing import Optional, List

import pandas as pd
import polars as pl
import polars_ds as pl_ds

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def plot_lines_and_bins(
    df: pl.DataFrame, 
    line_colnames: List[str], 
    y_legend_lines: List[str],
    bin_colname: str, # must be temporal type
    
    ax: Optional[mpl.axes.Axes] = None,
    segmentation: str = '1w', # see polars.Expr.dt.truncate 
    
    x_label: str = 'Дата выдачи',
    y_label: str = 'Вероятность дефолта',
    y_legend_bin: str = 'Количество выдач',
    
    title: str = 'Количество договоров и уровень дефолта',
    scale_top_offset_lines: Optional[float] = 1.1,
    scale_top_offset_bins: Optional[float] = 1.5,
    palette_name: str = 'dark',
    show_bins = True,
    marker = 'o'
):
    
    palette = sns.color_palette(palette_name)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    df_agg = df.group_by(
        pl.col(bin_colname).dt.truncate(segmentation).cast(pl.Date)
    ).agg(
        pl.col(line_colnames).mean().mul(100).round(3),
        pl.col(bin_colname).count().alias(f'count_{bin_colname}')
    ).sort(
        bin_colname
    )
    
    if isinstance(df_agg, pl.LazyFrame):
        df_agg = df_agg.collect().to_pandas()
    else:
        df_agg = df_agg.to_pandas()

    for i, (line_colname, y_legend_line) in enumerate(zip(line_colnames, y_legend_lines)):
        sns.lineplot(
            df_agg,
            x = bin_colname,
            y = line_colname,
            color = palette[i],
            marker = marker,
            ax = ax,
            linewidth = 1.8,
            label = y_legend_line
        )

    ax.grid(axis = 'y', linestyle = '--', color = palette[7])
    if scale_top_offset_lines is not None:
        ax.set_ylim(0,  df_agg[line_colnames].max().max() * scale_top_offset_lines)
    ax.set_xlim(df_agg[bin_colname].min(),  df_agg[bin_colname].max())
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if show_bins:
        ax_twin = ax.twinx()
        
        ax_twin.fill_between(
            x = df_agg[bin_colname].values,
            y1 = df_agg[f'count_{bin_colname}'].values,
            color = palette[0],
            edgecolor = palette[7],
            alpha = 0.5,
            label = y_legend_bin
        )
        
        if scale_top_offset_bins is not None:
            ax_twin.set_ylim(0, df_agg[f'count_{bin_colname}'].max() * scale_top_offset_bins)
        ax_twin.legend(loc='lower center', framealpha=0.4)

def plot_lines_and_bins_roc_auc(
    df: pl.DataFrame, 
    score_colnames: List[str], 
    y_legend_lines: List[str],
    bin_colname: str, # must be temporal type
    target_colname: str,
    
    ax: Optional[mpl.axes.Axes] = None,
    segmentation: str = '1w', # see polars.Expr.dt.truncate 
    
    x_label: str = 'Дата выдачи',
    y_label: str = 'Вероятность дефолта',
    y_legend_bin: str = 'Количество выдач',
    
    title: str = 'Количество договоров и уровень дефолта',
    scale_top_offset_lines: Optional[float] = 1.1,
    scale_top_offset_bins: Optional[float] = 1.5,
    palette_name: str = 'dark',
    show_bins = True,
    marker = 'o',
    gini_instead_roc_auc = True
):
    metric_name = 'gini' if gini_instead_roc_auc else 'roc_auc'
    metric_score_colnames = [f'{metric_name}_{score_colname}' for score_colname in score_colnames]
    palette = sns.color_palette(palette_name)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    df_agg = df.group_by(
        pl.col(bin_colname).dt.truncate(segmentation).cast(pl.Date)
    ).agg(
        [
            pl_ds.query_roc_auc(target_colname, score_colname).alias(metric_colname) 
            for score_colname, metric_colname in zip(score_colnames, metric_score_colnames)
        ] + [
            pl.col(bin_colname).count().alias(f'count_{bin_colname}')
        ]
    ).sort(
        bin_colname
    )

    if gini_instead_roc_auc:
        df_agg = df_agg.with_columns(pl.col(metric_score_colnames).mul(2).sub(1))

    if isinstance(df_agg, pl.LazyFrame):
        df_agg = df_agg.collect().to_pandas()
    else:
        df_agg = df_agg.to_pandas()

    for i, (line_colname, y_legend_line) in enumerate(zip(metric_score_colnames, y_legend_lines)):
        sns.lineplot(
            df_agg,
            x = bin_colname,
            y = line_colname,
            color = palette[i],
            marker = marker,
            ax = ax,
            linewidth = 1.8,
            label = y_legend_line
        )

    ax.grid(axis = 'y', linestyle = '--', color = palette[7])
    if scale_top_offset_lines is not None:
        ax.set_ylim(0,  df_agg[metric_score_colnames].max().max() * scale_top_offset_lines)
    ax.set_xlim(df_agg[bin_colname].min(),  df_agg[bin_colname].max())
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if show_bins: 
        ax_twin = ax.twinx()
        
        ax_twin.fill_between(
            x = df_agg[bin_colname].values,
            y1 = df_agg[f'count_{bin_colname}'].values,
            color = palette[0],
            edgecolor = palette[7],
            alpha = 0.5,
            label = y_legend_bin
        )
    
        if scale_top_offset_bins is not None:
            ax_twin.set_ylim(0, df_agg[f'count_{bin_colname}'].max() * scale_top_offset_bins)
        ax_twin.legend(loc='lower center', framealpha=0.4)



def get_df_info(df: pd.DataFrame, thr: float = 0.01, accur: int = 3):
    '''
    Возвращает поколоночную информацию о датафрейме в другом датафрейме.
    Если датафрейм пустой, вернется None.

    Все колонки имеют тип 'string' (для удобства работы), кроме
    'example_1' и 'example_2', которые имеют тип как в датафреме.

    Параметры
    ----------
    df : pandas датафрейм
    thr : float значение в диапазоне [0, 1]
        До какого значения считать долю медианных элементов
        нормальной. Если доля будет больше указанного значения,
        то будут учавствовать в подсчете trash_score
    accur : int в диапазоне >= 0
        Используется в округлении эначений в датафрейме

    Возвращает
    ----------
        датафрейм с информацией о колонках
    '''
    n, m = len(df), len(df.columns) # table (n x m)
    if n == 0:
        return None

    pd.Series

    # datamining for df_info (everything by columns)
    dtypes = pd.Series(df.dtypes, copy=True)
    unique = df.nunique(dropna=False) # include null
    nan = (1 - df.notnull().sum() / n).round(accur)
    zero = ((df == 0).sum() / n).round(accur)
    empty_str = ((df == '').sum() / n).round(accur)


    mode_v = df.mode(dropna=True).iloc[0] # exclude null
    # get count of elements == mode
    mode_c = [(df.iloc[:,i] == mode_v[i]).sum() / n for i in range(m)]
    mode_c = pd.Series(mode_c).round(accur)

    trash_x1 = nan + zero + empty_str
    trash_x2 = (mode_c > thr) * mode_v
    # apply max((nan + zero + empty_str), mode * (if more than normal))
    trash_score = pd.DataFrame(list(zip(trash_x1, trash_x2)), copy=False).max(axis=1)

    # get two different! examples
    examples = df.sample(2, replace=False)
    example_1 = examples.iloc[0]
    example_2 = examples.iloc[1]


    # change types for replase 0.000 to -1
    replace_dict = {'0.0': '-1'} # replace by dictionary, not by str.replace!
    unique = unique.astype('string').replace(replace_dict)
    nan = nan.astype('string').replace(replace_dict)
    zero = zero.astype('string').replace(replace_dict)
    empty_str = empty_str.astype('string').replace(replace_dict)
    mode_c = mode_c.astype('string').replace(replace_dict)
    trash_score = trash_score.astype('string').replace(replace_dict)

    dtypes = dtypes.astype('string').replace({'string[python]': 'string'})


    # df_info props
    cols = ['dtypes', 'unique', 'nan', 'zero', 'empty_str', 'mode_v', 'mode_c', 'trash_score', 'example_1', 'example_2']
    cols = pd.Series(cols)
    data = list(zip(dtypes, unique, nan, zero, empty_str, mode_v, mode_c, trash_score, example_1, example_2))

    # make and ret df_info
    df_info = pd.DataFrame(data, index=df.columns, columns=cols)
    # sorting in lexicographic order for convenient
    df_info.sort_index(inplace=True)
    return df_info.reindex(sorted(df_info.columns), axis=1)


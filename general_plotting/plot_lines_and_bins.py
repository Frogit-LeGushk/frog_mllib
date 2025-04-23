import pandas as pd
import numpy as np
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_lines_and_bins(
    lines:             pd.DataFrame, 
    bins:              pd.DataFrame, 
    ax:                object = None,
    title:             str = '',
    legend_loc:        str = 'upper left',
    bbox_to_anchor:    tuple = None,
    legend_title:      str = '',
    ylabels:           list = ['Count', 'Metric'],
    right_ylims:       list = None,
    annot:             bool = [True, False], # 0 - lines, 1 - bins
    fmt:               int = (3, 0), # 0 - lines, 1 - bins
    annot_bins_offset: float = -100,
    xticks_rotation:   int = 45,
    bins_alpha:        float = .1,
    cmap:              str = 'tab10',
    xticks_step:       int = 1,
):
    
    def _get_n_markers(n):
        from matplotlib.lines import Line2D
        if n <= len(Line2D.filled_markers):
            return Line2D.filled_markers[:n]
        else:
            n_repeat = n // len(Line2D.filled_markers) + 1
            return np.repeat(Line2D.filled_markers, n_repeat)
        
    
    def _get_n_colors(n, cmap='tab10'):
        from matplotlib import colormaps
        cmap = colormaps.get_cmap(cmap)
        cmap_ids = np.linspace(0, 1, n)
        return [cmap(cmap_id) for cmap_id in cmap_ids]
        
    
    if type(lines) is not pd.DataFrame:
        lines = pd.DataFrame(lines)
    
    if type(bins) is not pd.DataFrame:
        bins = pd.DataFrame(bins)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    bins.plot.bar(ax = ax, alpha = bins_alpha, legend=False, color='black')
    
    xticklabels = ax.get_xticklabels()[::xticks_step]
    ax.set_xticks(
        np.arange(
            len(ax.get_xticks()), 
            step=xticks_step)
    )
    ax.set_xticklabels(xticklabels) 
    ax.tick_params(axis='x', rotation=xticks_rotation)
    
    ax.set_ylabel(ylabels[0])
    if annot[1]:
        xticks = ax.get_xticks()
        for name, series in bins.items():
            for x, y in zip(xticks, series.values):
                ann = ax.annotate('{0:.{1}f}'.format(y, fmt[1]),
                             (x, 0),
                             textcoords="offset points",
                             xytext=(0,annot_bins_offset),
                             ha='center', 
                             va='bottom',
                             fontsize=9,
                             rotation=45)

    ax_ = ax.twinx()
    x = np.arange(len(lines))
    markers = _get_n_markers(lines.shape[1])
    colors = _get_n_colors(lines.shape[1], cmap=cmap)
    for line, marker, color in zip(lines, markers, colors):
        ax_.plot(x, lines[line], marker=marker, color=color)
    ax_.tick_params(axis='x', rotation=xticks_rotation)
    ax_.legend(
        lines.columns, 
        loc=legend_loc, 
        bbox_to_anchor=bbox_to_anchor, 
        title=legend_title,
    )
    ax_.set_ylabel(ylabels[1])
    if right_ylims is not None:
        ax_.set_ylim(right_ylims)
    ax_.set_title(title)
    ax_.grid()
    if annot[0]:
        for col in range(len(lines.columns)):
            for idx in range(len(lines)):
                ax_.annotate('{0:.{1}f}'.format(lines.iloc[idx, col], fmt[0]),
                             (idx, lines.iloc[idx, col]),
                             textcoords="offset points",
                             xytext=(0,5),
                             ha='center', fontsize=9)

def plot_beauty_dataframe(df: pd.DataFrame, index=None, cols=None, caption: str = ""):
    """
    Раскрашивает ячейки датафрейма, согласно их значениям.
    
    Parameters
    ----------
    df: pd.DataFrame
    index: pd.DataFrame.index
        подмножество строк, на которых будет производиться раскраска
    cols: List-like[str]
        подмножество колонок, на которых будет производиться раскраска
    caption: str
        подпись к датафрейму (если таковая нужна)
        
    Returns:
    --------
        pandas.io.formats.style.Styler - не совсем датафрейм, дальнейшие арифметические 
    операции производить не получится. Свои стили не наследуются. Применять в самом конце
    , для вывода информации
    """
    
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list('', [(0, 'white'), (1, 'indianred')])
    
    if index is None: index = df.index
    if cols is None: cols = df.columns
    
    subset = pd.IndexSlice[index, cols]
    styler = df.style.background_gradient(cmap=cmap, axis=None, subset=subset)

    #выравнивание содержимого ячеек датафрейма по центру
    styler.set_properties(**{'text-align': 'center'}).format("{:.2f}").set_properties(**{'font-size': '13pt'})

    #установка стилей заголовков и имен индексов
    index_names = {'selector': '.index_name', 'props': [('font-style', 'italic'), ('color', 'darkgrey'), ('font-weight', 'normal')]}
    headers = {'selector': 'th:not(.index_name)', 'props': [('background-color', '#EFEDED'), ('color', 'black')]}
    generation_header = {'selector': 'th.col_heading.level0', 'props': [('text-align', 'center'), ('font-size', '1.5em')]}
    styler.set_table_styles([index_names, headers, generation_header])
    return styler.set_caption(caption)

def plot_lines_and_bins_plotly(
    lines: pd.DataFrame,
    bins: pd.DataFrame,
    title: str = '',
    legend_loc: str = 'upper left',
    legend_title: str = '',
    ylabels: List[str] = ['Count', 'Metric'],
    right_ylims: Optional[List[float]] = None,
    annot: List[bool] = [True, False],  # 0 - lines, 1 - bins
    fmt: Tuple[int, int] = (3, 0),  # 0 - lines, 1 - bins
    xticks_rotation: int = 45,
    bins_alpha: float = 0.3,
    xticks_step: int = 1
) -> go.Figure:
    """
    Create a combined bar and line plot using Plotly with dual y-axes.
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Convert legend location to plotly format
    legend_x = 0 if 'left' in legend_loc else 1
    legend_y = 1 if 'upper' in legend_loc else 0
    legend_xanchor = 'left' if 'left' in legend_loc else 'right'
    legend_yanchor = 'top' if 'upper' in legend_loc else 'bottom'
    
    # Add bars
    x = np.arange(len(bins.index))
    for col in bins.columns:
        fig.add_trace(
            go.Bar(
                x=x,
                y=bins[col],
                name=col,
                opacity=bins_alpha,
                marker_color='black',
                text=bins[col].round(fmt[1]) if annot[1] else None,
                textposition='outside',
                showlegend=False
            ),
            secondary_y=False
        )
    
    # Add lines
    markers = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star']
    colors = px.colors.qualitative.Set1[:lines.shape[1]]  # Using Plotly's built-in color sequences
    
    for idx, col in enumerate(lines.columns):
        marker_symbol = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lines[col],
                name=col,
                mode='lines+markers+text' if annot[0] else 'lines+markers',
                marker=dict(symbol=marker_symbol, size=8, color=color),
                text=lines[col].round(fmt[0]) if annot[0] else None,
                textposition='top center',
                line=dict(color=color)
            ),
            secondary_y=True
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        legend=dict(
            title=legend_title,
            x=legend_x,
            y=legend_y,
            xanchor=legend_xanchor,
            yanchor=legend_yanchor
        ),
        xaxis=dict(
            tickmode='array',
            ticktext=bins.index[::xticks_step],
            tickvals=x[::xticks_step],
            tickangle=xticks_rotation
        ),
        barmode='overlay',
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text=ylabels[0], secondary_y=False)
    fig.update_yaxes(title_text=ylabels[1], secondary_y=True)
    
    # Set y-axis range if specified
    if right_ylims is not None:
        fig.update_yaxes(range=right_ylims, secondary_y=True)
    
    return fig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_validate_calibr_error_by_feature(
    data: pd.DataFrame,
    feature: str,
    q = 10,
    target = '',
    score: str = '',
    error_rate: float = 0.2,
    custom_bin = None,
    verbose = False
):
    """
    Анализирует калибровку скоринговой модели по заданному признаку.
    Вычисляет ошибки калибровки между предсказанными значениями (score) и фактическими (target).

    Параметры:
        data (pd.DataFrame): Исходный DataFrame с данными
        feature (str): Название признака для анализа
        q (int): Количество бинов для разбиения (по умолчанию 10)
        target (str): Название целевой переменной (по умолчанию 'MAX_TARGET_MOB06')
        score (str): Название колонки с предсказаниями модели (по умолчанию 'SCORE')
        error_rate (float): Порог для определения ошибки калибровки (по умолчанию 0.2)
        custom_bin (list): Пользовательские границы бинов (по умолчанию None)
        verbose (bool): Флаг вывода дополнительной информации (по умолчанию False)

    Возвращает:
        tuple: Кортеж с метриками ошибок и статистикой:
            - val_error: Процент ошибок калибровки
            - val_risk_error: Суммарная ошибка риска
            - val_bad_error: Ошибка "недоштрафования"
            - val_good_error: Ошибка "перештрафования"
            - df_stat: DataFrame с подробной статистикой по бинам
    """
    data = data.copy()
    data = data.dropna(subset = [target, score])
    

    if verbose is not False:
        print('size data, k:', round(len(data)/1e3, 2))
        print(f'{len(custom_bin)}_bins:', custom_bin)
        # custom_bin[-1] = custom_bin[-1]#+0.01

    #1.statistic and 10 bins
    if data[feature].nunique() <= 10:
         data[f'{feature}_bin_right'] = data[f'{feature}']
    else:
        if data[feature].dtype not in ('int64', 'float64'):
            raise f'{feature} has dtype {data[feature].dtype} and nunique {data[feature].nunique()}'
        if custom_bin is None:
            custom_bin = list(pd.qcut(data.loc[:, feature], q=q, retbins=True, duplicates='drop')[1].round(3))
        data[f'{feature}_bin'] = np.digitize(data[feature], bins = custom_bin, right=True)
        mapper = {i: custom_bin[i] for i in range(len(custom_bin))}
        # mapper = data.groupby([f'{feature}_bin'])[feature].mean().to_dict()
        data[f'{feature}_bin_right'] = data[f'{feature}_bin'].map(mapper)
    
    aggs = {score: 'mean', target: ['mean', 'count']}

    #2. calc mean score and target
    df_stat = data.loc[:].groupby(f'{feature}_bin_right', dropna=False)[[score] + [target]]\
    .agg(aggs).round(3)

    data['TOTAL'] = 1
    for key in aggs.keys():
        df_stat.loc['TOTAL', key] = data.groupby('TOTAL').agg(aggs)[key].values

    #3. calc calibration (or parity)
    df_stat['TARGET/SCORE'] = df_stat[(target, 'mean')]/df_stat[(score, 'mean')]

    #4. calc delta_val
    df_stat['DELTA'] = df_stat['TARGET/SCORE']/df_stat.loc['TOTAL', 'TARGET/SCORE'].values[0]

    #5. error
    df_stat['ERROR'] = df_stat['DELTA'].sub(df_stat.loc['TOTAL', 'DELTA'].values[0]).abs().gt(error_rate).astype(int)
    df_stat.loc['TOTAL', 'ERROR'] =  df_stat['ERROR'].iloc[:-1].sum()
    
    #6. count error
    df_stat['ER_CNT'] = df_stat['ERROR']*df_stat[(target, 'count')]
    df_stat.loc['TOTAL', 'ER_CNT'] =  df_stat['ER_CNT'].iloc[:-1].sum()

    #7.% of error
    val_error = df_stat['ER_CNT'].sum()/df_stat[(target, 'count')].sum()*100

    #8 (% of risk error)
    df_stat['RISK_ER_CNT'] = (
        df_stat['ER_CNT'] #кол-во бина с ошибкой более error_rate%
        *df_stat[(target, 'mean')] #средний таргет в бине
        *df_stat['DELTA'].sub(df_stat.loc['TOTAL', 'DELTA'].values[0]) #ошибка в риске
    )

    #доп просрочка, чем по скору
    mask = df_stat['RISK_ER_CNT'].gt(0)
    val_bad_error = df_stat.loc[mask, 'RISK_ER_CNT'].sum()\
                    /df_stat.loc['TOTAL', (target, 'count')]

    #доп скор, чем средняя просрочка
    mask = df_stat['RISK_ER_CNT'].lt(0)
    val_good_error = df_stat.loc[mask, 'RISK_ER_CNT'].sum()\
                    /df_stat.loc['TOTAL', (target, 'count')]

    #лишние недоштраф и перештраф по скору чем по таргету
    val_risk_error = val_bad_error + val_good_error
    
    # display(val_error)
    # display(df_stat)
    
    return val_error, val_risk_error, val_bad_error, val_good_error, df_stat#.iloc[:, :-1]



def display_by_feature_target(data: pd.DataFrame, feature: str, q = 10, target = '', scores: str =  None):
    """
    Визуализирует статистику по целевому показателю в разрезе бинов признака.
    
    Параметры:
        data (pd.DataFrame): Исходный DataFrame с данными
        feature (str): Название признака для анализа
        q (int): Количество бинов для разбиения (по умолчанию 10)
        target (str): Название целевой переменной (по умолчанию '')
        scores (str): Название колонки(-ок) с предсказаниями модели (по умолчанию None)
    
    Возвращает:
        None: Функция только выводит информацию, не возвращает значений
    """
    data = data.copy()
    
    #1.statistic and 10 bins
    if data[feature].nunique() <= 10:
         data[f'{feature}_bin_right'] = data[f'{feature}']
    else:
        if data[feature].dtype not in ('int64', 'float64'):
            raise f'{feature} has dtype {data[feature].dtype} and nunique {data[feature].nunique()}'
        target_bin = list(pd.qcut(data.loc[:, feature], q=q, retbins=True, duplicates='drop')[1].round(3))
        print('size data, k:', round(len(data)/1e3, 2))
        print('10bins:', target_bin)
        
        target_bin[-1] = target_bin[-1]#+0.01
        
        data[f'{feature}_bin'] = np.digitize(data[feature], bins = target_bin, right=True)
        data[f'{feature}_bin_right'] = data[f'{feature}_bin'].map({i: target_bin[i] for i in range(len(target_bin))})

    aggs = {score: 'mean' for score in scores}
    aggs.update({target: ['mean', 'count']})
    
    df_stat =  data.loc[mask].groupby(f'{feature}_bin_right', dropna=False)[scores + [target]]\
    .agg(aggs).round(3)

    data['TOTAL'] = 1
    for key in aggs.keys():
        df_stat.loc['TOTAL', key] = data.groupby('TOTAL').agg(aggs)[key].values
        
    display(df_stat.round(3))


def calc_pivot_table(score_name: str, feature_name: str, df_analys: pd.DataFrame, target: str, q=10, custom_bins: list = None):

    """
    Строит сводную таблицу с анализом целевого показателя по двум скоринговым признакам.
    
    Параметры:
        score_name (str): Название скора 
        feature_name (str): Название скорингового признака
        df_analys (pd.DataFrame): Исходный DataFrame с данными
        target (str): Название целевой переменной
        q (int): Количество бинов для разбиения (по умолчанию 10)
        custom_bins (list): Пользовательские границы бинов (по умолчанию None)
    
    Возвращает:
        None: Функция только выводит сводную таблицу
    """
    
    #1.statistic and 10 bins
    if df_analys[score_name].nunique() <= 10:
         df_analys[f'{score_name}_bin_right'] = df_analys[f'{score_name}']
    else:
        if df_analys[score_name].dtype not in ('int64', 'float64'):
            raise f'{score_name} has dtype {df_analys[score_name].dtype} and nunique {df_analys[score_name].nunique()}'
        target_bin = list(pd.qcut(df_analys.loc[:, score_name], q=q, retbins=True, duplicates='drop')[1].round(3))
        print('size data, k:', round(len(df_analys)/1e3, 2))
        print('10bins:', target_bin)
        target_bin[0] = target_bin[0]# - 0.001
        target_bin[-1] = target_bin[-1]+0.0001
        df_analys[f'{score_name}_bin'] = np.digitize(df_analys[score_name], bins = target_bin, right=True)
        df_analys[f'{score_name}_bin_right'] = df_analys[f'{score_name}_bin'].map({i:target_bin[i] for i in range(len(target_bin))}).round(4)
        mask = df_analys[f'{score_name}_bin'].gt(10)
        print(f'{score_name} greater 10', mask.sum())
        data = df_analys[~mask]

    
    #1.2 (score_name_2)
    #1.statistic and 10 bins
    if df_analys[feature_name].nunique() <= 10:
         df_analys[f'{feature_name}_bin_right'] = df_analys[f'{feature_name}']
    else:
        if data[score_name].dtype not in ('int64', 'float64'):
            raise f'{feature_name} has dtype {df_analys[feature_name].dtype} and nunique {df_analys[feature_name].nunique()}'
        if custom_bins is None:
            target_bin = list(pd.qcut(df_analys.loc[:, feature_name], q=q, retbins=True,  duplicates='drop')[1].round(3))
            print('size data, k:', round(len(df_analys)/1e3, 2))
            print('10bins:', target_bin)
        else:
            target_bin = custom_bins

        target_bin[0] = target_bin[0]# - 0.01
        target_bin[-1] = target_bin[-1]+0.01
        df_analys[f'{feature_name}_bin'] = np.digitize(df_analys[feature_name], bins = target_bin, right=True)
        df_analys[f'{feature_name}_bin_right'] = df_analys[f'{feature_name}_bin'].map({i:target_bin[i] for i in range(len(target_bin))}).round(4)
    
        mask = df_analys[f'{feature_name}_bin'].gt(10)
        print(f'{feature_name} greater 10', mask.sum())
        df_analys = df_analys[~mask]

    
    #only delta_incomes
    pivot_day_name = f'{score_name}_bin_right' # f'{score_name}_bin_right'
    groupby_cols =  f'{feature_name}_bin_right' #f'{score_name_2}_bin_right'
    
    stat = df_analys.groupby([pivot_day_name, groupby_cols], as_index=False, dropna=False)[[f'{target}']].agg(['mean', 'sum', 'count'])
    stat.columns = ['_'.join(i).rstrip('_') for i in stat.columns.to_flat_index()]
    
    stat_pivot = stat.pivot(index = groupby_cols, 
                   columns = pivot_day_name, 
                   values = [f'{target}_mean', f'{target}_sum', f'{target}_count']).round(4)
    
    # display(stat_pivot)


    #only delta_days
    stat_1 = df_analys.groupby(pivot_day_name, dropna=False)[[f'{target}']].agg(['mean', 'sum', 'count'])
    stat_1.columns = ['_'.join(i).rstrip('_') for i in stat_1.columns.to_flat_index()]
    # display(stat_1)

    #only delta_incomes
    stat_2 = df_analys.groupby(groupby_cols, dropna=False)[[f'{target}']].agg(['mean', 'sum', 'count'])
    stat_2.columns = ['_'.join(i).rstrip('_') for i in stat_2.columns.to_flat_index()]
    # display(stat_2)

    stat_all = df_analys[[f'{target}']].agg(['mean', 'sum', 'count'])

    if stat_2.index.isna().sum():
        stat_2 = stat_2.loc[[stat_2.index[-1]] + stat_2.index[:-1].to_list()]

    if stat_1.index.isna().sum():
        stat_1 = stat_1.loc[[stat_1.index[-1]] + stat_1.index[:-1].to_list()]

    #join all together
    stat_pivot[[f'{target}_mean_TOTAL', f'{target}_sum_TOTAL', f'{target}_count_TOTAL']] =\
    stat_2[[f'{target}_mean', f'{target}_sum', f'{target}_count']]

    # display(stat_pivot)

    # print(stat_1[f'{target}_mean'])
    stat_pivot.loc[f'TOTAL', f'{target}_mean']  = stat_1[f'{target}_mean'].values
    stat_pivot.loc[f'TOTAL', f'{target}_sum']   = stat_1[f'{target}_sum'].values
    stat_pivot.loc[f'TOTAL', f'{target}_count'] = stat_1[f'{target}_count'].values


    stat_pivot.loc[f'TOTAL', 
                   [f'{target}_mean_TOTAL', f'{target}_sum_TOTAL', f'{target}_count_TOTAL']] =\
    stat_all[f'{target}'].values

    
    return stat_pivot[[f'{target}_mean', f'{target}_mean_TOTAL',
                    f'{target}_sum', f'{target}_sum_TOTAL',
                    f'{target}_count', f'{target}_count_TOTAL']].rename(
            columns= {f'{target}_count': 'ALL_count', f'{target}_count_TOTAL': 'All_count_TOTAL'}).round(4)


def calc_pivot_table_date(pivot_day_name: str, feature_name: str, df_analys: pd.DataFrame, target: str, q=10, type_display=None):
    """
    Строит сводную таблицу с анализом целевого показателя по дате и скоринговому признаку.
    Поддерживает как горизонтальное, так и вертикальное отображение.
    
    Параметры:
        pivot_day_name (str): Название колонки с датой для анализа
        feature_name (str): Название скорингового признака
        df_analys (pd.DataFrame): Исходный DataFrame с данными
        target (str): Название целевой переменной
        q (int): Количество бинов для разбиения (по умолчанию 10)
        type_display (str): Тип отображения ('vertical' для вертикального)
    
    Возвращает:
        None: Функция только выводит сводную таблицу
    """
    df_analys = df_analys.copy()
    #1.2 (score_name_2)
    if df_analys[feature_name].nunique() <= 10:
         df_analys[f'{feature_name}_bin_right'] = df_analys[f'{feature_name}']
    else:
        if df_analys[score_name].dtype not in ('int64', 'float64'):
            raise f'{feature_name} has dtype {df_analys[feature_name].dtype} and nunique {df_analys[feature_name].nunique()}'
        target_bin = list(pd.qcut(df_analys.loc[:, feature_name], q=q, retbins=True, duplicates='drop')[1].round(3))
        print('size data, k:', round(len(df_analys)/1e3, 2))
        print('10bins:', target_bin)
    
        target_bin[0] = target_bin[0]# - 0.01
        target_bin[-1] = target_bin[-1]+0.01
        df_analys[f'{feature_name}_bin'] = np.digitize(df_analys[feature_name], bins = target_bin, right=True)
        df_analys[f'{feature_name}_bin_right'] = df_analys[f'{feature_name}_bin'].map({i: target_bin[i] for i in range(len(target_bin))}).round(4)
    
        mask = df_analys[f'{feature_name}_bin'].gt(10)
        print(f'{feature_name} greater 10', mask.sum())
        df_analys = df_analys[~mask]
    
    
    pivot_day_name = pivot_day_name
    groupby_cols =  f'{feature_name}_bin_right' #f'{score_name_2}_bin_right'

    if type_display == 'vertical':
        pivot_day_name, groupby_cols = groupby_cols, pivot_day_name
    
    stat = df_analys.groupby([pivot_day_name, groupby_cols], as_index=False, dropna=False)[[f'{target}']].agg(['mean', 'sum', 'count'])
    stat.columns = ['_'.join(i).rstrip('_') for i in stat.columns.to_flat_index()]
    
    stat_pivot = stat.pivot(index = groupby_cols, 
                   columns = pivot_day_name, 
                   values = [f'{target}_mean', f'{target}_sum', f'{target}_count']).round(4)
    
    # display(stat_pivot)


    #only delta_days
    stat_1 = df_analys.groupby(pivot_day_name, dropna=False)[[f'{target}']].agg(['mean', 'sum', 'count'])
    stat_1.columns = ['_'.join(i).rstrip('_') for i in stat_1.columns.to_flat_index()]
    # display(stat_1)

    #only delta_incomes
    stat_2 = df_analys.groupby(groupby_cols, dropna=False)[[f'{target}']].agg(['mean', 'sum', 'count'])
    stat_2.columns = ['_'.join(i).rstrip('_') for i in stat_2.columns.to_flat_index()]
    # display(stat_2)

    if stat_2.index.isna().sum():
        stat_2 = stat_2.loc[[stat_2.index[-1]] + stat_2.index[:-1].to_list()]

    if stat_1.index.isna().sum():
        stat_1 = stat_1.loc[[stat_1.index[-1]] + stat_1.index[:-1].to_list()]
        
    stat_all = df_analys[[f'{target}']].agg(['mean', 'sum', 'count'])

    #join all together
    stat_pivot[[f'{target}_mean_TOTAL', f'{target}_sum_TOTAL', f'{target}_count_TOTAL']] =\
    stat_2[[f'{target}_mean', f'{target}_sum', f'{target}_count']]

    # display(stat_pivot)
    # display( stat_2)

    stat_pivot.loc[f'TOTAL', f'{target}_mean']  = stat_1[f'{target}_mean'].values
    stat_pivot.loc[f'TOTAL', f'{target}_sum']   = stat_1[f'{target}_sum'].values
    stat_pivot.loc[f'TOTAL', f'{target}_count'] = stat_1[f'{target}_count'].values


    stat_pivot.loc[f'TOTAL', 
                   [f'{target}_mean_TOTAL', f'{target}_sum_TOTAL', f'{target}_count_TOTAL']] =\
    stat_all[f'{target}'].values

    return stat_pivot[[f'{target}_mean', f'{target}_mean_TOTAL',
                    f'{target}_sum', f'{target}_sum_TOTAL',
                    f'{target}_count', f'{target}_count_TOTAL']].rename(
            columns= {f'{target}_count': 'ALL_count', f'{target}_count_TOTAL': 'All_count_TOTAL'}).round(4)


def plot_pivot_table_res(res, target, title = None, figsize = (10, 4)):
    #можно передавать
    score_name = res.columns.names[1].split('_')[0]

    #считаем данные для heatmap
    res_mean = res[[f'{target}_mean', f'{target}_mean_TOTAL']]*100
    mask_heatmap = res[['ALL_count', 'All_count_TOTAL']].lt(25).values
    
    total_mean = res_mean.loc[f'TOTAL', f'{target}_mean_TOTAL'].values[0]
    res_mean.columns = [f'{col[1]}' if col[1] != '' else 'All' for col in res_mean.columns]

    #рисуем heat_map
    plt.figure(figsize=figsize)
    sns.heatmap(res_mean, annot=True, fmt=".1f", cmap="coolwarm", center=total_mean, vmax=3*total_mean, vmin = 0, mask=mask_heatmap,
                linewidths=.5, cbar_kws={'label': 'Просрочка'})
    
    #аннотируем там где меньше 25 контркатов
    for (i, j), label in np.ndenumerate(res_mean):
        if mask_heatmap[i, j]:
            label = "{:,}".format(round(label, 1)).replace(","," ")
            plt.text(j + 0.5, i + 0.5, label, fontdict=dict(ha='center',  va='center', color='black'))#, fontsize=13
    
    if None is None:
        title = f"{target}_mean ({score_name})"
    plt.title(title)
    plt.xticks(rotation = 30)
    plt.tight_layout()
    plt.show()
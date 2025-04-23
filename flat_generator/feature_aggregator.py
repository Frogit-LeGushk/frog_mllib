import polars as pl
from typing import Union, Dict, List, Tuple
from joblib import Parallel, delayed
import itertools


class FeatureAggregator:
    """Класс для агрегации признаков с использованием Polars и параллельных вычислений.

    Позволяет создавать сложные агрегационные выражения с фильтрами, 
    применять маскирование признаков и выполнять групповую агрегацию данных.

    Attributes:
        id_column (str): Название колонки с идентификатором для группировки.
        config (Dict): Конфигурация фильтров и агрегаций.
        n_jobs (int): Количество параллельных задач (по умолчанию -1, все ядра).
        agg_exprs (List[pl.Expr]): Список агрегационных выражений Polars.
        feature_names (list): Названия признаков из конфига
    """
    _is_init_exprs = False # const value for main process
    
    @staticmethod
    def _register_custom_functions():
        """Регистрирует пользовательские функции для пространства имён `bki_gen` в Polars.
        
        Добавляет метод `is_not_in` для проверки отсутствия значения в массиве.
        Добавляет метод `last_not_null` для взятия последнего ненулевого значения.
        """
        if not FeatureAggregator._is_init_exprs:
            @pl.api.register_expr_namespace("bki_gen")
            class BkiGenNamespace:
                def __init__(self, expr: pl.Expr) -> None:
                    self._expr = expr
                def is_not_in(self, arr) -> pl.Expr:
                    return ~self._expr.is_in(arr)
                def last_not_null(self) -> pl.Expr:
                    return self._expr.drop_nulls().last()
            FeatureAggregator._is_init_exprs = True

    @staticmethod
    def _get_agg_expr(agg: Dict, filters: List[Dict]) -> pl.Expr:
        """Генерирует выражение для агрегации с учётом фильтров.

        Args:
            agg (Dict): Конфигурация агрегации (колонка, функция, аргументы).
            filters (List[Dict]): Список фильтров для условия.

        Returns:
            pl.Expr: Выражение Polars для агрегации.
        """
        FeatureAggregator._register_custom_functions()
        if_getattr = lambda obj, attr, cond: getattr(obj, attr) if cond else obj
        get_expr = lambda fil: getattr(if_getattr(pl.col(fil['column']), fil['namespace'], fil['namespace'] != ""), fil['function'])(*fil['args'], **fil['kwargs'])
        get_alias = lambda agg: f"{agg['function_alias']}_{agg['column']}_{'_'.join(agg['filters'])}"
        def make_agg_item(agg):
            return (
                pl.all_horizontal([get_expr(filt) for filt in filters]),
                agg['column'], agg['function'], agg['args'], agg['kwargs'], get_alias(agg)
            )
        expr, col, func_name, args, kwargs, alias = make_agg_item(agg)
        func = getattr(if_getattr(pl.when(expr).then(col), agg['namespace'], agg['namespace'] != ""), func_name)
        return func(*args, **kwargs).alias(alias.strip('_')) 

    @staticmethod
    def _get_fil_expr(agg, filters: List[Dict]) -> Tuple[pl.Expr]:
        """Генерирует выражение для подсчёта выполнений фильтров.

        Args:
            agg (Dict): Конфигурация агрегации.
            filters (List[Dict]): Список фильтров.

        Returns:
            Tuple[pl.Expr]: Выражения Polars для подсчёта.
        """
        FeatureAggregator._register_custom_functions()
        if_getattr = lambda obj, attr, cond: getattr(obj, attr) if cond else obj
        get_expr = lambda fil: getattr(if_getattr(pl.col(fil['column']), fil['namespace'], fil['namespace'] != ""), fil['function'])(*fil['args'], **fil['kwargs'])
        get_alias = lambda agg: f"{agg['function_alias']}_{agg['column']}_{'_'.join(agg['filters'])}"
        expr_thr = pl.when(pl.all_horizontal([get_expr(filt) for filt in filters])).then(agg['column']).count().alias(get_alias(agg).strip('_'))
        expr_unq = pl.when(pl.all_horizontal([get_expr(filt) for filt in filters])).then(agg['column']).n_unique().alias(get_alias(agg).strip('_'))
        return expr_thr, expr_unq
        
    @staticmethod
    def _get_agg_exprs(config: Dict, n_jobs: int) -> List[pl.Expr]:
        """Генерирует список агрегационных выражений параллельно.

        Args:
            config (Dict): Конфигурация с фильтрами и агрегациями.
            n_jobs (int): Количество задач для параллельной обработки. Если значение отрицательное – будут задействованы все доступные ядра.

        Returns:
            List[pl.Expr]: Список выражений Polars.
        """
        concat_filters = lambda agg: [config['filters'][name] for name in agg['filters']]
        exprs = Parallel(n_jobs=n_jobs)(delayed(FeatureAggregator._get_agg_expr)(agg, concat_filters(agg)) for agg in config['aggregations'])
        return list(exprs)

    @staticmethod
    def _get_fil_exprs(config: Dict, n_jobs: int) -> List[pl.Expr]:
        """Генерирует список выражений для фильтров параллельно.

        Args:
            config (Dict): Конфигурация.
            n_jobs (int): Количество задач для параллельной обработки. Если значение отрицательное – будут задействованы все доступные ядра.

        Returns:
            List[pl.Expr]: Список выражений Polars.
        """
        concat_filters = lambda agg: [config['filters'][name] for name in agg['filters']]
        exprs = Parallel(n_jobs=n_jobs)(delayed(FeatureAggregator._get_fil_expr)(agg, concat_filters(agg)) for agg in config['aggregations'])
        return list(exprs)
    
    @staticmethod
    def make_expr_config(column: str, function: str, namespace: str, *args: List, **kwargs: Dict) -> Dict:
        """Создает конфигурацию для выражения.

        Args:
            column (str): Название колонки.
            function (str): Имя функции агрегации/фильтрации.
            namespace (str): Пространство имён (например, 'bki_gen').
            *args (List): Дополнительные позиционные аргументы для функции агрегации или фильтрации
            **kwargs (Dict): Дополнительные именованные аргументы

        Returns:
            Dict: Конфигурация выражения.
        """
        expr = {
            "column": column,
            "function": function,
            "namespace": namespace,
            "args": list(args),
            'kwargs': kwargs
        }
        if column == "": expr.pop('column', None)
        return expr

    @property
    def feature_names(self):
        feature_names = []
        for agg in self.config['aggregations']:
            als = agg['function_alias']
            col = agg['column']
            flt = '_'.join(agg['filters'])
            feature_name = f"{als}_{col}_{flt}".strip('_')
            feature_names.append(feature_name)
        return feature_names
    
    def __init__(
        self,
        id_column: str,
        config:    Dict,
        n_jobs:    int
    ):
        """Инициализация FeatureAggregator.

        Args:
            id_column (str): Колонка для группировки.
            config (Dict): Конфигурация фильтров и агрегаций.
            n_jobs (int): Количество параллельных задач.
        """
        self.id_column = id_column
        self.config = config
        self.n_jobs = n_jobs
        self.agg_exprs = FeatureAggregator._get_agg_exprs(self.config, n_jobs)

    def mask(self, data: pl.DataFrame, thr: float, unq: int) -> Tuple[bool]:
        """Создает маску для фильтрации признаков по порогу заполненности.

        Args:
            data (pl.DataFrame): Входные данные.
            thr (float): Порог на минимальную долю ненулевых значений.
            unq (int): Порог на минимальное кол-во уникальных значений.

        Returns:
            Tuple[bool]: Кортеж булевых значений, указывающих, какие признаки удовлетворяют условию порогового значения.
        """
        fil_exprs = FeatureAggregator._get_fil_exprs(self.config, self.n_jobs)
        return data.select(
            expr_thr.ge(pl.lit(thr * data.shape[0])) &
            expr_unq.ge(pl.lit(unq))
            for expr_thr, expr_unq in fil_exprs
        ).row(0)
    
    def agg(self, data: pl.DataFrame, mask: Union[None, tuple]) -> pl.DataFrame:
        """Выполняет агрегацию данных с учётом маски.

        Args:
            data (pl.DataFrame): Входные данные.
            mask (Union[None, tuple]): Маска для фильтрации признаков. Если None, используются все.

        Returns:
            pl.DataFrame: Агрегированные данные.
        """
        if mask is None:
            return data.group_by(self.id_column).agg(self.agg_exprs)
        else:
            agg_exprs_filtered = list(itertools.compress(self.agg_exprs, mask))
            return data.group_by(self.id_column).agg(agg_exprs_filtered)

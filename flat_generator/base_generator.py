import polars as pl
import json, copy, itertools
from typing import Union, Dict, List, Generator, Tuple

try:
    from tqdm.auto import tqdm
except Exception as err:
    raise err

try:
    from feature_aggregator import FeatureAggregator
except Exception as err:
    from generators.feature_aggregator import FeatureAggregator


class BaseGenerator:
    """Класс для генерации и предобработки кредитных данных.

    Обрабатывает сырые данные: переименование колонок, преобразование типов,
    создание новых признаков (например, из платежной строки), фильтрация и агрегация.

    Attributes:
        id_column (str): Колонка-идентификатор (по умолчанию "APPLICATION_NUMBER").
        date_column (str): Колонка с датой заявки.
        verbose (bool): Флаг вывода прогресса.
        mask (Tuple[bool]): Маска для фильтрации признаков.
        cast_dtypes (bool): Флаг приведения типов в предобработке (по умолчанию True)
        depth_filters (int): Макс. глубина комбинаций фильтров. Для оптимальности используйте значение 3. (по умолчанию 10)
        n_jobs (int): Количество параллельных задач. Для оптимальной скорости используйте значение 4. (по умолчанию 1)
        config (Dict): Готовый конфиг
        feature_names (list): Названия признаков из конфига
    """
    alias_null = ['None', 'Null', 'NaT', '', 'none', 'null', 'nat', 'NONE', 'NULL', 'NAT']

    @staticmethod
    def get_config(
        depth_filters: int,
        allowed_filters: Dict,
        numeric_columns: List,
        allowed_num_aggregations: Union[List, Tuple],
    ) -> Dict:
        """Генерирует конфигурацию агрегаций и фильтров.

        Args:
            depth_filters (int): Максимальная глубина комбинаций фильтров.
            allowed_filters (Dict): Фильтры
            numeric_columns (List): Числовые признаки
            allowed_num_aggregations (Union[List, Tuple]): Аггрегации
            
        Returns:
            Dict: Конфигурация для FeatureAggregator.
        """
        config = {
            "filters": copy.deepcopy(allowed_filters),
            "aggregations": []
        }
        
        filter_config = {}
        for k, v in config['filters'].items():
            if v['column'] in filter_config:
                filter_config[v['column']][k] = v
            else:
                filter_config[v['column']] = {k: v}
        
        filter_combinations = [list(filter_config['REQUESTID'].keys())]
        
        groups = list(k for k in filter_config.keys() if k != "REQUESTID")
        for r in range(1, min(len(groups), depth_filters) + 1):
            for group_comb in itertools.combinations(groups, r):
                group_filters = [list(filter_config[group].keys()) for group in group_comb]
                for combination in itertools.product(*group_filters):
                    filter_combinations.append(list(combination))

        for filter_combination in filter_combinations:
            for feature in copy.deepcopy(numeric_columns):
                for agg_func in copy.deepcopy(allowed_num_aggregations):
                    config['aggregations'].append(agg_func(feature) | {'filters': filter_combination})

        return config

    @property
    def feature_names(self):
        return self.feature_aggregator.feature_names

    def __init__(
        self,
        id_column:                str,
        date_column:              str,
        depth_filters:            int,
        allowed_filters:          Dict,
        numeric_columns:          List,
        allowed_num_aggregations: Union[List, Tuple],
        verbose:                  bool = False,
        cast_dtypes:              bool = True,
        n_jobs:                   int  = 1,
        config:                   Dict = None,
    ):
        """Инициализация BaseGenerator."""
        self.id_column = id_column
        self.date_column = date_column
        self.verbose = verbose
        self.mask = None
        self.cast_dtypes = cast_dtypes

        self.feature_aggregator = FeatureAggregator(
            id_column, 
            self.get_config(
                depth_filters, 
                allowed_filters,
                numeric_columns, 
                allowed_num_aggregations
            ) if config is None else config,
            n_jobs,
        )

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Предобработка данных: переименование, преобразование типов, создание признаков.

        Args:
            df (pl.DataFrame): Сырые данные.

        Returns:
            pl.DataFrame: Обработанные данные.
        """
        raise NotImplementedError('Base class!')

    def generate_mask(self, data: pl.DataFrame, thr: float, unq: int) -> Tuple[bool]:
        """Генерирует маску для отбора признаков на основе порога.

        Args:
            data (pl.DataFrame): Входные данные.
            thr (float): Порог на минимальную долю ненулевых значений.
            unq (int): Порог на минимальное кол-во уникальных значений.

        Returns:
            Tuple[bool]: Маска.
        """
        self.mask = self.feature_aggregator.mask(self.preprocess(data), thr, unq)
        return self
    
    def iter_transform(self, data: pl.DataFrame, n_splits: int = 20) -> Generator[pl.DataFrame, None, None]:
        """Итеративно обрабатывает данные партициями.

        Args:
            data (pl.DataFrame): Входные данные.
            n_splits (int): Количество партиций.

        Yields:
            pl.DataFrame: Обработанная партиция данных.
        """
        hash_expr = (pl.col(self.id_column).hash() % n_splits).alias('partition')
        if self.verbose: pbar = tqdm(desc='Processing batches', total=n_splits)
        for batch in data.with_columns(hash_expr).sort('partition').partition_by('partition'):
            yield self.feature_aggregator.agg(self.preprocess(batch), self.mask)
            if self.verbose: pbar.update(1)
        if self.verbose: pbar.close()
    
    def transform(self, data: pl.DataFrame, strict: bool = True) -> pl.DataFrame:
        """Обрабатывает и агрегирует данные.

        Args:
            data (pl.DataFrame): Сырые данные.
            strict (bool): Поведение функции при пустых данных (По умолчанию True)
                True - возвращает ошибку, если на вход поступает пустой датафрейм
                False - возвращает пустой датафрейм
                По умолчанию для пустого датафрейма типы возвращаемых значений следующие:
                    `id_column`: pl.String
                    `feature_names`: [pl.Float64]
        
        Returns:
            pl.DataFrame: Агрегированные данные с учётом маски.
        """
        if data.is_empty():
            if strict: raise ValueError('Пустой датафрейм: `data`!')
            else:
                schema = {self.id_column: pl.String}
                for f in self.feature_names: schema[f] = pl.Float64
                return pl.DataFrame({f: [] for f in [self.id_column] + self.feature_names}, schema)
                
        data = self.preprocess(data)
        return self.feature_aggregator.agg(data, self.mask)

# 🐸 frog_mllib 🐸

set of my (sometimes not mine;) ml/dl utils for any life-case

### ml1_hw2_tools notebook
- утилиты отрисовки распределений и основных статистик датасета (df.info() на стероидах)
- отлично подойдет для eda нового датасета
- датасет для ноутбука: https://www.kaggle.com/competitions/titanic
- нужно переписать логику на polars 

### ml1_hw5_knn_aggregator notebook
- генератор новых фичей на основе соседей
- поддерживает hnsw и pynndescent
- нужно переписать на polars
- нужно добавить пример на нормальном датасете

### ml1_hw6_embedding_plotter notebook
- годная рисовалка 2d/3d эмбедов на bokeh
- нужно дописать часть с кластеризацией, возможно вынести кластеризацию в отдельный ноутбук как knn
- используется сгенерированный датасет: datasets/aliens.parquet, нужно добавить примеры на чем-то более боевом

### ml2_hw3_tools notebook
- рисовалка категориальных shap-ов
- какой-то датасет с недвижкой: datasets/train_sber.parquet
- рисовалка с кластеризацией градиентов
- функция умного перевзвешивания выборки

### t-SNE, UMAP
- обзор + туториал для работы
- много доп ссылок
- датасет для ноутбука: https://www.kaggle.com/datasets/zalando-research/fashionmnist/data

## dummy_agent_library_hf notebook
- решение 1-го юнита с курса https://huggingface.co/learn/agents-course/unit0/introduction
- учимся работать с hf
- пишем промты llm для взаимодействия с нашими tools-ами (кастыли для понимания работы)
- тоже самое, но через samolagents https://huggingface.co/learn/agents-course/unit1/tutorial

## flat_generator 
- генератор фичей на основе аналитических срезов
- нужно добавить ноутбук с примерами
- нужно добавить фильтрацию по стабильности распределения (например psi)
- для использования нужно унаследоваться от BaseGenerator и определить config + preprocess method

## cuml_benchmarks
- экосистема rapids по ускорению всего и вся на gpu
- самое полезное UMAP и HDBSCAN
- не так уж и прост в установке, возможно придется повозиться
- стырил ноутбук с сайта: https://rapids.ai/
- наверное стоит добавить более 'живых' примеров из практики
- у них есть свой docker контейнер: https://hub.docker.com/r/rapidsai/rapidsai/

### knn_comparisions notebook
- сравнение различных алгоритмов приближенного поиска соседей (включая алг. с хранением выборки на диске)
- обзор параметров, их влияние на скорость построения индекса и точность
- много дополнительного материала
- датасет для ноутбука: https://www.kaggle.com/datasets/zalando-research/fashionmnist/data

### general plotting
- linear.py - отрисовка весов линейной модели
- validation.py - рисует значение таргета по бинам какой-либо фичи во времени. Крайне полезно в аналитике модели
- plot_lines_and_bins.py - хороший график гистограммы + какой-либо фичи во времени

### general_utils
- utils.py - вывод полей объектов на заданную глубину
- cursors.py - курсоры для коннекшена с базами oracle или postgres-like. Возможность многопоточной асинхронной выгрузки.

## TODO
- add DL tools and plots from dl and rl cources
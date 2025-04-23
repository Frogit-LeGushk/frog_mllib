from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns, pandas as pd
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import balanced_accuracy_score
import faiss
import hnswlib
import voyager
import pynndescent


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def tick(self, msg=None, reset=True):
        time_passed = time.time() - self.start_time
        if msg is not None:
            print(msg + ':', round(time_passed, 3), 'sec')
        if reset:
            self.reset()
        return time_passed

    def reset(self):
        self.start_time = time.time()


def calc_recall(true_neighbors, pred_neighbors, k, exclude_self=False, return_mistakes=False):
    '''
    calculates recall@k of approx nearest neighbors
    
    true_neighbors: np.array (n_samples, k)
    pred_neighbors: np.array (n_samples, k)
    
    exclude_self: bool
        Если query_data была в трейне, считаем recall по k ближайшим соседям, не считая самого себя
    
    returns:
        recall@k
        mistakes: np.array (n_samples, ) с количеством ошибок
    '''
    n = true_neighbors.shape[0]
    n_success = []
    shift = int(exclude_self)

    for i in range(n):
        n_success.append(np.intersect1d(true_neighbors[i, shift:k+shift], pred_neighbors[i, shift:k+shift]).shape[0])
        
    recall = sum(n_success) / n / k
    if return_mistakes:
        mistakes = k - np.array(n_success)
        return recall, mistakes
    return recall


class AnnPlotter: # качество мерится только для классификации
    def __init__(self, X_tr, y_tr, k, metric, n_classes):
        '''
            X_tr, y_tr: обучалка
            k: кол-во соседей
            metric: метрика
            n_classes: кол-во классов
        '''
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.k = k
        self.metric = metric
        self.n_classes = n_classes
        self.algorithms = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def _calc_score_recall(self, predicted_nn_idx, y_val):
        y_pred = knn_predict_classification(predicted_nn_idx, self.y_tr, n_classes=self.n_classes)
        score =  balanced_accuracy_score(y_val, y_pred)
        recall = calc_recall(self.true_nn_idx, predicted_nn_idx, self.k)
        return score, recall
    
    def calc_true_nn(self, X_val, y_val):
        '''
            считает истинных соседей и истинное качество для X_val, y_val методом sklearn->brute
        '''
        brute_knn = NearestNeighbors(n_neighbors=self.k, algorithm='brute', metric=self.metric)
        brute_knn.fit(self.X_tr)
        timer = Timer()
        predicted_nn_idx = brute_knn.kneighbors(X_val, return_distance=False)
        self.brute_qps = X_val.shape[0] / (timer.tick('Brute | search_time') + 1e-10)
        self.true_nn_idx = predicted_nn_idx
        y_pred = knn_predict_classification(predicted_nn_idx, self.y_tr, n_classes=self.n_classes)
        self.brute_score = balanced_accuracy_score(y_val, y_pred)

    def _save_characteristics(self, lib_name, name, queries_per_second, score, recall, speed_param):
        '''
            index_build_time сохраняется отдельно
        '''
        self.algorithms[lib_name][name]['queries_per_second'].append(queries_per_second)
        self.algorithms[lib_name][name]['score'].append(score)
        self.algorithms[lib_name][name]['recall'].append(recall)
        
        if not isinstance(speed_param, tuple):
            speed_param = (speed_param,)
        self.algorithms[lib_name][name]['speed_params'].append(speed_param)
    
    def _prepare_single_sklearn(self, name, X_val, y_val, leaf_size, **kwargs):
        index = NearestNeighbors(n_neighbors=self.k, leaf_size=leaf_size, **kwargs)
        timer = Timer()
        index.fit(self.X_tr)
        index_build_time = timer.tick(f'\n{name}.leaf_size={leaf_size} | build_time')
        predicted_nn_idx = index.kneighbors(X_val, return_distance=False)
        search_time = timer.tick(f'{name}.leaf_size={leaf_size} | search_time')
        score, recall = self._calc_score_recall(predicted_nn_idx, y_val)
        return index_build_time, search_time, score, recall
        
    def prepare_sklearn(self, X_val, y_val, name, leaf_sizes=None, leaf_size=30, **kwargs):
        '''
            считает время построения индекса, скорость поиска, точность поиска, качество ...
                if leaf_sizes is not None: для каждого leaf_size из leaf_sizes 
                if leaf_sizes is None: для **kwargs
            name: ваше название алгоритма
            kwargs: аргументы в sklearn.neighbors.NearestNeighbors
        '''
        if name in self.algorithms['sklearn']:
            del self.algorithms['sklearn'][name]
            
        if leaf_sizes is None:
            leaf_sizes = [leaf_size]
            
        for leaf_size in leaf_sizes:
            index_build_time, search_time, score, recall = self._prepare_single_sklearn(name, X_val, y_val, leaf_size=leaf_size, **kwargs)
            queries_per_second = X_val.shape[0] / (search_time + 1e-10)
            self.algorithms['sklearn'][name]['index_build_time'].append(index_build_time)
            self._save_characteristics('sklearn', name, queries_per_second, score, recall, leaf_size)


    def _prepare_single_lsh(self, name, X_val, y_val, nbit):
        index = faiss.IndexLSH(X_val.shape[1], nbit)
        timer = Timer()
        index.train(self.X_tr)
        index.add(self.X_tr)
        index_build_time = timer.tick(f'\n{name}. nbit={nbit} | build_time')
        _, predicted_nn_idx = index.search(X_val, self.k)
        search_time = timer.tick(f'{name}. nbit={nbit} | search_time')
        score, recall = self._calc_score_recall(predicted_nn_idx, y_val)
        return index_build_time, search_time, score, recall

    
    def prepare_lsh(self, X_val, y_val, name, nbits=None, nbit=None):
        '''
            считает время построения индекса, скорость поиска, точность поиска, качество ...
                if nbits is not None: для каждого nbit из nbits 
                if nbits is None: для **kwargs
            name: ваше название алгоритма
        '''
        if name in self.algorithms['faiss_LSH']:
            del self.algorithms['faiss_LSH'][name]
            
        if nbits is None:
            nbits = [nbit]
            
        for nbit in nbits:
            index_build_time, search_time, score, recall = self._prepare_single_lsh(name, X_val, y_val, nbit=nbit)
            queries_per_second = X_val.shape[0] / (search_time + 1e-10)
            self.algorithms['faiss_LSH'][name]['index_build_time'].append(index_build_time)
            self._save_characteristics('faiss_LSH', name, queries_per_second, score, recall, nbit)

    
    def _build_single_hnswlib(self, name, ef_construction, M):
        index = hnswlib.Index(space='l2', dim=self.X_tr.shape[1]) # l2-hard-coded
        timer = Timer()
        index.init_index(max_elements=self.X_tr.shape[0], ef_construction=ef_construction, M=M)
        index.add_items(self.X_tr)
        index_build_time = timer.tick(f'\n{name}. ef_construction={ef_construction}, M={M} | build_time')
        return index, index_build_time

    
    def _search_single_hnswlib(self, name, index, X_val, y_val, ef_search):
        index.set_ef(ef_search)
        timer = Timer()
        predicted_nn_idx, _ = index.knn_query(X_val, self.k)
        search_time = timer.tick(f'{name}. ef_search={ef_search} | search_time')
        score, recall = self._calc_score_recall(predicted_nn_idx.astype(np.int64), y_val)
        return search_time, score, recall

    
    def prepare_hnswlib(self, X_val, y_val, name, ef_construction, M, ef_search_list=None, ef_search=None):
        '''
            считает время построения индекса, скорость поиска, точность поиска, качество ...
                if ef_search_list is not None: для каждого ef из ef_search_list 
                if ef_search_list is None: для **kwargs
            name: ваше название алгоритма
        '''
        if name in self.algorithms['hnswlib']:
            del self.algorithms['hnswlib'][name]
            
        if ef_search_list is None:
            ef_search_list = [ef_search]

        index, index_build_time = self._build_single_hnswlib(name, ef_construction, M)
        self.algorithms['hnswlib'][name]['index_build_time'].append(index_build_time)
        for ef_search in ef_search_list:
            search_time, score, recall = self._search_single_hnswlib(name, index, X_val, y_val, ef_search)
            queries_per_second = X_val.shape[0] / (search_time + 1e-10)
            self._save_characteristics('hnswlib', name, queries_per_second, score, recall, ef_search)


    def _build_single_voyager(self, name, ef_construction, M):
        index = voyager.Index(space=voyager.Space.Cosine, num_dimensions=self.X_tr.shape[1],
                             ef_construction=ef_construction, M=M, max_elements=self.X_tr.shape[0]) # l2-hard-coded
        timer = Timer()
        index.add_items(self.X_tr.astype(np.float32), num_threads=1)
        index_build_time = timer.tick(f'\n{name}. ef_construction={ef_construction}, M={M} | build_time')
        return index, index_build_time

    
    def _search_single_voyager(self, name, index, X_val, y_val, ef_search):
        timer = Timer()
        predicted_nn_idx, _ = index.query(X_val, k=self.k, query_ef=ef_search, num_threads=1)
        search_time = timer.tick(f'{name}. ef_search={ef_search} | search_time')
        score, recall = self._calc_score_recall(predicted_nn_idx.astype(np.int64), y_val)
        return search_time, score, recall

    
    def prepare_voyager(self, X_val, y_val, name, ef_construction, M, ef_search_list=None, ef_search=None):
        '''
            считает время построения индекса, скорость поиска, точность поиска, качество ...
                if ef_search_list is not None: для каждого ef из ef_search_list 
                if ef_search_list is None: для **kwargs
            name: ваше название алгоритма
        '''
        if name in self.algorithms['voyager']:
            del self.algorithms['voyager'][name]
            
        if ef_search_list is None:
            ef_search_list = [ef_search]

        index, index_build_time = self._build_single_voyager(name, ef_construction, M)
        self.algorithms['voyager'][name]['index_build_time'].append(index_build_time)
        for ef_search in ef_search_list:
            search_time, score, recall = self._search_single_voyager(name, index, X_val, y_val, ef_search)
            queries_per_second = X_val.shape[0] / (search_time + 1e-10)
            self._save_characteristics('voyager', name, queries_per_second, score, recall, ef_search)
    
    

    def _build_single_pynndescent(self, name, leaf_size, pruning_degree_multiplier, diversify_prob, **kwargs):
        timer = Timer()
        index = pynndescent.NNDescent(
            self.X_tr, leaf_size=leaf_size, pruning_degree_multiplier=pruning_degree_multiplier,
            diversify_prob=diversify_prob, **kwargs
        )
        index.prepare()
        suffix = f'leaf_size={leaf_size}, PDM={pruning_degree_multiplier}, DP={diversify_prob}'
        index_build_time = timer.tick(f'\n{name}. {suffix} | build_time')
        return index, index_build_time

    def _search_single_pynndescent(self, name, index, X_val, y_val):
        timer = Timer()
        predicted_nn_idx, _ = index.query(X_val, self.k)
        search_time = timer.tick(f'{name}. | search_time')
        score, recall = self._calc_score_recall(predicted_nn_idx.astype(np.int64), y_val)
        return search_time, score, recall

    
    def prepare_pynndescent(self, X_val, y_val, name, leaf_size=None, pruning_degree_multiplier=1.5, diversify_prob=1, **kwargs):
        '''
            считает время построения индекса, скорость поиска, точность поиска, качество ...
                leaf_size, pruning_degree_multiplier, diversify_prob можно задать как списками, так и числами
            name: ваше название алгоритма
            kwargs: аргументы в sklearn.neighbors.NearestNeighbors
        '''
        if name in self.algorithms['pynndescent']:
            del self.algorithms['pynndescent'][name]
            
        if not isinstance(leaf_size, list):
            leaf_size = [leaf_size]

        if not isinstance(pruning_degree_multiplier, list):
            pruning_degree_multiplier = [pruning_degree_multiplier]

        if not isinstance(diversify_prob, list):
            diversify_prob = [diversify_prob]

        for ls in leaf_size:
            for pdm in pruning_degree_multiplier:
                for dp in diversify_prob:
                    index, index_build_time = self._build_single_pynndescent(name, leaf_size=ls, pruning_degree_multiplier=pdm, diversify_prob=dp, **kwargs)
                    self.algorithms['pynndescent'][name]['index_build_time'].append(index_build_time)
                    search_time, score, recall = self._search_single_pynndescent(name, index, X_val, y_val, **kwargs)
                    queries_per_second = X_val.shape[0] / (search_time + 1e-10)
                    self._save_characteristics('pynndescent', name, queries_per_second, score, recall, (ls, pdm, dp))


    def _build_single_faiss(self, name, index_factory_str):
        index = faiss.index_factory(self.X_tr.shape[1], index_factory_str, faiss.METRIC_L2)
        timer = Timer()
        index.train(self.X_tr)
        index.add(self.X_tr)
        index_build_time = timer.tick(f'\n{name}. {index_factory_str} | build_time')
        return index, index_build_time

    
    def _search_single_faiss(self, name, index, X_val, y_val, nprobe):
        index.nprobe = nprobe
        timer = Timer()
        _, predicted_nn_idx = index.search(X_val, self.k)
        search_time = timer.tick(f'{name}. nprobe={nprobe} | search_time')
        score, recall = self._calc_score_recall(predicted_nn_idx.astype(np.int64), y_val)
        return search_time, score, recall

    
    def prepare_faiss(self, X_val, y_val, name, index_factory_str, nprobe=1):
        '''
            считает время построения индекса, скорость поиска, точность поиска, качество ...
                if nprobe is not None: для каждого prob из nprobe 
                if nprobe is None: для **kwargs
            name: ваше название алгоритма
        '''
        if name in self.algorithms['faiss']:
            del self.algorithms['faiss'][name]
            
        if not isinstance(nprobe, list):
            nprobe = [nprobe]

        index, index_build_time = self._build_single_faiss(name, index_factory_str)
        self.algorithms['faiss'][name]['index_build_time'].append(index_build_time)
        for prob in nprobe:
            search_time, score, recall = self._search_single_faiss(name, index, X_val, y_val, prob)
            queries_per_second = X_val.shape[0] / (search_time + 1e-10)
            self._save_characteristics('faiss', name, queries_per_second, score, recall, prob)


    
    def show(self):
        '''
            отрисовывает все имеющиеся алгоритмы
        '''
        mosaic = [['build', '.', 'search', 'search'], ['.', 'accuracy', 'accuracy', '.']]
        fig, ax = plt.subplot_mosaic(mosaic, figsize=(18, 10))

        speed_param_names = {
            'sklearn': ('leaf_size',),
            'faiss_LSH': ('nbits',),
            'hnswlib': ('efSearch',),
            'voyager': ('efSearch',),
            'pynndescent': ('leaf_size', 'PDM', 'DP'),
            'faiss': ('nprobe',)
        }
        build_times = []
        algorithm_names = []
        algorithm_suffixes = []
        for lib_name in self.algorithms:
            speed_param_name = speed_param_names[lib_name]
            for algorithm_name in self.algorithms[lib_name]:
                recalls = self.algorithms[lib_name][algorithm_name]['recall']
                qps = self.algorithms[lib_name][algorithm_name]['queries_per_second']

                # SEARCH
                ax['search'].plot(recalls, qps, marker='o', lw=3, label=algorithm_name)

                speed_params = self.algorithms[lib_name][algorithm_name]['speed_params']
                for i in range(len(recalls)):
                    text = ', '.join([f'{speed_param_name[j]}={speed_params[i][j]}' for j in range(len(speed_param_name))])
                    ax['search'].annotate(text=text, xy=(recalls[i], qps[i]), fontsize=20)

                # ACCURACY
                scores = self.algorithms[lib_name][algorithm_name]['score']
                ax['accuracy'].plot(recalls, scores, marker='o', lw=3, label=algorithm_name)
                
                # BUILD
                index_build_times = self.algorithms[lib_name][algorithm_name]['index_build_time']
                build_times.extend(index_build_times)
                algorithm_names.extend([algorithm_name] * len(index_build_times))
                if len(index_build_times) == 1:
                    algorithm_suffixes.append('')
                else:
                    algorithm_suffixes.extend([' | ' + ', '.join([f'{speed_param_name[j]}={speed_param[j]}' for j in range(len(speed_param_name))]) for speed_param in speed_params])

        y_labels = [name + suff for name, suff in zip(algorithm_names, algorithm_suffixes)]
        sns.barplot(x=build_times, y=y_labels, hue=algorithm_names, ax=ax['build'], legend=False)
        
        ax['search'].axhline(self.brute_qps, color='red', ls='--', label=f'brute: {self.brute_qps:.1e} qps', lw=3)
        ax['accuracy'].axhline(self.brute_score, color='red', ls='--', label=f'brute balanced_acc: {self.brute_score:.3f}', lw=3)

        # ax['search'].legend(fontsize=15)
        ax['search'].grid(True)
        ax['search'].tick_params('both', labelsize=15)
        ax['search'].set_yscale('log')
        ax['search'].set_xlabel(f'recall@{self.k}', fontsize=15)
        ax['search'].set_ylabel('queries per second', fontsize=15)

        ax['accuracy'].legend(fontsize=15, bbox_to_anchor=(1.8, 1))
        ax['accuracy'].grid(True)
        ax['accuracy'].tick_params('both', labelsize=15)
        ax['accuracy'].set_xlabel(f'recall@{self.k}', fontsize=15)
        ax['accuracy'].set_ylabel('balanced_accuracy', fontsize=15)

        ax['build'].grid(True)
        ax['build'].tick_params('both', labelsize=15)
        ax['build'].set_xlabel('time, sec', fontsize=15)
        ax['build'].set_title('build time', fontsize=15)

        plt.show()


def knn_predict_classification(neighbor_ids, tr_labels, n_classes, distances=None, weights='uniform'):
    '''
    по расстояниям и айдишникам получает ответ для задачи классификации
    
    distances: (n_samples, k) - расстояния до соседей
    neighbor_ids: (n_samples, k) - айдишники соседей
    tr_labels: (n_samples,) - метки трейна (числа 0, 1, 2, ..., n_classes - 1)
    n_classes: кол-во классов
    
    returns:
        labels: (n_samples,) - предсказанные метки
    '''
    
    n, k = neighbor_ids.shape

    labels = np.take(tr_labels, neighbor_ids)
    labels = labels + np.arange(n).reshape(-1, 1) * n_classes

    if weights == 'uniform':
        w = np.ones(n * k)
    elif weights == 'distance' and distances is not None:
        w = 1. / (distances.ravel() + 1e-10)
    else:
        raise NotImplementedError()

    labels = np.bincount(labels.ravel(), weights=w, minlength=n * n_classes)
    labels = labels.reshape(n, n_classes).argmax(axis=1).ravel()
    return labels

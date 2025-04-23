import numpy as np
from matplotlib import colormaps
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

import time
from matplotlib.colors import ListedColormap
import glasbey # палитры

def make_meshgrid(x, y, h=.02, odd_range=1):
    '''
    делает координатную сетку по входным точкам
    
    x, y: исходные точки, "вокруг которых" делаем сетку
    h: шаг сетки
    odd_range: величина zoom-out'a от точек х, y
    '''
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    x_min, x_max = x.min() - odd_range * x_range, x.max() + odd_range * x_range
    y_min, y_max = y.min() - odd_range * y_range, y.max() + odd_range * y_range
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, model, xx, yy, **params):
    '''
    предсказываем моделью точки сетки и отрисовываем разделяющую прямую и цвета
    
    xx, yy: точки сетки
    '''
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    stm = time.time()
    Z = model.predict(grid)
    print('predict time:', round(time.time() - stm, 3), 'sec. |', 'grid shape:', grid.shape)
    
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, **params, alpha=0.5, levels=np.unique(Z).shape[0])

    
def plot_classification(model, X, hue, odd_range=0.3, mesh_h=0.02, figsize=None, ax=None):
    '''
    рисует разделяющую прямую модели для 2D-задачи классификации (<= 10 классов из-за палитр :)
    
    model: что-то, у чего есть метод .predict, который дает ту картину предсказаний, какую вы хотите
    X: точки выборки, которые нужно отрисовать вместе с разделяющей прямой
    hue: их классы
    
    odd_range: чем больше, тем больше zoom-out от поданых на вход точек
    mesh_h: шаг сетки
    '''
    n_classes = np.unique(hue).shape[0]
    color_list = glasbey.create_palette(n_classes, as_hex=True)
    cmap = ListedColormap(color_list)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # делаем частую сетку из точек "с центром" в поданных точках x, y
    xx, yy = make_meshgrid(X[:, 0], X[:, 1], h=mesh_h, odd_range=odd_range)
    
    # предсказываем моделью все точки сетки, отрисовываем цвета
    plot_contours(ax, model, xx, yy, cmap=cmap)
    
    # отрисовываем туда же поданные на вход точки x, y
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=hue, hue_order=np.unique(hue), palette=color_list, marker='o', s=2,
                    edgecolor='black', ax=ax, legend=True)

    ax.legend(fontsize=10, markerscale=4)
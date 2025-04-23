import numpy as np
import matplotlib.pyplot as plt

def plot_weights(model, features, top_k=20):
    '''
    рисует значения весов линейной модели при признаках
    
    top_k: рисовать первые top_k весов по модулю
    '''
    # подготовка необходимого
    num_features_to_plot = min(top_k, len(features))
    weights = model.coef_[0]
    sorted_idx = np.argsort(-np.abs(weights))
    bias = model.intercept_[0]
    
    fig, ax = plt.subplots(figsize=(8, num_features_to_plot / 2))
    
    # сами бары
    container = ax.barh(y=features[sorted_idx][:top_k][::-1], width=weights[sorted_idx][:top_k][::-1])
    
    # приписать к ним значения весов
    ax.bar_label(container, weights[sorted_idx][:top_k][::-1].round(3), color='red', fontsize=15)
    
    # настройка ах'a
    ax.margins(0.2, 0.05)
    ax.set_title(f'bias: {bias:.1e}', fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_xlabel('weight', fontsize=15)
    
    plt.show()

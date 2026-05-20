import torch

import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def make_evaluation_on_L_p_grid(loader, decoder, L_grid, p_grid, b_size=4096, size=int(1e5)):
    loader.batch_generator.noise_model.qmc_flag = False
    results = np.zeros((2, len(L_grid), len(p_grid)))
    for i, L in tqdm(enumerate(L_grid)):
        for j, p in enumerate(p_grid):
            loader.adjust_L_p(L, p)
            decoder.adjust_L(L)

            with torch.no_grad():
                acc, std, _, _ = loader.make_evaluation(decoder, b_size=b_size, total_size=size)
                results[0, i, j] = acc
                results[1, i, j] = std
    return results


def find_threshold(acc_results, p_grid, L_grid):
    def curve(p_noise_L, p_th, A, B, C, D, mu):
        xx = (p_noise_L[0] - p_th) * (p_noise_L[1]) ** mu
        return A + B * xx + C * xx ** 2 + D * xx ** 3

    p_target = acc_results.flatten()
    p_feature = np.repeat(np.array(p_grid).reshape(1, -1), len(L_grid), axis=0).flatten()
    L_feature = np.repeat(L_grid, len(p_grid)).flatten()
    x_data = np.vstack((p_feature, L_feature))
    popt, pcov = curve_fit(curve, x_data, p_target, bounds=([0, -np.inf, -np.inf, -np.inf, -np.inf, 0],
                                                            [1, np.inf, np.inf, np.inf, np.inf, np.inf]))
    return popt, pcov, x_data

def make_threshold_plot(L_grid, p_grid, popt, results, ax=None, save_path=None):
    def curve(p_noise_L, p_th, A, B, C, D, mu):
        xx = (p_noise_L[0] - p_th) * (p_noise_L[1]) ** mu
        return A + B * xx + C * xx ** 2 + D * xx ** 3

    if ax is None:
        #fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=500)
        fig = plt.figure(figsize=(10, 10), dpi=500)
        ax_none = True
    else:
        ax_none = False

    color = plt.cm.plasma
    for i in range(len(L_grid)):
        c = color.colors[i * len(L_grid) ** 2]
        plt.errorbar(x=p_grid, y=results[0][i], yerr=results[1][i],
                     label=fr'$L={{{L_grid[i]}}}$', alpha=0.85, color=c, ecolor='red',
                     linestyle='--', marker='o', markersize=3.)
    plt.grid(True)
    p_grid = np.array(p_grid)
    p_grid_min = np.min(p_grid)
    p_grid_max = np.max(p_grid)
    #plt.xticks(np.linspace(p_grid_min, p_grid_max, 21), rotation=45)
    y_min = np.min(results[0])
    y_max = np.max(results[0])
    plt.yticks(np.linspace(y_min, y_max, 21))

    mean_L = int(np.mean(L_grid))
    mean_L_data = np.vstack((p_grid, [mean_L] * len(p_grid)))
    plt.plot(mean_L_data[0], curve(mean_L_data, *popt), '--', color='green', linewidth=3,
            label=fr'$f(p,p^*={{{np.round(popt[0], 3)}}}, L={{{mean_L}}})$')
    #plt.vlines(np.round(popt[0], 3), y_min, y_max, linewidth=1, color='black')
    plt.legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=500)

    if ax_none:
        plt.tight_layout();
        plt.close(fig)
        result = fig
    else:
        result = 0
    return result

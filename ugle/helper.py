from omegaconf import OmegaConf
from ugle.process import euclidean_distance
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, gaussian_kde
import os
import matplotlib
from ugle.logger import ugle_path
from copy import deepcopy
import ranky as rk
from itertools import combinations
from scipy.stats import linregress
from scipy.stats import spearmanr 
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib
import scienceplots
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import random

def search_results(folder, filename):

    for root, dirs, files in os.walk(f'{ugle_path}/{folder}'):
        if filename in files:
            return os.path.join(root, filename)
    return None


def get_all_results_from_storage(datasets: list, algorithms: list, folder: str, empty: str = 'minus_ten',
                                 collect_all: bool = False):
    if empty == 'minus_ten':
        empty_result = OmegaConf.create(
            {'conductance_mean': -10.0,
             'conductance_std': -10.0,
             'f1_mean': -10.0,
             'f1_std': -10.0,
             'modularity_mean': -10.0,
             'modularity_std': -10.0,
             'nmi_mean': -10.0,
             'nmi_std': -10.0,
             })
    elif empty == 'zeros':
        empty_result = OmegaConf.create(
            {'conductance_mean': 0.0,
             'conductance_std': 0.0,
             'f1_mean': 0.0,
             'f1_std': 0.0,
             'modularity_mean': 0.0,
             'modularity_std': 0.0,
             'nmi_mean': 0.0,
             'nmi_std': 0.0,
             })

    result_holder = OmegaConf.create({})

    for dataset in datasets:
        for algo in algorithms:
            filename = f"{dataset}_{algo}.pkl"
            file_found = search_results(folder, filename)
            if file_found:
                result = pickle.load(open(file_found, "rb"))
                if collect_all:
                    # parse seed
                    return_results = np.zeros(shape=(4, 10))
                    # go thru every seed
                    for i, seed_result in enumerate(result.results):
                        # go thru every best hps configuration
                        for metric_result in seed_result.study_output:
                            # go thru each best metric in configuration
                            for metric in metric_result.metrics:
                                # add result to correct place
                                if metric == 'f1':
                                    return_results[0, i] = metric_result.results[metric]
                                elif metric == 'nmi':
                                    return_results[1, i] = metric_result.results[metric]
                                elif metric == 'modularity':
                                    return_results[2, i] = metric_result.results[metric]
                                elif metric == 'conductance':
                                    return_results[3, i] = metric_result.results[metric]

                    result_holder[f"{dataset}_{algo}"] = return_results.tolist()

                else:
                    result_holder[f"{dataset}_{algo}"] = result.average_results
            else:
                if collect_all:
                    if empty == 'minus_ten':
                        place_holder = np.ones(shape=(4, 10)) * -10
                        result_holder[f"{dataset}_{algo}"] = place_holder.tolist()
                    elif empty == 'zeros':
                        result_holder[f"{dataset}_{algo}"] = np.zeros(shape=(4, 10))
                else:
                    result_holder[f"{dataset}_{algo}"] = empty_result

    return result_holder


def get_values_from_results_holder(result_holder, dataset_name, metric_name, return_std=False):
    metric_values = []
    if return_std:
        std_values = []
    for k, v in result_holder.items():
        if k.__contains__(dataset_name):
            metric_values.append(v[f'{metric_name}_mean'])
            if return_std:
                std_values.append(v[f'{metric_name}_std'])

    if return_std:
        return metric_values, std_values
    else:
        return metric_values


def make_test_performance_object(datasets, algorithms, metrics, seeds, folder):
    # get results object
    result_object = np.zeros(shape=(len(datasets), len(algorithms), len(metrics), len(seeds)))
    try:
        result_holder = get_all_results_from_storage(datasets, algorithms, folder, collect_all=True)
    except:
        return result_object
    for d, dataset in enumerate(datasets):
        for a, algo in enumerate(algorithms):
            result_object[d, a] = result_holder[f"{dataset}_{algo}"]
    return result_object


def calculate_abs_std(result_object, datasets, metrics):
    # calculate deviation over each seed
    std_object = np.zeros(shape=np.shape(result_object)[:-1])
    for d, _ in enumerate(datasets):
        for m, _ in enumerate(metrics):
            std_object[d, :, m] = np.std(result_object[d, :, m, :] , axis=1)
    return std_object


def calculate_ranking_performance(result_object, datasets, metrics, seeds, calc_ave_first=False):
    if calc_ave_first:
        # calculate ranking on each seed
        ranking_object = np.zeros(shape=np.shape(result_object)[:-1])
        for d, _ in enumerate(datasets):
            for m, metric_name in enumerate(metrics):
                metric_values = np.mean(result_object[d, :, m, :] , axis=1)
                last_place_zero = np.argwhere(np.array(metric_values) == -10).flatten()
                if metric_name != 'conductance':
                    ranking_of_algorithms = np.flip(np.argsort(metric_values)) + 1
                else:
                    ranking_of_algorithms = np.argsort(metric_values) + 1
                ranking_of_algorithms[last_place_zero] = len(ranking_of_algorithms)
                ranking_object[d, :, m] = ranking_of_algorithms
    else:
        # calculate ranking on each seed
        ranking_object = np.zeros_like(result_object)
        for d, _ in enumerate(datasets):
            for m, metric_name in enumerate(metrics):
                for s, _, in enumerate(seeds):
                    metric_values = result_object[d, :, m, s]
                    last_place_zero = np.argwhere(np.array(metric_values) == -10).flatten()
                    if metric_name != 'conductance':
                        ranking_of_algorithms = np.flip(np.argsort(metric_values)) + 1
                    else:
                        ranking_of_algorithms = np.argsort(metric_values) + 1
                    ranking_of_algorithms[last_place_zero] = len(ranking_of_algorithms)
                    ranking_object[d, :, m, s] = ranking_of_algorithms
    
    return ranking_object


def create_result_bar_chart(dataset_name, algorithms, folder, default_algos, default_folder, ax=None):
    """
    displays the results in matplotlib with dashed borders for original comparison on single dataset
    :param hpo_results: hyperparameter results
    :param default_results: default parameter results
    :param dataset_name: the dataset on which the results were gathered
    :param ax: optional input axis
    :return ax: axis on which figure is displayed

    """
    #plt.rcParams["font.family"] = "Times New Roman"
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))

    #alt_colours = ['#dc143c', '#0bb5ff', '#2ca02c', '#800080']
    alt_colours = ['C2', 'C0', 'C1', 'C3']

    # extract key arrays for results
    result_holder = get_all_results_from_storage([dataset_name], algorithms, folder, empty='zeros')
    default_result_holder = get_all_results_from_storage([dataset_name], default_algos, default_folder, empty='zeros')

    f1, f1_std = get_values_from_results_holder(result_holder, dataset_name, 'f1', return_std=True)
    nmi, nmi_std = get_values_from_results_holder(result_holder, dataset_name, 'nmi', return_std=True)
    modularity, modularity_std = get_values_from_results_holder(result_holder, dataset_name, 'modularity',
                                                                return_std=True)
    conductance, conductance_std = get_values_from_results_holder(result_holder, dataset_name, 'conductance',
                                                                  return_std=True)

    default_f1, default_f1_std = get_values_from_results_holder(default_result_holder, dataset_name, 'f1',
                                                                return_std=True)
    default_nmi, default_nmi_std = get_values_from_results_holder(default_result_holder, dataset_name, 'nmi',
                                                                  return_std=True)
    default_modularity, default_modularity_std = get_values_from_results_holder(default_result_holder, dataset_name,
                                                                                'modularity',
                                                                                return_std=True)
    default_conductance, default_conductance_std = get_values_from_results_holder(default_result_holder, dataset_name,
                                                                                  'conductance',
                                                                                  return_std=True)
    
    print(f'{dataset_name} & nmi & nmi-d & f1 & f1-d & modularity & modularity-d & conductance & conductance-d \\\\')
    for a, algo in enumerate(algorithms):
        print(f"{algo} &", end=" ")
        if nmi[a] > default_nmi[a]: 
            print(f'\\textbf{{{nmi[a]} $\pm${nmi_std[a]}}} & {default_nmi[a]}$\pm${default_nmi_std[a]} &', end=" ")
        else:
            print(f'{nmi[a]} $\pm${nmi_std[a]} & \\textbf{{{default_nmi[a]}$\pm${default_nmi_std[a]}}} &', end=" ")

        if f1[a] > default_f1[a]: 
            print(f'\\textbf{{{f1[a]} $\pm${f1_std[a]}}} & {default_f1[a]}$\pm${default_f1_std[a]} &', end=" ")
        else:
            print(f'{f1[a]} $\pm${f1_std[a]} & \\textbf{{{default_f1[a]}$\pm${default_f1_std[a]}}} &', end=" ")

        if modularity[a] > default_modularity[a]: 
            print(f'\\textbf{{{modularity[a]} $\pm${modularity_std[a]}}} & {default_modularity[a]}$\pm${default_modularity_std[a]} &', end=" ")
        else:
            print(f'{modularity[a]} $\pm${modularity_std[a]} & \\textbf{{{default_modularity[a]}$\pm${default_modularity_std[a]}}} &', end=" ")

        if conductance[a] > default_conductance[a]: 
            print(f'\\textbf{{{conductance[a]} $\pm${conductance_std[a]}}} & {default_conductance[a]}$\pm${default_conductance_std[a]} &', end=" ")
        else:
            print(f'{conductance[a]} $\pm${conductance_std[a]} & \\textbf{{{default_conductance[a]}$\pm${default_conductance_std[a]}}} \\\\')

        # repeat for rest 

    bar_width = 1 / 4
    x_axis_names = np.arange(len(algorithms))

    # plot hyperparameter results in full colour
    ax.bar(x_axis_names, f1, yerr=f1_std, ecolor=alt_colours[0],
           width=bar_width, facecolor=alt_colours[0], alpha=0.8, linewidth=0, label='f1')
    ax.errorbar(x_axis_names, f1, f1_std, ecolor=alt_colours[0], elinewidth=2.5, linewidth=0)
    ax.bar(x_axis_names + bar_width, nmi, yerr=nmi_std, ecolor=alt_colours[1],
           width=bar_width, facecolor=alt_colours[1], alpha=0.8, linewidth=0, label='nmi')
    ax.errorbar(x_axis_names + bar_width, nmi, nmi_std, ecolor=alt_colours[1], elinewidth=2.5, linewidth=0)
    ax.bar(x_axis_names + (2 * bar_width), modularity, yerr=modularity_std, ecolor=alt_colours[2],
           width=bar_width, facecolor=alt_colours[2], alpha=0.8, linewidth=0, label='modularity')
    ax.errorbar(x_axis_names + (2 * bar_width), modularity, modularity_std, ecolor=alt_colours[2], elinewidth=2.5, linewidth=0)
    ax.bar(x_axis_names + (3 * bar_width), conductance, yerr=conductance_std, ecolor=alt_colours[3],
           width=bar_width, facecolor=alt_colours[3], alpha=0.8, linewidth=0, label='conductance')
    ax.errorbar(x_axis_names + (3 * bar_width), conductance, conductance_std, ecolor=alt_colours[3], elinewidth=2.5, linewidth=0)

    # plot default parameters bars in dashed lines
    blank_colours = np.zeros(4)
    ax.bar(x_axis_names, default_f1, yerr=default_f1_std, width=bar_width,
           facecolor=blank_colours, edgecolor='black', linewidth=2, linestyle='--', label='default values')
    ax.errorbar(x_axis_names, default_f1, default_f1_std, ecolor='black', elinewidth=3, linewidth=3, linestyle='none')
    ax.bar(x_axis_names + bar_width, default_nmi, yerr=default_nmi_std, width=bar_width,
           facecolor=blank_colours, edgecolor='black', linewidth=2, linestyle='--')
    ax.errorbar(x_axis_names + bar_width, default_nmi, default_nmi_std, ecolor='black', elinewidth=3, linewidth=3, linestyle='none')
    ax.bar(x_axis_names + (2 * bar_width), default_modularity, yerr=default_modularity_std, width=bar_width, 
           facecolor=blank_colours, edgecolor='black', linewidth=2, linestyle='--')
    ax.errorbar(x_axis_names + (2 * bar_width), default_modularity, default_modularity_std, ecolor='black', elinewidth=3, linewidth=3, linestyle='none')
    ax.bar(x_axis_names + (3 * bar_width), default_conductance, yerr=default_conductance_std,
           facecolor=blank_colours, width=bar_width, edgecolor='black', linewidth=2, linestyle='--')
    ax.errorbar(x_axis_names + (3 * bar_width), default_conductance, default_conductance_std, ecolor='black', elinewidth=3, linewidth=3, linestyle='none')

    # create the tick labels for axis
    ax.set_xticks(x_axis_names - 0.5 * bar_width)
    ax.set_xticklabels(algorithms, ha='left', rotation=-45, position=(-0.3, 0))
    ax.set_axisbelow(True)

    # Axis styling.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    # tighten the layout
    if dataset_name == 'citeseer':
        title_name = 'CiteSeer'
    elif dataset_name == 'texas' or dataset_name == 'cornell' or dataset_name == 'wisc' or dataset_name == 'cora':
        title_name = dataset_name.capitalize()
    else:
         title_name = dataset_name.upper()
    ax.set_title(title_name, y=0.95, fontsize=98)
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(42)
    
    ax.set_ylim(bottom=0)
    #ax.set_xlabel('Algorithm')
    #ax.set_ylabel('Test Metric Result')

    return ax


def create_dataset_landscape_fig(ax, title, datasets, algorithms, metrics, seeds, folder, calc_ave_first=False):
    # get datasest information
    try:
        clustering = pickle.load(open(f"{ugle_path}/dataset_stats/clustering.pkl", "rb"))
        closeness = pickle.load(open(f"{ugle_path}/dataset_stats/closeness.pkl", "rb"))
    except:
        from ugle.datasets import compute_datasets_info
        clustering, closeness = compute_datasets_info(datasets)
        pickle.dump(clustering, open(f"{ugle_path}/dataset_stats/clustering.pkl", "wb"))
        pickle.dump(closeness, open(f"{ugle_path}/dataset_stats/closeness.pkl", "wb"))

    # make data landscape
    dataset_points = np.array([clustering, closeness])
    n_dataset = len(clustering)
    n_points = 512
    x_range = 0.8
    y_range = 0.6
    X, Y = np.meshgrid(np.linspace(0, x_range, n_points), np.linspace(0, y_range, n_points))
    xy = np.vstack((X.flatten(), Y.flatten())).T
    # find the idx of the dataset point closest in the landscape
    all_distances = np.zeros((n_dataset, np.shape(xy)[0]))
    for i in range(n_dataset):
        all_distances[i, :] = [euclidean_distance(x, dataset_points[:, i]) for x in xy]
    closest_point = np.argsort(all_distances, axis=0)[0]
    Z = closest_point.reshape((n_points, n_points))

    # get results
    result_object = make_test_performance_object(datasets, algorithms, metrics, seeds, folder)
    ranking_object = calculate_ranking_performance(result_object, datasets, metrics, seeds, calc_ave_first=calc_ave_first)
    if calc_ave_first:
        # calculate the ranking averages 
        ranking_over_datasets = np.mean(ranking_object, axis=2)
    else:
        # calculate the ranking averages 
        ranking_over_datasets = np.mean(ranking_object, axis=(2, 3))
    
    # get best algorithm for each dataset
    best_algo_idx_over_datasets = np.argsort(np.array(ranking_over_datasets), axis=1)[:, 0]
    # work out best algorithm for any point on dataset landscape
    best_algo_over_space = best_algo_idx_over_datasets[Z]
    n_best_algos = len(np.unique(best_algo_over_space))
    # scale the best algorithm space to work nicely with visualisation
    dataset_replace = np.array(list(range(n_dataset)))
    dataset_replace[np.unique(best_algo_over_space)] = np.array(list(range(n_best_algos)))
    scaled_best_algo_over_space = dataset_replace[best_algo_over_space]
    # get colouring for the visualisation
    cmap = plt.get_cmap('Blues', n_best_algos)
    cax = ax.imshow(np.flip(scaled_best_algo_over_space, axis=0), cmap=cmap, interpolation='nearest',
                    extent=[0, x_range, 0, y_range])
    # add color bar
    best_algo_list = [algorithms[i] for i in np.unique(best_algo_over_space)]
    best_algo_list = [albel.split("_")[0] for albel in best_algo_list]
    if len(best_algo_list) == 2:
        tick_spacing = [0.25, 0.75]
    elif len(best_algo_list) == 3:
        tick_spacing = [0.33, 1., 1.66]
    else:
        tick_spacing = np.linspace(0.5, n_best_algos - 1.5, n_best_algos)
    cbar = plt.colorbar(cax, ticks=range(n_best_algos), orientation="vertical", shrink=0.8)
    cbar.set_ticks(tick_spacing)
    cbar.ax.set_yticklabels(best_algo_list, fontsize=16, ha='left', rotation=-40)
    # add dataset locations
    ax.scatter(clustering, closeness, marker='x', s=20, color='black', label='datasets')
    # add nice visual elements
    ax.set_xlabel('Clustering Coefficient', fontsize=16)
    ax.set_ylabel('Closeness Centrality', fontsize=16)
    ax.set_title(title, fontsize=22, pad=15)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18)
    return ax


def print_2d_array(array):
    for row in array:
        row_str = " & ".join(f"{val:.3f}" for val in row)
        print(row_str + " \\\\")


def create_ranking_charts(datasets: list, algorithms: list, metrics: list, seeds: list, folder: str, title_name: str, 
     calc_ave_first: bool=False, set_legend: bool=True, ax=None, ax1=None, fig=None, fig1=None):

    # init axis
    if not ax and not fig:
        fig, ax = plt.subplots(figsize=(12, 5))
    elif (ax and not fig) or (fig and not ax):
        print('please provide both fig and ax or neither')

    if not ax1:
        fig1, ax1 = plt.subplots(figsize=(12, 5))
    elif (ax1 and not fig1) or (fig1 and not ax1):
        print('please provide both fig1 and ax1 or neither')

    #fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 5))
    #ax = axes[0]
    #ax1 = axes[1]
    #ax2= axes[2]

    # fetch results
    result_object = make_test_performance_object(datasets, algorithms, metrics, seeds, folder)

    # calculate ranking 
    ranking_object = calculate_ranking_performance(result_object, datasets, metrics, seeds, calc_ave_first=calc_ave_first)

    # aggregate ranks 
    if calc_ave_first:

        overrall_ranking = np.transpose(ranking_object, (1, 0, 2))
        overrall_ranking = overrall_ranking.reshape(-1, overrall_ranking.shape[1]*overrall_ranking.shape[2])
        overrall_ranking = rk.center(overrall_ranking, axis=1, method='kendalltau')
        
        rank_on_metric_agg_over_dataset = []
        for d, _ in enumerate(datasets):
            rank_on_metric_agg_over_dataset.append(rk.center(ranking_object[d], axis=1, method='kendalltau'))
        
        rank_on_dataset_agg_over_metric = []
        for m, _ in enumerate(metrics):
            rank_on_dataset_agg_over_metric.append(rk.center(ranking_object[:, :, m].T, axis=1, method='kendalltau'))
    else:

        overall_ranking = np.transpose(ranking_object, (1, 0, 2, 3))
        overall_ranking = overall_ranking.reshape(-1, ranking_object.shape[1]*ranking_object.shape[2],*ranking_object.shape[3])
        overall_ranking = rk.center(overall_ranking, axis=1, method='kendalltau')

        rank_on_metric_agg_over_dataset = []
        for d, _ in enumerate(datasets):
            rank_on_metric_agg_over_dataset.append(rk.center(ranking_object[d].reshape(-1, ranking_object.shape[1]* ranking_object.shape[2]), axis=1, method='kendalltau'))
        
        rank_on_dataset_agg_over_metric = []
        for m, _ in enumerate(metrics):
            temp_rank_object = np.transpose(ranking_object[:, :, m], (1, 0, 2))
            temp_rank_object = temp_rank_object.reshape(-1, temp_rank_object.shape[1]* temp_rank_object.shape[2])
            rank_on_dataset_agg_over_metric.append(rk.center(temp_rank_object, axis=1, method='kendalltau'))


    ranking_over_datasets = np.array(rank_on_metric_agg_over_dataset) # should be metrics * algorithms dimensionality 
    ranking_over_metrics = np.array(rank_on_dataset_agg_over_metric) # should be dataset * algorithms dimensionality 

    from scipy.stats import spearmanr 

    # next step is to calculate the p-value of how correlated the aggregated ranks are... 
    spear_stats_metrics = spearmanr(ranking_over_metrics, axis=1, alternative='greater')
    spear_stats_datasets = spearmanr(ranking_over_datasets, axis=1, alternative='greater')
    print(metrics)
    print_2d_array(spear_stats_metrics.pvalue)
    print(datasets)
    print_2d_array(spear_stats_datasets.pvalue)

    algorithms = np.array(algorithms)
    ranking_order = np.argsort(overrall_ranking)
    algo_labels = algorithms[ranking_order]
    algo_labels = [albel.split("_")[0] for albel in algo_labels]

    # plot the by metric ranking
    for rank_on_metric_ave_over_dataset, metric in zip(ranking_over_metrics, metrics):
        ax.plot(algo_labels, rank_on_metric_ave_over_dataset[ranking_order],
                marker="o", label=metric, alpha=0.5, zorder=20)

    # plot the overrall ranking
    ax.scatter(algo_labels, overrall_ranking[ranking_order],
               marker="x", c='black', s=20, label='overall rank', zorder=100)

    # set legend
    if set_legend:
        ax.legend(loc='upper center', fontsize=15, ncol=2, bbox_to_anchor=(0.5, -0.35))
    # configure y axis
    ax.set_ylim(0, 10.5)
    ax.set_ylabel('Concensus Ranking\nover Datasets', fontsize=15)
    ax.tick_params(axis='y', labelsize=18)
    # configure x axis
    plt.setp(ax.xaxis.get_majorticklabels(), ha='left', rotation=-40, fontsize=14)
    # create offset transform by 5 points in x direction
    dx = 5 / 72.
    dy = 0 / 72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    # apply offset transform to all x ticklabels.
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() - offset)
    # set title
    ax.set_title(f'{title_name}: Ranking by Metric', fontsize=20)
    plt.tight_layout()


    # plot the by dataset ranking
    colours = plt.get_cmap('tab20').colors
    for rank_on_dataset_ave_over_metric, dataset, c in zip(ranking_over_datasets, datasets, colours):
        ax1.plot(algo_labels, rank_on_dataset_ave_over_metric[ranking_order],
                 marker="o", label=dataset, alpha=0.5, zorder=20, c=c)
    # plot the overrall ranking
    ax1.scatter(algo_labels, overrall_ranking[ranking_order],
                marker="x", c='black', s=20, label='overall rank', zorder=100)
    # set legend
    if set_legend:
        ax1.legend(loc='upper center', fontsize=14, ncol=4, bbox_to_anchor=(0.5, -0.35))
    # set y axis
    ax1.set_ylim(0, 10.5)
    ax1.set_ylabel('Consensus Ranking\nover Metrics', fontsize=15)
    ax1.tick_params(axis='y', labelsize=20)
    # set x axis
    plt.setp(ax1.xaxis.get_majorticklabels(), ha='left', rotation=-40, fontsize=14)
    # create offset transform by 5 points in x direction
    dx = 5 / 72.
    dy = 0 / 72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig1.dpi_scale_trans)
    # apply offset transform to all x ticklabels.
    for label in ax1.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() - offset)
    # set title
    ax1.set_title(f'{title_name}: Ranking by Dataset' , fontsize=20)
    plt.tight_layout()

    return ax, ax1


def create_rand_dist_fig(ax, algorithms, all_ranks_per_algo, set_legend=False):
    """
    def kendall_w(expt_ratings):
        if expt_ratings.ndim!=2:
            raise 'ratings matrix must be 2-dimensional'
        m = expt_ratings.shape[0] # raters
        n = expt_ratings.shape[1] # items rated
        denom = m**2*(n**3-n)
        rating_sums = np.sum(expt_ratings, axis=0)
        S = n*np.var(rating_sums)
        return 12*S/denom
    
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    x_axis = np.arange(0, len(algorithms), 0.001)

    result_object = make_test_performance_object(datasets, algorithms, metrics, seeds, folder)
    ranking_object = calculate_ranking_performance(result_object, datasets, metrics, seeds, calc_ave_first=calc_ave_first)

    # datasets, algorithms, metrics, seeds ->  tests(datasets+metrics), seeds, algorithms
    ranking_object = np.transpose(ranking_object, axes=(0, 2, 3, 1))
    ranking_object = ranking_object.reshape((-1,) + ranking_object.shape[2:])
    wills_order = []
    for test in ranking_object:
        wills_order.append(kendall_w(test))
    wills_order = np.array(wills_order)
    print(title + f"{np.mean(wills_order):.3f} +- {np.std(wills_order):.3f}")


    """
    cm = plt.get_cmap('tab10').colors
    #ax.set_prop_cycle(color=[cm(1.*i/len(algorithms)) for i in range(len(algorithms))])
    x_axis = np.arange(0, len(algorithms), 0.001)
    max_y = 0
    for j, algo_ranks in enumerate(all_ranks_per_algo.T):
        try:
            kde = gaussian_kde(algo_ranks)
            y_axis = kde.evaluate(x_axis)
            ax.plot(x_axis, y_axis, label=algorithms[j], zorder=10, color=cm[j])
            print(f'{algorithms[j]}: {cm[j]}')
            max_y = max(max_y, max(y_axis)) + 0.05
        except:
            ax.axvline(x=algo_ranks[0], label=algorithms[j], zorder=10, color=cm[j])

    if set_legend:
        #ax.legend(loc='best', fontsize=20, ncol=1, bbox_to_anchor=(1, -0.5))
        #ax.legend(loc='upper center', fontsize=15, ncol=3, bbox_to_anchor=(0.475, -0.5))
        ax.set_xlabel(r'$r$' + ' : algorithm ranking', fontsize=20)
        #ax.set_xlabel('algorithm rank distribution over all tests', fontsize=18)

    #ax.set_ybound(0, 3)
    ax.set_xbound(0.9, 10.1)
    ax.set_ybound(0, max_y)
    ax.set_ylabel(r'$f_{j}(r)$', fontsize=20) #kde estimatation of rank distribution

    #ax.text(0.4, 0.85, ave_overlap_text, fontsize=20, transform=ax.transAxes, zorder=1000)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=15)

    return ax


def create_big_figure(datasets, algorithms, folder, default_algos, default_folder):
    """
    creates figure for all datasets tested comparing default and hpo results
    """
    # create holder figure
    nrows, ncols = 4, 3
    col_n, row_n = 0, 0
    fig, axs = plt.subplots(nrows, ncols, figsize=(56, 74))

    for dataset_name in datasets:
        # create a figure on the axis
        if row_n == nrows:
            col_n += 1
            row_n = 0

        axs[row_n, col_n] = create_result_bar_chart(dataset_name, algorithms, folder, default_algos, default_folder, axs[row_n, col_n])
        row_n += 1

    axs[row_n, col_n].spines['top'].set_visible(False)
    axs[row_n, col_n].spines['bottom'].set_visible(False)
    axs[row_n, col_n].spines['left'].set_visible(False)
    axs[row_n, col_n].spines['right'].set_visible(False)
    axs[row_n, col_n].set_xticks([])
    axs[row_n, col_n].set_yticks([])
    handles = []
    alt_colours = ['C2', 'C0', 'C1', 'C3']
    metrics = ['f1', 'nmi', 'modularity', 'conductance']
    for i in range(len(alt_colours)):
        handles.append(mlines.Line2D([], [], color=alt_colours[i], linewidth=8, label=metrics[i]))

    handles.append(mlines.Line2D([], [], color='black', linewidth=8, linestyle='--', label='Default\nHyperparameters'))

    blank_ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    blank_ax.axis('off')
    blank_ax.legend(handles=handles, bbox_to_anchor=(0.975, 0.2), fontsize=90)
    #axs[0].legend(loc='upper right', bbox_to_anchor=(1, 0.95))
    #for item in axs[0].get_legend().get_texts():
    #   item.set_fontsize(36)

    fig.tight_layout()
    fig.savefig(f"{ugle_path}/figures/hpo_investigation.eps", format='eps', bbox_inches='tight')
    return


def create_algo_selection_on_dataset_landscape(datasets, algorithms, default_algos, metrics, seeds, folder, default_folder, titles):

    fig, ax = plt.subplots(2, 2, figsize=(15, 7.5))

    titles[0] = r'$\mathcal{T}_{(ave-d)}$'
    titles[1] = r'$\mathcal{T}_{(ave-hpo)}$'
    titles[2] = r'$\mathcal{T}_{(ind-d)}$'
    titles[3] = r'$\mathcal{T}_{(ind-hpo)}$'

    ax[0, 0] = create_dataset_landscape_fig(ax[0, 0], titles[0], datasets, default_algos, metrics, seeds, default_folder, calc_ave_first=True)
    ax[0, 1] = create_dataset_landscape_fig(ax[0, 1], titles[1], datasets, algorithms, metrics, seeds, folder, calc_ave_first=True)
    ax[1, 0] = create_dataset_landscape_fig(ax[1, 0], titles[2], datasets, default_algos, metrics, seeds, default_folder)
    ax[1, 1] = create_dataset_landscape_fig(ax[1, 1], titles[3], datasets, algorithms, metrics, seeds, folder)
    
    fig.tight_layout()
    fig.savefig(f"{ugle_path}/figures/le_dataset_landscape_comparison.png")
    return


def create_comparison_figures(datasets: list, algorithms: list, metrics: list, seeds: list, folder: str,
                              default_algos: list, default_folder: str, titles: list):
    # create holder figure
    nrows, ncols = 2, 2
    fig0, axs0 = plt.subplots(nrows, ncols, figsize=(15, 7.5))

    # create holder figure
    nrows, ncols = 2, 2
    fig1, axs1 = plt.subplots(nrows, ncols, figsize=(15, 7.5))

    titles[0] = r'$\mathcal{T}_{(ave-d)}$'
    titles[1] = r'$\mathcal{T}_{(ave-hpo)}$'
    titles[2] = r'$\mathcal{T}_{(ind-d)}$'
    titles[3] = r'$\mathcal{T}_{(ind-hpo)}$'

    axs0[0, 0], axs1[0, 0] = create_ranking_charts(datasets, default_algos, metrics, seeds, default_folder,
                                           title_name=titles[0], calc_ave_first=True, set_legend=False, ax=axs0[0, 0], ax1=axs1[0, 0], fig=fig0,
                                           fig1=fig1)
    axs0[0, 1], axs1[0, 1] = create_ranking_charts(datasets, algorithms, metrics, seeds, folder, title_name=titles[1],
                                           calc_ave_first=True, set_legend=False, ax=axs0[0, 1], ax1=axs1[0, 1], fig=fig0,
                                           fig1=fig1)
    axs0[1, 0], axs1[1, 0] = create_ranking_charts(datasets, default_algos, metrics, seeds, default_folder,
                                           title_name=titles[2],
                                           set_legend=False, ax=axs0[1, 0], ax1=axs1[1, 0], fig=fig0,
                                           fig1=fig1)
    axs0[1, 1], axs1[1, 1] = create_ranking_charts(datasets, algorithms, metrics, seeds, folder, title_name=titles[3],
                                            set_legend=True, ax=axs0[1, 1], ax1=axs1[1, 1], fig=fig0,
                                           fig1=fig1)

    fig0.tight_layout()
    fig1.tight_layout()

    fig0.savefig(f"{ugle_path}/figures/le_ranking_comparison_metrics.png", bbox_inches='tight')
    fig1.savefig(f"{ugle_path}/figures/le_ranking_comparison_datasets.png", bbox_inches='tight')
    return


def create_rand_dist_comparison(datasets: list, algorithms: list, metrics: list, seeds: list, folder: str, default_algos: list, default_folder: str):

    # create holder figure
    nrows, ncols = 2, 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 7.5))

    result_object = make_test_performance_object(datasets, algorithms, metrics, seeds, folder)
    default_result_object = make_test_performance_object(datasets, default_algos, metrics, seeds, default_folder)
    
    ranking_object = calculate_ranking_performance(result_object, datasets, metrics, seeds).squeeze(0)[:, 0, :].T
    default_ranking_object = calculate_ranking_performance(default_result_object, datasets, metrics, seeds).squeeze(0)[:, 0, :].T

    # rank values 
    def wordr(rankings):
        m = rankings.shape[0] # raters
        n = rankings.shape[1] # items rated
        denom = m**2*(n**3-n)
        rating_sums = np.sum(rankings, axis=0)
        S = n*np.var(rating_sums)
        return 1 - 12*S/denom

    hpo_w = wordr(ranking_object)
    default_w = wordr(default_ranking_object)

    titles_0 = 'Default ' + r'$W$: ' + str(round(default_w, 3))
    titles_1 = 'HPO ' + r'$W$: ' + str(round(hpo_w, 3))

    ax[0] = create_rand_dist_fig(ax[0], algorithms, default_ranking_object, set_legend=False)
    ax[1] = create_rand_dist_fig(ax[1], algorithms, ranking_object, set_legend=True)

    n_comparisons = result_object.flatten().shape[0]
    rankings = np.zeros((n_comparisons, 2))

    result_object = result_object.flatten()
    default_result_object = default_result_object.flatten()
    for i in range(n_comparisons):
        if result_object[i] > default_result_object[i]:
            rankings[i] = [1, 2]
        elif default_result_object[i] < result_object[i]:
            rankings[i] = [2, 1]
        else:
            rankings[i] = [1.5, 1.5]
    means_hpo = np.mean(rankings, axis=0)[0]
    means_def = np.mean(rankings, axis=0)[1]
    std_hpo = np.std(rankings, axis=0)[0]
    std_def = np.std(rankings, axis=0)[1]
    print("Default: " + f"{means_def:.3f}\\neq{std_def:.3f}")
    print("HPO: " + f"{means_hpo:.3f}\\neq{std_hpo:.3f}")

    titles_0 += f' FCR: {means_def:.3f}'
    titles_1 += f' FCR: {means_hpo:.3f}'
    ax[0].set_title(titles_0, fontsize=20)
    ax[1].set_title(titles_1, fontsize=20)

    #fig.suptitle('Algorithm F1 Score Rank Distribution\n Estimation Comparison on Cora', fontsize=24)
    fig.tight_layout()
    fig.savefig(f'{ugle_path}/figures/le_rand_dist_comparison.png', bbox_inches='tight')
    return


def create_all_paper_figures(datasets, algorithms, metrics, seeds, folder, default_folder, default_algos):
    titles = ['a) Default HPs w/ AveSeed', 'b) HPO w/ AveSeed', 'c) Default HPs w/ SeedRanking', 'd) HPO w/ SeedRanking']
    #create_rand_dist_comparison(datasets, algorithms, metrics, seeds, folder, default_algos, default_folder, deepcopy(titles))
    #print('done rand dist')
    #create_algo_selection_on_dataset_landscape(datasets, algorithms, default_algos, metrics, seeds, folder, default_folder, deepcopy(titles))
    #print('done algo selection')


    ## only need one figure for this 
    #fig0, axs0 = plt.subplots(1, 1, figsize=(7.5, 5.5))
    #fig1, axs1 = plt.subplots(1, 1, figsize=(7.5, 5.5))
    #axs0, axs1 = create_ranking_charts(datasets, algorithms, metrics, seeds, folder, title_name=r'$\mathcal{T}_{(ave-hpo)}$', calc_ave_first=True, set_legend=True,
                                    # ax=axs0, ax1=axs1, fig=fig0, fig1=fig1)
    # fig0.tight_layout()
    # fig1.tight_layout()

    # fig0.savefig(f"{ugle_path}/figures/le_ranking_comparison_metrics.png", bbox_inches='tight')
    # fig1.savefig(f"{ugle_path}/figures/le_ranking_comparison_datasets.png", bbox_inches='tight')

    # print('done metric + dataset comparison')
    create_big_figure(datasets, algorithms, folder, default_algos, default_folder)
    # print('done fats%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%o')
    return

# creates a tikz figure to show someone
#fig, ax = plt.subplots(1, 1, figsize=(15, 15))
#ax = create_rand_dist_fig(ax, r'Framework Rank Distinction Coefficient $\Omega(\mathcal{\hat{T}}_{(hpo)}) : $', datasets, algorithms, metrics, seeds, folder, calc_ave_first=True, set_legend=True)
#fig.tight_layout()
#fig.savefig(f'{ugle_path}/figures/tkiz_fig.png', bbox_inches='tight')



def reshape_ranking_to_test_object(ranking_object):
    # datasets, algorithms, metrics, seeds ->  tests(datasets+metrics), seeds, algorithms
    ranking_object = np.transpose(ranking_object, axes=(0, 2, 3, 1))
    ranking_object = ranking_object.reshape((-1,) + ranking_object.shape[2:])
    return ranking_object

def og_randomness(ranking_object):
    # og W coefficient where draws are the lowest rank that would occur from the ties
    n_draws = 0
    wills_order = []
    for test in ranking_object:
        for rs, rs_test in enumerate(test):
            unique_scores, counts = np.unique(rs_test, return_counts=True)
            if len(unique_scores) != 10:
                n_draws += 10 - len(unique_scores)
        wills_order.append(kendall_w(test))
    wills_order = np.array(wills_order)
    #print(f'n_draws: {n_draws}', end=',  ')

    return np.mean(wills_order)

def og_newOld_randomness(ranking_object):
    # og W coefficient where draws are the average rank between those tied
    wills_order = []
    for test in ranking_object:
        rank_test = np.zeros_like(test)
        for rs, rs_test in enumerate(test):
            rank_test[rs] = rank_scores(rs_test)
        wills_order.append(kendall_w(rank_test))
    wills_order = np.array(wills_order)
    return np.mean(wills_order)

def ties_randomness(ranking_object):
    # TIES W coefficient where draws are the average rank between those tied
    wills_order = []
    for test in ranking_object:
        rank_test = np.zeros_like(test)
        for rs, rs_test in enumerate(test):
            rank_test[rs] = rank_scores(rs_test)
        wills_order.append(w_randomness_w_ties(rank_test))
    wills_order = np.array(wills_order)
    return np.mean(wills_order)

def wasserstein_randomness(ranking_object):
    #  W coefficient from wasserstein where draws are the average rank between those tied
    # ranking_object shape [tests, seeds, algorithms]
    wills_order = []
    for test in ranking_object:
        rank_test = np.zeros_like(test)
        for rs, rs_test in enumerate(test):
            rank_test[rs] = rank_scores(rs_test)
        wills_order.append(w_rand_wasserstein(rank_test))
    wills_order = np.array(wills_order)
    return np.mean(wills_order)

def nrule(n):
    j = 0 
    for i in range(1,n+1):
        j += (i*(i-1))/2
    return j

def w_rand_wasserstein(rankings):
    n_algorithms = rankings.shape[1]
    # rank_test[:, 0] -> all seeds, one algorithm
    wass_agg = []
    for i in range(n_algorithms):
        for j in range(i):  # Iterate up to i to stay to the left of the diagonal
            wass_agg.append(wasserstein_distance(rankings[:, i], rankings[:, j]))
    return 1 - (np.sum(wass_agg) / nrule(n_algorithms))

def rank_scores(scores):
    # Get indices in descending order
    indices = np.flip(np.argsort(scores))

    # Initialize an array to store the ranks
    ranks = np.zeros_like(indices, dtype=float)

    # Assign ranks to the sorted indices
    ranks[indices] = np.arange(len(scores)) + 1

    # Find unique scores and their counts
    unique_scores, counts = np.unique(scores, return_counts=True)

    # Calculate mean ranks for tied scores
    for score, count in zip(unique_scores, counts):
        if count > 1:
            score_indices = np.where(scores == score)[0]
            mean_rank = np.mean(ranks[score_indices])
            ranks[score_indices] = mean_rank

    return ranks

def w_randomness_w_ties(test):
    n = test.shape[0]
    a = test.shape[1]

    Ti = np.zeros(n)
    for j in range(n):
        _, counts = np.unique(test[j, :], return_counts=True)
        tied_groups = counts[counts > 1]
        Ti[j] = np.sum(tied_groups ** 3 - tied_groups)

    T = np.sum(Ti) * n
    
    R = np.sum(test, axis=0)
    R = sum(r ** 2 for r in R)

    W = ((12*R) - 3 * (n**2) * a *((a + 1) ** 2)) / (((n ** 2) * a * ((a ** 2)  - 1)) -  T)

    return 1 - W

def create_random_results(n_tests, n_seeds, n_algorithms):
    ranks = []
    for i in range(n_tests):
        test = []
        for j in range(n_seeds):
            test.append(np.random.permutation(n_algorithms))
        ranks.append(test)
    return np.asarray(ranks)

def extract_results(datasets, algorithms, folder, extract_validation=False, return_df=False):
    # modularity and conductance may have different hyperparameters or model selection points 
    mod_results = []
    con_results = []
    columns = ['Dataset', 'Algorithm', 'Seed', 'A_Metric', 'A_Metric_Value', 'B_Metric', 'B_Metric_Value']
    df = pd.DataFrame(columns=columns)

    for dataset in datasets:
        for algo in algorithms:
            filename = f"{dataset}_{algo}.pkl"
            file_found = search_results(folder, filename)
            if file_found:
                result = pickle.load(open(file_found, "rb"))
            
                for seed_result in result.results:
                    for metric_result in seed_result.study_output:
                        if 'modularity' in metric_result.metrics:
                            if extract_validation: 
                                mod = metric_result.validation_results['modularity']['modularity']
                            else: 
                                mod = metric_result.results['modularity']
                            f1 = metric_result.results['f1']
                            nmi = metric_result.results['nmi']
                            mod_results.append([mod, f1, nmi])
                            df.loc[len(df)] = {'Dataset': dataset, 'Algorithm': algo, 'Seed': seed_result.seed, 'A_Metric': 'Modularity', 'A_Metric_Value': mod, 'B_Metric': 'F1', 'B_Metric_Value': f1}
                            df.loc[len(df)] = {'Dataset': dataset, 'Algorithm': algo, 'Seed': seed_result.seed, 'A_Metric': 'Modularity', 'A_Metric_Value': mod, 'B_Metric': 'NMI', 'B_Metric_Value': nmi}


                        if 'conductance' in metric_result.metrics:
                            if extract_validation: 
                                con = metric_result.validation_results['conductance']['conductance']
                            else:
                                con = metric_result.results['conductance']
                            f1 = metric_result.results['f1']
                            nmi = metric_result.results['nmi']
                            con_results.append([con, f1, nmi])
                            df.loc[len(df)] = {'Dataset': dataset, 'Algorithm': algo, 'Seed': seed_result.seed, 'A_Metric': 'Conductance', 'A_Metric_Value': con, 'B_Metric': 'F1', 'B_Metric_Value': f1}
                            df.loc[len(df)] = {'Dataset': dataset, 'Algorithm': algo, 'Seed': seed_result.seed, 'A_Metric': 'Conductance', 'A_Metric_Value': con, 'B_Metric': 'NMI', 'B_Metric_Value': nmi}
            else:
                print(f"did not find: {filename}")
                # can only really do this because i know that there's 10 seeds
                for seed in range(10):
                    mod_results.append([0., 0., 0.])
                    con_results.append([0., 0., 0.])


    mod_results = np.asarray(mod_results)
    con_results = np.asarray(con_results)
    if return_df:
        return mod_results, con_results, df
    else: 
        return mod_results, con_results

def extract_supervised_results(datasets, algorithms, folder):
    f1_nmi_results = np.zeros((len(datasets)*len(algorithms)*10, 2))
    i = 0
    
    for dataset in datasets:
        for algo in algorithms:
            filename = f"{dataset}_{algo}.pkl"
            file_found = search_results(folder, filename)
            if file_found:
                result = pickle.load(open(file_found, "rb"))
            
                for seed_result in result.results:
                    for metric_result in seed_result.study_output:
                        if 'f1' in metric_result.metrics:
                            f1_nmi_results[i, 0] = metric_result.results['f1']
                        if 'nmi' in metric_result.metrics:
                            f1_nmi_results[i, 1] = metric_result.results['nmi']
                    i += 1
            else:
                print(f"did not find: {filename}")
    return f1_nmi_results

def calc_percent_increase(f1_nmi_results, dmod_results, dcon_results):

    diff = np.zeros((f1_nmi_results.shape[0], 4))
    for i, row in enumerate(f1_nmi_results):
        # f1 - modf1 
        diff[i, 0] = (row[0] - dmod_results[i, 1]) / dmod_results[i, 1]

        # f1 - conf1 
        diff[i, 1] = (row[0] - dcon_results[i, 1]) / dcon_results[i, 1]

        # nmi - modnmi
        diff[i, 2] = (row[1] - dmod_results[i, 2]) / dmod_results[i, 2]

        # nmi - connmi 
        diff[i, 3] = (row[1] - dcon_results[i, 2]) / dcon_results[i, 2]

    increases = np.mean(diff, axis=0)
    print(f'Increase from using Modularity to select for F1 compared to just F1: {increases[0]*100:.2f}%')
    print(f'Increase from using Conductance to select for F1 compared to just F1: {increases[1]*100:.2f}%')
    print(f'Increase from using Modularity to select for NMI compared to just NMI: {increases[2]*100:.2f}%')
    print(f'Increase from using Conductance to select for NMI compared to just NMI: {increases[3]*100:.2f}%')
    return

def print_dataset_table(datasets, algorithms, folder, power_d=2):
    # extract results
    for dataset in datasets:
        mod_results, con_results = extract_results([dataset], algorithms, folder)
        print(dataset, end = ' ')
        print(f'mod_f1_nmi: {np.mean(mod_results[:, 0]):.3f} & {np.mean(mod_results[:, 1]):.3f} & {np.mean(mod_results[:, 2]):.3f}', end=' ')
        print(f'con_f1_nmi: {np.mean(con_results[:, 0]):.3f} & {np.mean(con_results[:, 1]):.3f} & {np.mean(con_results[:, 2]):.3f}', end=' ')
        print(f'correlations: ', end=' ')
        for x, y in [[mod_results[:, 0], mod_results[:, 1]], [mod_results[:, 0], mod_results[:, 2]], [con_results[:, 0], con_results[:, 1]], [con_results[:, 0], con_results[:, 2]]]:
            coefficients = np.polyfit(x, y, power_d)
            poly = np.poly1d(coefficients)
            # Calculate predicted values
            predicted_y = poly(x)
            r_value_quad = np.round(r2_score(y, predicted_y), 3)
            print(f'& {r_value_quad:.2f}', end=' ')
        print('')

def print_algo_table(datasets, algorithms, folder, power_d=2):
    for algorithm in algorithms:
        mod_results, con_results = extract_results(datasets, [algorithm], folder)
        print(algorithm, end = ' ')
        print(f'mod_f1_nmi: {np.mean(mod_results[:, 0]):.3f} & {np.mean(mod_results[:, 1]):.3f} & {np.mean(mod_results[:, 2]):.3f}', end=' ')
        print(f'con_f1_nmi: {np.mean(con_results[:, 0]):.3f} & {np.mean(con_results[:, 1]):.3f} & {np.mean(con_results[:, 2]):.3f}', end=' ')
        print(f'correlations: ', end=' ')
        for x, y in [[mod_results[:, 0], mod_results[:, 1]], [mod_results[:, 0], mod_results[:, 2]], [con_results[:, 0], con_results[:, 1]], [con_results[:, 0], con_results[:, 2]]]:
            coefficients = np.polyfit(x, y, power_d)
            poly = np.poly1d(coefficients)
            # Calculate predicted values
            predicted_y = poly(x)
            r_value_quad = np.round(r2_score(y, predicted_y), 3)
            print(f'& {r_value_quad:.2f}', end=' ')
        print('')

def kendall_w(expt_ratings):
    if expt_ratings.ndim!=2:
        raise 'ratings matrix must be 2-dimensional'
    m = expt_ratings.shape[0] # raters
    n = expt_ratings.shape[1] # items rated
    denom = m**2*(n**3-n)
    rating_sums = np.sum(expt_ratings, axis=0)
    S = n*np.var(rating_sums)
    return 1 - (12*S/denom)

def rank_values(scores):
    argsort_array = scores.argsort().argsort()
    ranks_array = np.empty_like(argsort_array)
    ranks_array[argsort_array] = np.arange(len(scores))
    return ranks_array

def compute_w_order_for_mod_and_con(mod_results, con_results, testnames, algorithms, datasets):
    different_comparisons = [mod_results[:, 0], mod_results[:, 1], mod_results[:, 2], con_results[:, 0], con_results[:, 1], con_results[:, 2]]
    total_w_order = []

    for result, testname in zip(different_comparisons, testnames):
        tempresult = result.reshape(10, len(algorithms), len(datasets), order="F")
        tempresult = np.transpose(tempresult, axes=(2, 0, 1))
        ranking = np.zeros_like(tempresult)
        for d, dataset_res in enumerate(tempresult):
            for s, seed_res in enumerate(dataset_res):
                if testname == 'Conductance':
                    ranking[d, s, :] = rank_values(seed_res) + 1
                else:
                    ranking[d, s, :] = np.flip(rank_values(seed_res) + 1)

        w_order = []
        for test in ranking:
            w_order.append(kendall_w(test))
        w_order = np.array(w_order)
        total_w_order.append(w_order)
        # the W order of each metric tests
        print(f"{testname} W Order: {np.mean(w_order):.3f} +- {np.std(w_order):.3f}")
    # the W order over the whole 66% experiment
    total_w_order = np.asarray(total_w_order)
    
    return total_w_order

def unsupervised_prediction_graph(datasets, algorithms, folder, title):
    plt.style.use(['science', 'nature'])
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.dpi"] = 144
    nature_colours = ["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"]
    algorithm_colors = ["#434982", "#00B945", "#EA907A", "#845B97", "#4F8A8B", "#FFCB74", "#B5DEFF"]
    # extract results 
    if "q4" in title:
        extract_validation = True
    else: 
        extract_validation = False
    
    mod_results, con_results, df = extract_results(datasets, algorithms, folder, extract_validation=extract_validation, return_df=True)
    testnames = ["Modularity", "Modularity F1", "Modularity NMI", "Conductance", "Conductance F1", "Conductance NMI"]

    total_w_order = compute_w_order_for_mod_and_con(mod_results, con_results, testnames, algorithms, datasets)
    print(f"Overall W Order: {np.mean(total_w_order):.3f} +- {np.std(total_w_order):.3f}")
    # the W order of each comparison 
    indv_tests = np.array([[0, 1], [0, 2], [3, 4], [3, 5]])
    moretestnames = [testnames[i] for i in indv_tests[:, 1]]
    more_w_orders = [np.mean(np.asarray(total_w_order[i])) for i in indv_tests]
    for n, w in zip(moretestnames, more_w_orders):
        print(f"W Order {n}: {w:.2f}")
   
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7, 4))
    for i, ax in enumerate(axes.flat):
        if i == 0:
            x_label = "Modularity"
            y_label = "F1"
            x = mod_results[:, 0]
            y = mod_results[:, 1]
            W_order = more_w_orders[0]
        elif i == 2:
            x_label = "Modularity"
            y_label = "NMI"
            x = mod_results[:, 0]
            y = mod_results[:, 2]
            W_order = more_w_orders[1]
        elif i == 1:
            x_label = "Conductance"
            y_label = "F1"
            x = con_results[:, 0]
            y = con_results[:, 1]
            W_order = more_w_orders[2]
        elif i == 3: 
            x_label = "Conductance"
            y_label = "NMI"
            x = con_results[:, 0]
            y = con_results[:, 2]
            W_order = more_w_orders[3]

        print(f"\n{x_label} -> {y_label}")
        # compute regression and give correlation 
        slope, intercept, _, _, _ = linregress(x, y)
        y_line_predicted = (x * slope) + intercept
        r_value_line = r2_score(y, y_line_predicted)
        print(f"Coefficient of Determination (R^2): {r_value_line:.2f}")
        x_space = np.linspace(np.min(x), np.max(x), 200)
        y_line = (x_space * slope) + intercept

        # Fit a quadratic 
        coefficients = np.polyfit(x, y, 2)
        poly = np.poly1d(coefficients)
        # Calculate predicted values
        predicted_y = poly(x)
        r_value_quad = np.round(r2_score(y, predicted_y), 3)
        print(f"Variance Explained by Quadratic: {r_value_quad}")
        y_line_quad = poly(x_space)

        # spearmans  
        spearman_stats = spearmanr(a=x, b=y)
        print(f"Spearmans Correlation Coefficient: {spearman_stats.correlation:.2f}")
        
        if "Large" in title:
            continue
        markers =  ['.', 'v', '2', 'D', '*', 'X']
        handles = []
        for opt_in in range(len(markers)): 
            df_axis = df[(df['Dataset'] == datasets[opt_in]) & (df['A_Metric'] == x_label) & (df['B_Metric'] == y_label)]
            handles.append(mlines.Line2D([], [], label=datasets[opt_in], color="#474747", marker=markers[opt_in], linestyle='None'))

            for a, col in enumerate(algorithm_colors):
                if "_" in algorithms[a]:
                    algo_name = algorithms[a].split("_")[0]
                else:
                    algo_name = algorithms[a]
                if opt_in == 5:
                    handles.append(mlines.Line2D([], [], label=algo_name, color=col, marker="s", linestyle='None'))

                ax.scatter(df_axis[df_axis['Algorithm'] == algorithms[a]]['A_Metric_Value'], df_axis[df_axis['Algorithm'] == algorithms[a]]['B_Metric_Value'], color=col, s=10, marker=markers[opt_in])

        handles.append(mlines.Line2D([], [], color=nature_colours[3], linestyle='dashed', linewidth=2, label='Quadratic Fit'))#, $R^2$' + f': {r_value_quad:.2f}'))
        handles.append(mlines.Line2D([], [], color=nature_colours[3], linewidth=2, label='Linear Fit'))
        ax.plot(x_space, y_line, color=nature_colours[3], linewidth=2)#, label=r'Linear Fit', $R^2$' + f': {r_value_line:.2f}')
        ax.plot(x_space, y_line_quad, color=nature_colours[3], linestyle='dashed', linewidth=2)#, label=r'Quadratic Fit')#, $R^2$' + f': {r_value_quad:.2f}', linewidth=2)
        if i == 0:
            ax.set_ylabel(y_label, fontsize=12)
        if i == 2:
            ax.set_ylabel(y_label, fontsize=12) 
            ax.set_xlabel(x_label, fontsize=12)
        if i == 3:
            ax.set_xlabel(x_label, fontsize=12)
        
        ax.set_ylim(0, 1)
        ax.set_title(f"{x_label[:3]}" + r' $\rightarrow$ '+ y_label + " (l-"+ r"$R^2$" + f": {r_value_line:.2f}, q-" + r"$R^2$"  + f": {r_value_quad:.2f}, " + r"$W$" + f": {W_order:.2f})", fontsize=9)
        #if i == 1:
        #    ax.legend(handles=handles, bbox_to_anchor=(1.10, 1.5), fontsize=8, ncols=1)
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=7)
       
    print_algo_table(datasets, algorithms, folder)
    print_dataset_table(datasets, algorithms, folder)

    if "Large" in title:
        print('TO THE POWER OF ONE')
        print_algo_table(datasets, algorithms, folder)
        print_dataset_table(datasets, algorithms, folder)

    if "Large" not in title:
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.85, bottom=0.22, hspace=0.33)
        #fig.add_subplot(111, frameon=False)
        #plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        #plt.legend(handles=handles, bbox_to_anchor=(0, 0), fontsize=8, ncols=6)
        blank_ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        blank_ax.axis('off')
        blank_ax.legend(handles=handles, loc='lower center', fontsize=8, ncols=6)
        #plt.tight_layout()
        if "66" in title:
            title_name = '66_data'
        elif "33" in title:
            title_name  = '33_data'
        else:
            title_name = title.replace(" ", "_")
            title_name = title_name.replace("\'", "")
        plt.savefig(f'./figures/unsupervised_limit/{title_name}.eps', format='eps')
    return

def create_abs_performance_figure(datasets, algorithms, folder, title, plot_dims, figsize):
    plt.style.use(['science', 'nature'])
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.dpi"] = 144
    plt.rcParams["hatch.linewidth"] = 0.3

    nature_colours = ["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"]

    nrows, ncols = plot_dims[0], plot_dims[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    x_axis_names = [algo.split("_")[0] for algo in algorithms]
    x_axis_points = np.arange(len(x_axis_names))
    bar_width = 1/6

    for i, ax in enumerate(axes.flat):
        dataset = datasets[i]
        mod_results, con_results = extract_results([dataset], algorithms, folder)
        testnames = ["Modularity", "Modularity F1", "Modularity NMI", "Conductance", "Conductance F1", "Conductance NMI"]
        total_w_order = np.mean(compute_w_order_for_mod_and_con(mod_results, con_results, testnames, algorithms, [dataset]))

        if dataset == 'dblp':
            ax.set_title("DBLP" + ' (' + r'$W$' + f': {total_w_order:.2f})', fontsize=15)
        elif 'synth' in dataset:
            dataset_name = r"$A$" + ": " + dataset.split("_")[1] + '  ' + r"$X$" ": " + dataset.split("_")[2]
            dataset_name = dataset_name.replace("disjoint", "Distinct")
            dataset_name = dataset_name.replace("random", "Random")
            dataset_name = dataset_name.replace("complete", "Null")
            ax.set_title(dataset_name + ' (' + r'$W$' + f': {total_w_order:.2f})', fontsize=12)
        else:
            ax.set_title(dataset.capitalize() + ' (' + r'$W$' + f': {total_w_order:.2f})', fontsize=15)


        mod = []
        mod_f1 = []
        mod_nmi = []
        con = []
        con_f1 = []
        con_nmi = []

        mod_std = []
        mod_f1_std = []
        mod_nmi_std = []
        con_std = []
        con_f1_std = []
        con_nmi_std = []

        for algo in algorithms:
            mod_results, con_results = extract_results([dataset], [algo], folder)
            mod.append(np.mean(mod_results[:, 0]))
            mod_f1.append(np.mean(mod_results[:, 1]))
            mod_nmi.append(np.mean(mod_results[:, 2]))
            con.append(np.mean(con_results[:, 0]))
            con_f1.append(np.mean(con_results[:, 1]))
            con_nmi.append(np.mean(con_results[:, 2]))

            mod_std.append(np.std(mod_results[:, 0]))
            mod_f1_std.append(np.std(mod_results[:, 1]))
            mod_nmi_std.append(np.std(mod_results[:, 2]))
            con_std.append(np.std(con_results[:, 0]))
            con_f1_std.append(np.std(con_results[:, 1]))
            con_nmi_std.append(np.std(con_results[:, 2]))


        ax.bar(x_axis_points, mod, width=bar_width, linewidth=0, facecolor=nature_colours[0], label=r'Mod')
        ax.errorbar(x_axis_points, mod, mod_std, ecolor=nature_colours[5], elinewidth=0.75, linewidth=0)

        ax.bar(x_axis_points + 1/6, mod_f1, width=bar_width, linewidth=0, facecolor=nature_colours[0], alpha=0.9, hatch="/////", label=r'Mod$\rightarrow$F1')
        ax.errorbar(x_axis_points + 1/6, mod_f1, mod_f1_std, ecolor=nature_colours[5], elinewidth=0.75, linewidth=0)

        ax.bar(x_axis_points + 1/3, mod_nmi, width=bar_width, linewidth=0, facecolor=nature_colours[0], alpha=0.7, hatch="\\\\\\\\\\",label=r'Mod$\rightarrow$NMI')
        ax.errorbar(x_axis_points + 1/3, mod_nmi, mod_nmi_std, ecolor=nature_colours[5], elinewidth=0.75, linewidth=0)

        ax.bar(x_axis_points + 1/2, con, width=bar_width, linewidth=0,  facecolor=nature_colours[2], label=r'Con')
        ax.errorbar(x_axis_points + 1/2, con, con_std, ecolor=nature_colours[5], elinewidth=0.75, linewidth=0)
        
        ax.bar(x_axis_points + 2/3, con_f1, width=bar_width, linewidth=0, facecolor=nature_colours[2],alpha=0.9, hatch="/////", label=r'Con$\rightarrow$F1')
        ax.errorbar(x_axis_points + 2/3, con_f1, con_f1_std, ecolor=nature_colours[5], elinewidth=0.75, linewidth=0)
        
        ax.bar(x_axis_points + 5/6, con_nmi, width=bar_width, linewidth=0, facecolor=nature_colours[2], alpha=0.7, hatch="\\\\\\\\\\", label=r'Con$\rightarrow$NMI')
        ax.errorbar(x_axis_points + 5/6, con_nmi, con_nmi_std, ecolor=nature_colours[5], elinewidth=0.75, linewidth=0)

        if "Large" not in title:
            ax.set_xticks(x_axis_points - 0.5 * bar_width)
            ax.set_xticklabels([name.upper() for name in x_axis_names], ha='left', rotation=-45, position=(-0.5, 0.0), fontsize=9)
        else:
            ax.set_xticks(x_axis_points + 0.4)
            ax.set_xticklabels([name.upper() for name in x_axis_names], fontsize=15)
        
        ax.set_axisbelow(True)
        ax.set_ylim(0.0, 1.0)

        if "Large" in title and i == 1:
            ax.legend(loc='best')
        elif "Synthetic" in title and i == 4: 
            ax.legend(loc='best')
        elif "Optimisation" in title and i == 5: 
            ax.legend(loc='best')
        elif i == 3 and "Optimisation" not in title:
            ax.legend(loc='best')

    if "Large" not in title:
        fig.suptitle(title, fontsize=20)
    else:
        fig.suptitle(title, fontsize=16) 
    save_name = f'{os.path.basename(os.path.normpath(folder))}'
    save_name = save_name.replace("\'", "")
    plt.tight_layout()
    plt.savefig(f'./figures/unsupervised_limit/{save_name}.eps', format='eps')

def calculate_framework_comparison_rank(datasets, algorithms, folder, default_algorithms, dfolder):
    # get results for both
    mod_results, con_results = extract_results(datasets, algorithms, folder)
    dmod_results, dcon_results = extract_results(datasets, default_algorithms, dfolder)
    dcon_results[:, 0] = 1 - dcon_results[:, 0]
    con_results[:, 0] = 1 - con_results[:, 0]
    result_object = np.concatenate((mod_results.flatten(), con_results.flatten()))
    default_result_object = np.concatenate((dmod_results.flatten(), dcon_results.flatten()))

    # rank them 
    n_comparisons = result_object.flatten().shape[0]
    rankings = np.zeros((n_comparisons, 2))
    for i in range(n_comparisons):
        if result_object[i] > default_result_object[i]:
            rankings[i] = [1, 2]
        elif default_result_object[i] < result_object[i]:
            rankings[i] = [2, 1]
        else:
            rankings[i] = [1.5, 1.5]
    
    # calculate the average
    means_hpo = np.mean(rankings, axis=0)[0]
    means_def = np.mean(rankings, axis=0)[1]
    print(f'HPO FCR: {means_hpo:.3f}')
    print(f'Default FCR: {means_def:.3f}')
    return 



if __name__ == "__main__":
    matplotlib.use("macosx")
    make_ugle = True
    make_big_figure = False
    make_dist_figure = True
    make_presentation_figures = False

    make_unsuper = False
    calc_increases = True


    if make_ugle:
        algorithms = ['daegc', 'dgi', 'dmon', 'grace', 'mvgrl', 'selfgnn', 'sublime', 'bgrl', 'vgaer', 'cagc']
        datasets = ['cora', 'citeseer', 'dblp', 'bat', 'eat', 'texas', 'wisc', 'cornell', 'uat', 'amac', 'amap']
        metrics = ['f1', 'nmi', 'modularity', 'conductance']
        folder = './results/legacy_results/progress_results/'
        seeds = [42, 24, 976, 12345, 98765, 7, 856, 90, 672, 785]
        default_algos = ['daegc_default', 'dgi_default', 'dmon_default', 'grace_default', 'mvgrl_default', 'selfgnn_default',
                        'sublime_default', 'bgrl_default', 'vgaer_default', 'cagc_default']
        default_folder = './results/legacy_results/default_results/'

        if make_presentation_figures: 
            create_rand_dist_comparison(['cora'], algorithms, metrics, seeds, folder, default_algos, default_folder)

            # create holder figure
            fig, ax = plt.subplots(1, 1, figsize=(20, 16))
            ax = create_result_bar_chart('cora', algorithms, folder, default_algos, default_folder, ax)

            ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
            for item in ax.get_legend().get_texts():
                item.set_fontsize(36)

            fig.tight_layout()
            fig.savefig(f"{ugle_path}/figures/hpo_investigation_presentation.png", format='png', bbox_inches='tight')
        else: 
            if make_big_figure:
                create_big_figure(datasets, algorithms, folder, default_algos, default_folder)

            # fetch absolute results
            result_object = make_test_performance_object(datasets, algorithms, metrics, seeds, folder)
            default_result_object = make_test_performance_object(datasets, default_algos, metrics, seeds, default_folder)

            # change conductance to be one minus so that it works with FCR
            con_out = np.array([1 - res if res != -10 else res for res in result_object[:, :, 3, :].flatten()])
            dcon_out = np.array([1 - res if res != -10 else res for res in default_result_object[:, :, 3, :].flatten()])

            # flatten object 
            result_object_fcr = np.concatenate((con_out, result_object[:, :, 0:3, :].flatten()))
            default_result_object_fcr = np.concatenate((dcon_out, default_result_object[:, :, 0:3, :].flatten()))
            
            # make comparisons
            n_comparisons = result_object_fcr.shape[0]
            rankings = np.zeros((n_comparisons, 2))
            result_object_fcr = result_object_fcr.flatten()
            default_result_object_fcr = default_result_object_fcr.flatten()
            for i in range(n_comparisons):
                if result_object_fcr[i] > default_result_object_fcr[i]:
                    rankings[i] = [1, 2]
                elif default_result_object_fcr[i] < result_object_fcr[i]:
                    rankings[i] = [2, 1]
                else:
                    rankings[i] = [1.5, 1.5]
            means_hpo = np.mean(rankings, axis=0)[0]
            means_def = np.mean(rankings, axis=0)[1]

            print(f'HPO FCR: {means_hpo:.3f}+_ {np.std(rankings, axis=0)[0]:.2f}')
            print(f'Default FCR: {means_def:.3f}+_ {np.std(rankings, axis=0)[1]:.2f}')

            # calculate ranking of each metric
            ranking_object = calculate_ranking_performance(result_object, datasets, metrics, seeds, calc_ave_first=False)
            default_ranking_object = calculate_ranking_performance(default_result_object, datasets, metrics, seeds, calc_ave_first=False)
            ranking_object = reshape_ranking_to_test_object(ranking_object)
            default_ranking_object = reshape_ranking_to_test_object(default_ranking_object)

            og_w = og_randomness(ranking_object)
            print(f"OG W HPO: {og_w:.3f}")
            og_w_def = og_randomness(default_ranking_object)
            print(f"OG W Default: {og_w_def:.3f}")

            result_object = np.concatenate((result_object[:, :, 0:3, :], con_out.reshape((result_object.shape[0], result_object.shape[1], 1, result_object.shape[3]))), axis=2)
            default_result_object = np.concatenate((default_result_object[:, :, 0:3, :], dcon_out.reshape((result_object.shape[0], result_object.shape[1], 1, result_object.shape[3]))), axis=2)
            result_object = reshape_ranking_to_test_object(result_object)
            default_result_object = reshape_ranking_to_test_object(default_result_object)

            ties_w = og_newOld_randomness(result_object)
            print(f"New ranking system, OLD W HPO: {ties_w:.3f}")
            ties_w_def = og_newOld_randomness(default_result_object)
            print(f"New ranking system, OLD W Default: {ties_w_def:.3f}")

            ties_w = ties_randomness(result_object)
            print(f"TIES W HPO: {ties_w:.3f}")
            ties_w_def = ties_randomness(default_result_object)
            print(f"TIES W Default: {ties_w_def:.3f}")

            ties_w = wasserstein_randomness(result_object)
            print(f"WW HPO: {ties_w:.3f}")
            ties_w_def =  wasserstein_randomness(default_result_object)
            print(f"WW Default: {ties_w_def:.3f}")

            random_rankings = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
            print(f'{random_rankings[:, 0]} -> all seeds, one algorithm')
            print(f'WW random: {w_rand_wasserstein(random_rankings)}')
            perfect_rankings = random_rankings.T
            print(f'WW perfect: {w_rand_wasserstein(perfect_rankings)}')

            #random_rankings = create_random_results(44, 10, 10)

            if make_dist_figure: 
                plt.style.use(['science', 'nature'])
                plt.rcParams["font.family"] = "Times New Roman"
                plt.rcParams["figure.dpi"] = 300
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 4))

                test_interval = 1
                n_tests = result_object.shape[0]
                n_repeats = 10
                input_idxs = range(0, n_tests)
                titles = ['Original ' + r'$W$' + ' Randomness', r'$W$' + ' Randomness\nw/ Mean Ties', 'Tied ' + r'$W_t$' + ' Randomness', r'$W_w$' + ' Wasserstein Randomness']
                w_fns = ['og_randomness', 'og_newOld_randomness', 'ties_randomness', 'wasserstein_randomness']
                

                for a, ax in enumerate(axes.flat):
                    print(f'axis: {a}')
                    if w_fns[a] == 'og_randomness':
                        temp_object = ranking_object
                        default_temp_object = default_ranking_object
                    else:
                        temp_object = result_object
                        default_temp_object = default_result_object
                    for test_interval in range(1, n_tests):
                        for i in range(n_repeats):
                            test_idx = random.sample(input_idxs, test_interval)

                            temp_rankings = temp_object[test_idx, :, :]
                            WW_HPO = locals()[w_fns[a]](temp_rankings)
                            ax.plot(test_interval, WW_HPO, 'x', c='C4', markersize=2)

                            temp_rankings = default_temp_object[test_idx, :, :]
                            WW_DEF = locals()[w_fns[a]](temp_rankings)
                            ax.plot(test_interval, WW_DEF, 'x', c='C3', markersize=2)
                    
                    handles = []
                    handles.append(mlines.Line2D([], [], label='HPO', color="C4", marker='x', linestyle='None'))
                    handles.append(mlines.Line2D([], [], label='Default', color="C3", marker='x', linestyle='None'))

                    ax.set_xlabel('N Tests', fontsize=6)
                    ax.set_ylabel(r'$W$', fontsize=6)
                    ax.set_title(titles[a], fontsize=8)
                    ax.set_ylim(0, 0.75)
                    ax.legend(handles=handles, loc='best', fontsize=4)
                

                fig.tight_layout()
                fig.savefig(f"{ugle_path}/figures/w_distribution_entropy.eps", format='eps', bbox_inches='tight')



    if make_unsuper:
        q1_folder = './results/unsupervised_limit/default_q1/'
        q2_folder = './results/unsupervised_limit/hpo_q2/'
        qlarge_folder = './results/unsupervised_limit/hpo_large/'
        q5_folder = './results/unsupervised_limit/synth_default_q5/'
        q5_folder1 = './results/unsupervised_limit/33_default_q4/'
        q5_folder2 = './results/unsupervised_limit/66_default_q4/'

        seeds = [42, 24, 976, 12345, 98765, 7, 856, 90, 672, 785]
        datasets = ['citeseer', 'cora', 'texas', 'dblp', 'wisc', 'cornell']
        synth_datasets = ['synth_disjoint_disjoint_2', 'synth_disjoint_random_2', 'synth_disjoint_complete_2',
                            'synth_random_disjoint_2', 'synth_random_random_2', 'synth_random_complete_2',
                            'synth_complete_disjoint_2', 'synth_complete_random_2', 'synth_complete_complete_2']

        default_algorithms = ['daegc_default', 'dmon_default', 'grace_default', 'sublime_default', 'bgrl_default', 'vgaer_default']
        algorithms = ['dgi', 'daegc', 'dmon', 'grace', 'sublime', 'bgrl', 'vgaer']

        if calc_increases:
            # calculate the percentage drops
            f1_nmi_results = extract_supervised_results(datasets, default_algorithms, './results/unsupervised_limit/default_sup_select/')
            dmod_results, dcon_results = extract_results(datasets, default_algorithms, q1_folder)
            calc_percent_increase(f1_nmi_results, dmod_results, dcon_results)


        #calculate_framework_comparison_rank(datasets, algorithms, q2_folder, default_algorithms, q1_folder)

        #create_abs_performance_figure(datasets, algorithms, q2_folder, title="Hyperparameter Optimisation Performance", plot_dims=[2, 3], figsize=(8, 6))
        #create_abs_performance_figure(datasets, default_algorithms, q1_folder, title="Default Hyperparameter's Performance", plot_dims=[2, 3], figsize=(8, 6))
        #create_abs_performance_figure(['Computers', 'Photo'], ['dmon'], qlarge_folder, title="DMoN Performance Large Datasets with HPO", plot_dims=[1, 2], figsize=(5, 2.5))
        #create_abs_performance_figure(synth_datasets, default_algorithms, q5_folder, title="Default Hyperparameter's Performance on Synthetic Data", plot_dims=[3, 3], figsize=(9, 9))
        #create_abs_performance_figure(datasets, default_algorithms, q5_folder2, title="Default Hyperparameters with 66\% of Training Data", plot_dims=[2, 3], figsize=(8, 6))
        #create_abs_performance_figure(datasets, default_algorithms, q5_folder1, title="Default Hyperparameters with 33\% of Training Data", plot_dims=[2, 3], figsize=(8, 6))

        #unsupervised_prediction_graph(datasets, algorithms, q2_folder, title="Hyperparameter Optimisation Correlation")
        #unsupervised_prediction_graph(datasets, default_algorithms, q1_folder, title="Default Hyperparameter's Correlation")
        #unsupervised_prediction_graph(['Computers', 'Photo'], ['dmon'], qlarge_folder, title="Large Dataset HPO")
        #unsupervised_prediction_graph(datasets, default_algorithms, q5_folder2, title="q4: 66\% of the data")
        #unsupervised_prediction_graph(datasets, default_algorithms, q5_folder1, title="q4: 33\% of the data")
        
from omegaconf import OmegaConf, ListConfig
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
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from matplotlib.ticker import IndexLocator

random.seed(42)
np.random.seed(42)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": [
        r"\usepackage{amsmath}",]
        # r"\usepackage[utf8]{inputenc}",
        # r"\usepackage[svgnames]{xcolor}",
        # r"\usepackage{dashbox}",
        # r"\usepackage{mathtools, amssymb}",
        # r"\newcommand\dashedph[1][H]{\setlength{\fboxsep}{0pt}\setlength{\dashlength}{2.2pt}\setlength{\dashdash}{1.1pt} \dbox{\phantom{#1}}}"]
})

def mod_to_real(x):
    return ((2/3)*x) + (1/3)

def search_results(folder, filename):

    for root, dirs, files in os.walk(f'{ugle_path}/{folder}'):
        if filename in files:
            return os.path.join(root, filename)
    return None

def get_all_results_from_storage(datasets: list, algorithms: list, folder: str, search_first_folder: str, empty: str = 'minus_ten',
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
            file_found1 = search_results(search_first_folder, filename)
            file_found = search_results(folder, filename)
            if file_found or file_found1:
                if file_found1:
                    result = pickle.load(open(file_found1, "rb"))
                elif file_found: 
                    result = pickle.load(open(file_found, "rb"))
                if collect_all:
                    # parse seed
                    return_results = np.zeros(shape=(4, 10))
                    # go thru every seed
                    for i, seed_result in enumerate(result.results):
                        # go thru every best hps configuration
                        for metric_result in seed_result.study_output:
                            # go thru each best metric in configuration
                            if isinstance(metric_result.metrics, list) or isinstance(metric_result.metrics, ListConfig):
                                for metric in metric_result.metrics:
                                    # add result to correct place
                                    if metric == 'f1':
                                        return_results[0, i] = metric_result.results[metric]
                                    elif metric == 'nmi':
                                        return_results[1, i] = metric_result.results[metric]
                                    elif metric == 'modularity':
                                        return_results[2, i] = mod_to_real(metric_result.results[metric])
                                    elif metric == 'conductance':
                                        return_results[3, i] = 1. - metric_result.results[metric]
                            elif isinstance(metric_result.metrics, str):
                                metric = metric_result.metrics
                                if metric == 'f1':
                                    return_results[0, i] = metric_result.results[metric]
                                elif metric == 'nmi':
                                    return_results[1, i] = metric_result.results[metric]
                                elif metric == 'modularity':
                                    return_results[2, i] = mod_to_real(metric_result.results[metric])
                                elif metric == 'conductance':
                                    return_results[3, i] = 1. - metric_result.results[metric]

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


def make_test_performance_object(datasets, algorithms, metrics, seeds, folder, search_first_folder):
    # get results object
    result_object = np.zeros(shape=(len(datasets), len(algorithms), len(metrics), len(seeds)))
    try:
        result_holder = get_all_results_from_storage(datasets, algorithms, folder, search_first_folder, collect_all=True)
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
                ranking_of_algorithms = np.flip(np.argsort(metric_values)) + 1
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
                    ranking_of_algorithms = np.flip(np.argsort(metric_values)) + 1
                    ranking_of_algorithms[last_place_zero] = len(ranking_of_algorithms)
                    ranking_object[d, :, m, s] = ranking_of_algorithms
    
    return ranking_object


def create_result_bar_chart(dataset_name, algorithms, folder, default_algos, default_folder, ax=None, search_first_hpo=None, search_first_default=None, include_defaults=True):
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
    alt_colours = ["#2CA02C", '#BF699F', 'tab:red', 'tab:blue']
    error_colours = ["#66D166" , "#E89BCC" ,"#D66363" , "#68B8ED"]

    # extract key arrays for results

    ################## DIFFUCULT ##########################
    #### CHANGE TO AVERAGE FROM THE OTHER RESULT THING? 
    ### GET TEST OBJECT AND ACTUALLY DO THE AVERAGING?

    result_holder = get_all_results_from_storage([dataset_name], algorithms, folder, empty='zeros', search_first_folder=search_first_hpo)
    if include_defaults:
        default_result_holder = get_all_results_from_storage([dataset_name], default_algos, default_folder, empty='zeros', search_first_folder=search_first_default)

    f1, f1_std = get_values_from_results_holder(result_holder, dataset_name, 'f1', return_std=True)
    nmi, nmi_std = get_values_from_results_holder(result_holder, dataset_name, 'nmi', return_std=True)
    modularity, modularity_std = get_values_from_results_holder(result_holder, dataset_name, 'modularity',
                                                                return_std=True)
    conductance, conductance_std = get_values_from_results_holder(result_holder, dataset_name, 'conductance',
                                                                  return_std=True)

    if include_defaults:
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

    colour_width = 1.5
    black_width = 1.
    e_width = 0.1

    # default_conductance = [1 - default_cond for default_cond in default_conductance ]
    # conductance = [1 - cond for cond in conductance]

    # plot hyperparameter results in full colour
    ax.bar(x_axis_names, f1, ecolor=alt_colours[0],
           width=bar_width, facecolor=alt_colours[0], alpha=1., linewidth=0, label='f1')
    ax.errorbar(x_axis_names, f1, f1_std, ecolor=error_colours[0], elinewidth=colour_width, linewidth=0)
    ax.bar(x_axis_names + bar_width, nmi, ecolor=alt_colours[1],
           width=bar_width, facecolor=alt_colours[1], alpha=1., linewidth=0, label='nmi')
    ax.errorbar(x_axis_names + bar_width, nmi, nmi_std, ecolor=error_colours[1], elinewidth=colour_width, linewidth=0)
    ax.bar(x_axis_names + (2 * bar_width), modularity, ecolor=alt_colours[2],
           width=bar_width, facecolor=alt_colours[2], alpha=1., linewidth=0, label='modularity')
    ax.errorbar(x_axis_names + (2 * bar_width), modularity, modularity_std, ecolor=error_colours[2], elinewidth=colour_width, linewidth=0)
    ax.bar(x_axis_names + (3 * bar_width), conductance, ecolor=alt_colours[3],
           width=bar_width, facecolor=alt_colours[3], alpha=1., linewidth=0, label='conductance')
    ax.errorbar(x_axis_names + (3 * bar_width), conductance, conductance_std, ecolor=error_colours[3], elinewidth=colour_width, linewidth=0)

    if include_defaults:
        # plot default parameters bars in dashed lines
        blank_colours = np.zeros(4)
        ax.bar(x_axis_names, default_f1, yerr=default_f1_std, width=bar_width,
            facecolor=blank_colours, edgecolor='black', linewidth=black_width, linestyle='--', label='default values')
        ax.errorbar(x_axis_names, default_f1, default_f1_std, ecolor='black', elinewidth=e_width, linewidth=e_width, linestyle='none')
        ax.bar(x_axis_names + bar_width, default_nmi, yerr=default_nmi_std, width=bar_width,
            facecolor=blank_colours, edgecolor='black', linewidth=black_width, linestyle='--')
        ax.errorbar(x_axis_names + bar_width, default_nmi, default_nmi_std, ecolor='black', elinewidth=e_width, linewidth=e_width, linestyle='none')
        ax.bar(x_axis_names + (2 * bar_width), default_modularity, yerr=default_modularity_std, width=bar_width, 
            facecolor=blank_colours, edgecolor='black', linewidth=black_width, linestyle='--')
        ax.errorbar(x_axis_names + (2 * bar_width), default_modularity, default_modularity_std, ecolor='black', elinewidth=e_width, linewidth=e_width, linestyle='none')
        ax.bar(x_axis_names + (3 * bar_width), default_conductance, yerr=default_conductance_std,
            facecolor=blank_colours, width=bar_width, edgecolor='black', linewidth=black_width, linestyle='--')
        ax.errorbar(x_axis_names + (3 * bar_width), default_conductance, default_conductance_std, ecolor='black', elinewidth=e_width, linewidth=e_width, linestyle='none')

    # create the tick labels for axis
    # ax.set_xticks(x_axis_names - 0.5 * bar_width)
    # algorithms = [algo.upper() if algo != 'dmon' else 'DMoN' for algo in algorithms]
    # if dataset_name == 'cora' or dataset_name == 'amap' or dataset_name == 'citeseer' or dataset_name == 'dblp' or dataset_name == 'bat' or dataset_name == 'texas':
    #     ax.set_xticklabels(algorithms,ha='left', position=(-0.15, 0))
    # else:
    #     ax.set_xticklabels(algorithms, ha='left', rotation=-45, position=(-0.3, 0))
    
    if dataset_name == 'cora' or dataset_name == 'amap' or dataset_name == 'citeseer' or dataset_name == 'dblp' or dataset_name == 'bat' or dataset_name == 'texas':
        ax.tick_params(which='major', length=0., color='black', axis='x', pad=7)
        ax.xaxis.set_major_locator(IndexLocator(base=1, offset=0.5))
        ax.set_xticklabels([algo.upper() if algo != 'dmon' else 'DMoN' for algo in algorithms], ha='center')

        # ax.tick_params(which='major', length=0., color='black', axis='x')
        ax.xaxis.set_major_locator(IndexLocator(base=1, offset=0.5))

        ax.tick_params(which='minor', length=7, color='black', width=1)
        ax.xaxis.set_minor_locator(IndexLocator(base=0.5, offset=0.))
    else:
        ax.set_xticks(x_axis_names - 0.5 * bar_width)
        ax.set_xticklabels([algo.upper() if algo != 'dmon' else 'DMoN' for algo in algorithms], ha='left', rotation=-45, position=(-0.3, 0))

    ax.set_axisbelow(True)

    # Axis styling.
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_color('#DDDDDD')
    # ax.yaxis.grid(True, color='#EEEEEE')
    # ax.xaxis.grid(False)

    # tighten the layout
    if dataset_name == 'citeseer':
        title_name = 'CiteSeer'
    elif dataset_name == 'texas' or dataset_name == 'cornell' or dataset_name == 'wisc' or dataset_name == 'cora':
        title_name = dataset_name.capitalize()
    else:
         title_name = dataset_name.upper()
    ax.set_title(title_name, fontsize=18)
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
    
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.0-(bar_width/2), len(algorithms)-(bar_width/2))
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
    # x_axis = np.arange(0, len(algorithms), 0.001)
    # max_y = 0
    possible_ranks = np.arange(1, len(algorithms) + 1)
    bar_width = 1 / len(algorithms)

    for alg, algo_ranks in enumerate(all_ranks_per_algo.T):

        rank_counts = [np.sum(algo_ranks == rank) for rank in possible_ranks]
        ax.bar(possible_ranks + (alg * bar_width), rank_counts, width=bar_width, label=algorithms[alg], color=cm[alg])

    if set_legend:
        #ax.legend(loc='best', fontsize=20, ncol=1, bbox_to_anchor=(1, -0.5))
        #ax.legend(loc='upper center', fontsize=15, ncol=3, bbox_to_anchor=(0.475, -0.5))
        ax.set_xlabel('Algorithm Rank ' + r'$(r)$', fontsize=16)
        #ax.set_xlabel('algorithm rank distribution over all tests', fontsize=18)

    #ax.set_ybound(0, 3)
    # Set major ticks at 0.5, 1.5, 2.5, ..., 7.5 and label them as 1, 2, 3, ..., 7
    major_ticks = [(i+0.5) - (bar_width/2) for i in range(1, len(algorithms)+1)] # 0.5, 1.5, 2.5, ..., 7.5
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(range(1, len(algorithms)+1)) # Labels 1, 2, 3, ..., 7

    # Set minor ticks at 0, 1, 2, 3, ..., 7 with no labels
    minor_ticks = [i * 1 - (bar_width/2) for i in range(1, len(algorithms)+2)] # 0, 1, 2, 3, ..., 7
    ax.set_xticks(minor_ticks, minor=True)

    # Optionally, show gridlines for minor ticks
    ax.grid(True, which='minor', axis='x', linestyle='--', color='lightgrey', linewidth=0.5)
   
    ax.set_ylabel('Rank Occurences', fontsize=16) #kde estimatation of rank distribution
    ax.set_xlim(1-(bar_width/2), len(algorithms)+1-(bar_width/2))
    ax.set_yticks(range(0, all_ranks_per_algo.shape[0]+1))

    ax.set_ylim(0, 10)
    #ax.text(0.4, 0.85, ave_overlap_text, fontsize=20, transform=ax.transAxes, zorder=1000)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    return ax


def create_big_figure(datasets, algorithms, folder, default_algos, default_folder, search_first_hpo, search_first_default):
    """
    creates figure for all datasets tested comparing default and hpo results
    """
    # create holder figure
    fig = plt.figure(figsize=(9, 10))
    axs = []

    if 'uadgn' in algorithms:
        include_defaults = False
    else:
        include_defaults = True

    # Define GridSpec: 2 rows and 2 columns for more precise control
    gs = gridspec.GridSpec(2, 2)
    axs.append(fig.add_subplot(gs[0, 0:2]))  
    if 'cora' in datasets or 'amap' in datasets:
        axs.append(fig.add_subplot(gs[1, 0:2]))
    else:
        axs.append(fig.add_subplot(gs[1, 0]))  # First plot spans columns 
        axs.append(fig.add_subplot(gs[1, 1]))  # Second plot spans columns 

    for dataset_name, ax in zip(datasets, axs):
        ax = create_result_bar_chart(dataset_name, algorithms, folder, default_algos, default_folder, ax, search_first_hpo, search_first_default, include_defaults)

    handles = []
    alt_colours = ["#2CA02C", '#BF699F', 'tab:red', 'tab:blue']
    metrics = ['F1', 'NMI', r'$\mathcal{M}$', r'$\mathcal{C}$']
    fig.tight_layout()
    for i in range(len(alt_colours)):
        handles.append(mlines.Line2D([], [], color=alt_colours[i], linewidth=8, label=metrics[i]))

    if include_defaults:

        class LabeledObject(object):
            def __init__(self, label):
                self.label = label
            
            def get_label(self):
                return self.label

        class CustomRectangleHandler(object):
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                patch = mpatches.Rectangle([x0, y0], width, height, facecolor='none',
                                        edgecolor='black', linestyle='--', lw=1,
                                        transform=handlebox.get_transform())
                handlebox.add_artist(patch)
                return patch

        custom_rect = LabeledObject('Default\nHyperparameters')
        handles.append(custom_rect)


        # handles.append(mlines.Line2D([], [], color=[0,0,0,0], linewidth=1, markeredgewidth=1.5, marker="s", markersize=10, linestyle=':', markeredgecolor='black', label='Default\nHyperparameters'))

    blank_ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    blank_ax.axis('off')
    #blank_ax.legend(handles=handles, bbox_to_anchor=(0.975, 0.2), fontsize=14)
    blank_ax.legend(handles=handles, loc='lower center', fontsize=14, ncols=5, handler_map={LabeledObject: CustomRectangleHandler()})
    fig.subplots_adjust(bottom=0.15)

    if 'cora' in datasets or 'amap' in datasets:
        fig.subplots_adjust(bottom=0.10, hspace=0.17, top=0.97)

    #axs[0].legend(loc='upper right', bbox_to_anchor=(1, 0.95))
    #for item in axs[0].get_legend().get_texts():
    #   item.set_fontsize(36)

    if 'cora' in datasets:
        fig.savefig(f"{ugle_path}/figures/postvivathesis/hpo_investigation_1.pdf", bbox_inches='tight')
    elif 'amap' in datasets:
        fig.savefig(f"{ugle_path}/figures/postvivathesis/hpo_investigation_2.pdf", bbox_inches='tight')
    elif 'texas' in datasets:
        fig.savefig(f"{ugle_path}/figures/postvivathesis/hpo_investigation_3.pdf", bbox_inches='tight')
    elif 'eat' in datasets:
        fig.savefig(f"{ugle_path}/figures/postvivathesis/hpo_investigation_4.pdf", bbox_inches='tight')
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


def create_rand_dist_comparison(datasets: list, algorithms: list, metrics: list, seeds: list, folder: str, default_algos: list, default_folder: str, search_first_post_viva, search_first_post_viva_default):

    # create holder figure
    nrows, ncols = 2, 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(9, 6))

    # i think the answer is that you gotta try do the load in all cases there's a load with the both and overwrite where not minus 10 
    # with the old results?
    # maybe go thru all the bits that need to be changed here 

    # 1. check every instance where results are loaded 
    # if using the same function the just change range 0-1 in the load results
    # make a note of real folder that needs to be used 

    ################# ========== LOADED OBJECT ========== ###################
    result_object = make_test_performance_object(datasets, algorithms, metrics, seeds, folder, search_first_post_viva)

    ################# ========== LOADED OBJECT ========== ###################
    default_result_object = make_test_performance_object(datasets, default_algos, metrics, seeds, default_folder,  search_first_post_viva_default)
    
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

    titles_0 = 'Default' 
    titles_1 = 'HPO'
    print(r'$\mathcal{W}$: ' + str(round(default_w, 3)))
    print(r'$\mathcal{W}$: ' + str(round(hpo_w, 3)))

    ax[0] = create_rand_dist_fig(ax[0], algorithms, default_ranking_object, set_legend=False)
    ax[1] = create_rand_dist_fig(ax[1], algorithms, ranking_object, set_legend=True)

    result_object = result_object[0, :, 0, :].flatten()
    default_result_object = default_result_object[0, :, 0, :].flatten()
    n_comparisons = result_object.shape[0]
    rankings = np.zeros((n_comparisons, 2))

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

    print('   ' + r'$\mathcal{R}_{\text{def}}$' + f': {means_def:.3f}')
    print('   ' + r'$\mathcal{R}_{\text{hpo}}$' + f': {means_hpo:.3f}')
    ax[0].set_title(titles_0, fontsize=20)
    ax[1].set_title(titles_1, fontsize=20)

    fig.tight_layout()
    ## add figure bit at the end 
    cm = plt.get_cmap('tab10').colors
    handles = [] 
    for i, algo_name in enumerate(algorithms):
        algo_name = algo_name.upper()
        if algo_name == 'DMON':
            algo_name = 'DMoN'
        handles.append(mlines.Line2D([], [], color=cm[i], linewidth=3, label=algo_name))
    blank_ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    blank_ax.axis('off')
    blank_ax.legend(handles=handles, loc='lower center', fontsize=12, ncols=4)
    fig.subplots_adjust(bottom=0.215, top=0.95)
    #fig.suptitle('Algorithm F1 Score Rank Distribution\n Estimation Comparison on Cora', fontsize=24)
    fig.savefig(f'{ugle_path}/figures/postvivathesis/le_rand_dist_comparison.pdf', bbox_inches='tight')
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

def reshape_ranking_to_test_object(ranking_object):
    # datasets, algorithms, metrics, seeds ->  tests(datasets+metrics), seeds, algorithms
    ranking_object = np.transpose(ranking_object, axes=(0, 2, 3, 1))
    ranking_object = ranking_object.reshape((-1,) + ranking_object.shape[2:])
    return ranking_object

def og_randomness(ranking_object, print_draws=False):
    # og W coefficient where draws are the lowest rank that would occur from the ties
    n_draws = 0
    wills_order = []
    for test in ranking_object:
        for rs, rs_test in enumerate(test):
            unique_scores, counts = np.unique(rs_test, return_counts=True)
            if len(unique_scores) != ranking_object.shape[2]:
                n_draws += ranking_object.shape[2] - 1 - len(unique_scores)
        wills_order.append(kendall_w(test))
    wills_order = np.array(wills_order)
    if print_draws:
        print(f'n_draws: {n_draws}')

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

def extract_results(datasets, algorithms, folder, sfirst_folder, extract_validation=False, return_df=False):
    # modularity and conductance may have different hyperparameters or model selection points 
    mod_results = []
    con_results = []
    columns = ['Dataset', 'Algorithm', 'Seed', 'A_Metric', 'A_Metric_Value', 'B_Metric', 'B_Metric_Value']
    df = pd.DataFrame(columns=columns)

    for dataset in datasets:
        for algo in algorithms:
            filename = f"{dataset}_{algo}.pkl"
            file_found1 = search_results(sfirst_folder, filename)
            file_found = search_results(folder, filename)
            if file_found or file_found1:
                if file_found1:
                    result = pickle.load(open(file_found1, "rb"))
                    print(f'search first found: {filename}')
                elif file_found: 
                    result = pickle.load(open(file_found, "rb"))
                    print(f'old result: {filename}')

                # if filename == 'cora_sublime.pkl':
                #     result = pickle.load(open(file_found, "rb"))
                    
                for seed_result in result.results:
                    ################# ========== CHANGE BECOS STRING? ========== ###################
                    # print(f'{seed_result.seed}')
                    for metric_result in seed_result.study_output:

                        if 'modularity' in metric_result.metrics or 'modularity' == metric_result.metrics:
                            # print('Mod')
                            if extract_validation: 
                                mod = metric_result.validation_results['modularity']['modularity']
                            else: 
                                mod = metric_result.results['modularity']
                            mod = mod_to_real(mod)
                            f1 = metric_result.results['f1']
                            nmi = metric_result.results['nmi']
                            mod_results.append([mod, f1, nmi])
                            df.loc[len(df)] = {'Dataset': dataset, 'Algorithm': algo, 'Seed': seed_result.seed, 'A_Metric': 'Modularity', 'A_Metric_Value': mod, 'B_Metric': 'F1', 'B_Metric_Value': f1}
                            df.loc[len(df)] = {'Dataset': dataset, 'Algorithm': algo, 'Seed': seed_result.seed, 'A_Metric': 'Modularity', 'A_Metric_Value': mod, 'B_Metric': 'NMI', 'B_Metric_Value': nmi}


                        if 'conductance' in metric_result.metrics or 'conductance' == metric_result.metrics:
                            # print('Con')
                            if extract_validation: 
                                con = metric_result.validation_results['conductance']['conductance']
                            else:
                                con = metric_result.results['conductance']
                            con = 1 - con
                            f1 = metric_result.results['f1']
                            nmi = metric_result.results['nmi']
                            con_results.append([con, f1, nmi])
                            df.loc[len(df)] = {'Dataset': dataset, 'Algorithm': algo, 'Seed': seed_result.seed, 'A_Metric': 'Conductance', 'A_Metric_Value': con, 'B_Metric': 'F1', 'B_Metric_Value': f1}
                            df.loc[len(df)] = {'Dataset': dataset, 'Algorithm': algo, 'Seed': seed_result.seed, 'A_Metric': 'Conductance', 'A_Metric_Value': con, 'B_Metric': 'NMI', 'B_Metric_Value': nmi}
            else:
                print(f"did not find: {filename}")
                # can only really do this because i know that there's 10 seeds
                for seed in range(10):
                    ################# ========== CHANGE TO MINUS 10 ========== ###################
                    mod_results.append([0., 0., 0.])
                    con_results.append([0., 0., 0.])


    mod_results = np.asarray(mod_results)
    con_results = np.asarray(con_results)
    if return_df:
        return mod_results, con_results, df
    else: 
        return mod_results, con_results

def extract_supervised_results(datasets, algorithms, folder, sfirst_folder):
    ################# ========== CHANGE TO MINUS 10 ========== ###################
    f1_nmi_results = np.zeros((len(datasets)*len(algorithms)*10, 2))
    i = 0
    
    for dataset in datasets:
        for algo in algorithms:
            filename = f"{dataset}_{algo}.pkl"
            file_found1 = search_results(sfirst_folder, filename)
            file_found = search_results(folder, filename)
            if file_found or file_found1:
                if file_found1:
                    result = pickle.load(open(file_found1, "rb"))
                    print(f'search first found: {filename}')
                elif file_found: 
                    result = pickle.load(open(file_found, "rb"))
                    print(f'old result: {filename}')
            
                for seed_result in result.results:
                    for metric_result in seed_result.study_output:
                        ################# ========== CHANGE BECOS STRING? ========== ###################
                        if 'f1' in metric_result.metrics or 'f1' == metric_result.metrics:
                            f1_nmi_results[i, 0] = metric_result.results['f1']
                        if 'nmi' in metric_result.metrics or 'nmi' == metric_result.metrics:
                            f1_nmi_results[i, 1] = metric_result.results['nmi']
                    i += 1
            else:
                print(f"did not find: {filename}")
    return f1_nmi_results

def calc_percent_increase(f1_nmi_results, dmod_results, dcon_results):

    # diff = np.zeros((f1_nmi_results.shape[0], 4))
    # for i, row in enumerate(f1_nmi_results):
    #     # f1 - modf1 
    #     diff[i, 0] = (row[0] - dmod_results[i, 1]) / dmod_results[i, 1]

    #     # f1 - conf1 
    #     diff[i, 1] = (row[0] - dcon_results[i, 1]) / dcon_results[i, 1]

    #     # nmi - modnmi
    #     diff[i, 2] = (row[1] - dmod_results[i, 2]) / dmod_results[i, 2]

    #     # nmi - connmi 
    #     diff[i, 3] = (row[1] - dcon_results[i, 2]) / dcon_results[i, 2]

    # increases = np.mean(diff, axis=0)

    # if np.isnan(increases).any() or np.isinf(increases).any():
    #     print('error?')

    mod_f1 = np.mean(dmod_results[:, 1] - f1_nmi_results[:, 0])
    con_f1 = np.mean(dcon_results[:, 1] - f1_nmi_results[:, 0])
    mod_nmi = np.mean(dmod_results[:, 2] - f1_nmi_results[:, 1])
    con_nmi = np.mean(dcon_results[:, 2] -  f1_nmi_results[:, 1])

    print(f'Abs Difference from using Modularity to select for F1 compared to just F1: {mod_f1:.2f}')
    print(f'Abs Difference from using Conductance to select for F1 compared to just F1: {con_f1:.2f}')
    print(f'Abs Difference from using Modularity to select for NMI compared to just NMI: {mod_nmi:.2f}')
    print(f'Abs Difference from using Conductance to select for NMI compared to just NMI: {con_nmi:.2f}')
    return

def print_dataset_table(datasets, algorithms, folder, sfirst_folder, power_d=2):

    result_return = np.zeros((len(datasets), 4))
    # extract results
    for d, dataset in enumerate(datasets):
        mod_results, con_results = extract_results([dataset], algorithms, folder, sfirst_folder)
        print(dataset, end = ' ')
        print(f'mod_f1_nmi: {np.mean(mod_results[:, 0]):.3f} & {np.mean(mod_results[:, 1]):.3f} & {np.mean(mod_results[:, 2]):.3f}', end=' ')
        print(f'con_f1_nmi: {np.mean(con_results[:, 0]):.3f} & {np.mean(con_results[:, 1]):.3f} & {np.mean(con_results[:, 2]):.3f}', end=' ')
        print(f'correlations: ', end=' ')
        pos_idx = 0 
        for x, y in [[mod_results[:, 0], mod_results[:, 1]], [mod_results[:, 0], mod_results[:, 2]], [con_results[:, 0], con_results[:, 1]], [con_results[:, 0], con_results[:, 2]]]:
            coefficients = np.polyfit(x, y, power_d)
            poly = np.poly1d(coefficients)
            # Calculate predicted values
            predicted_y = poly(x)
            r_value_quad = np.round(r2_score(y, predicted_y), 3)
            predictor_varibables = 2
            r_value_quad = 1 - (1-r_value_quad) * (len(y)-1)/(len(y)-predictor_varibables-1)
            print(f'& {r_value_quad:.2f}', end=' ')

            result_return[d, pos_idx] = np.round(r_value_quad, 2)
            pos_idx += 1
        print('')

    return result_return

def print_algo_table(datasets, algorithms, folder, sfirst_folder, power_d=2):

    result_return = np.zeros((len(algorithms), 4))


    for a, algorithm in enumerate(algorithms):
        mod_results, con_results = extract_results(datasets, [algorithm], folder, sfirst_folder)
        print(algorithm, end = ' ')
        print(f'mod_f1_nmi: {np.mean(mod_results[:, 0]):.3f} & {np.mean(mod_results[:, 1]):.3f} & {np.mean(mod_results[:, 2]):.3f}', end=' ')
        print(f'con_f1_nmi: {np.mean(con_results[:, 0]):.3f} & {np.mean(con_results[:, 1]):.3f} & {np.mean(con_results[:, 2]):.3f}', end=' ')
        print(f'correlations: ', end=' ')
        pos_idx = 0 
        for x, y in [[mod_results[:, 0], mod_results[:, 1]], [mod_results[:, 0], mod_results[:, 2]], [con_results[:, 0], con_results[:, 1]], [con_results[:, 0], con_results[:, 2]]]:
            coefficients = np.polyfit(x, y, power_d)
            poly = np.poly1d(coefficients)
            # Calculate predicted values
            predicted_y = poly(x)
            r_value_quad = np.round(r2_score(y, predicted_y), 3)
            predictor_varibables = 2
            r_value_quad = 1 - (1-r_value_quad) * (len(y)-1)/(len(y)-predictor_varibables-1)
            print(f'& {r_value_quad:.2f}', end=' ')

            result_return[a, pos_idx] = np.round(r_value_quad, 2)
            pos_idx += 1
        print('')

    return result_return 

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

    indices = np.argsort(scores)

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

def compute_w_order_for_mod_and_con(mod_results, con_results, testnames, algorithms, datasets):
    different_comparisons = [mod_results[:, 0], mod_results[:, 1], mod_results[:, 2], con_results[:, 0], con_results[:, 1], con_results[:, 2]]
    total_w_order = []

    for result, testname in zip(different_comparisons, testnames):
        tempresult = result.reshape(10, len(algorithms), len(datasets), order="F")
        tempresult = np.transpose(tempresult, axes=(2, 0, 1))
        ranking = np.zeros_like(tempresult)
        for d, dataset_res in enumerate(tempresult):
            for s, seed_res in enumerate(dataset_res):
                ranking[d, s, :] = rank_values(seed_res)

        w_order = []
        for test in ranking:
            w_order.append(kendall_w(test))
        w_order = np.array(w_order)
        total_w_order.append(w_order)
        # the W order of each metric tests
        print(f"{testname} W_m Order: {np.mean(w_order):.3f}")
    # the W order over the whole 66% experiment
    total_w_order = np.asarray(total_w_order)
    
    return total_w_order

def unsupervised_prediction_graph(datasets, algorithms, folder, sfirst_folder, title, pltlegend=False):
    #plt.style.use(['science', 'nature'])
    nature_colours = ["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"]
    algorithm_colors = ["#434982", "#00B945", "#EA907A", "#845B97", "#4F8A8B", "#FFCB74", "#B5DEFF"]
    # extract results 
    if "q4" in title:
        extract_validation = True
    else: 
        extract_validation = False
    ################# ========== LOADED OBJECT ========== ###################
    mod_results, con_results, df = extract_results(datasets, algorithms, folder, sfirst_folder, extract_validation=extract_validation, return_df=True)

    testnames = ["Modularity", "Modularity F1", "Modularity NMI", "Conductance", "Conductance F1", "Conductance NMI"]

    total_w_order = compute_w_order_for_mod_and_con(mod_results, con_results, testnames, algorithms, datasets)
    print(f"Overall W Order: {np.mean(total_w_order):.3f} +- {np.std(total_w_order):.3f}")
    # the W order of each comparison 
    indv_tests = np.array([[0, 1], [0, 2], [3, 4], [3, 5]])
    moretestnames = [testnames[i] for i in indv_tests[:, 1]]
    more_w_orders = [np.mean(np.asarray(total_w_order[i])) for i in indv_tests]
    for n, w in zip(moretestnames, more_w_orders):
        print(f"LOOky lOOKy man ->> W Order {n}: {w:.2f}")
   
    nrows, ncols = 2, 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 10))

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
        x_space = np.linspace(np.min(x), np.max(x), 200)

        # Fit a linear line 
        coefficients = np.polyfit(x, y, 1)
        poly = np.poly1d(coefficients)
        # Calculate predicted values
        predicted_y = poly(x)
        r_value_line = np.round(r2_score(y, predicted_y), 3)
        predictor_varibables = 1
        r_value_line = 1 - (1-r_value_line) * (len(y)-1)/(len(y)-predictor_varibables-1)
        print(f"LOOky lOOKy man ->> Linear Adjusted (R^2): {r_value_line:.2f}")
        y_line = poly(x_space)
        
        # Fit a quadratic line
        coefficients = np.polyfit(x, y, 2)
        poly = np.poly1d(coefficients)
        # Calculate predicted values
        predicted_y = poly(x)
        r_value_quad = np.round(r2_score(y, predicted_y), 3)
        predictor_varibables = 2
        r_value_quad = 1 - (1-r_value_quad) * (len(y)-1)/(len(y)-predictor_varibables-1)
        print(f"LOOky lOOKy man ->> Quadratic Adjusted (R^2): {r_value_quad:.2f}")
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
            if datasets[opt_in] == 'dblp':
                dname = datasets[opt_in].upper()
            elif datasets[opt_in] == 'citeseer':
                dname = 'CiteSeer'
            else:
                dname = datasets[opt_in].capitalize()
            handles.append(mlines.Line2D([], [], label=dname, color="#474747", marker=markers[opt_in], linestyle='None', markersize=7))

            for a, col in enumerate(algorithm_colors):
                if "_" in algorithms[a]:
                    algo_name = algorithms[a].split("_")[0]
                else:
                    algo_name = algorithms[a]
                if opt_in == 5:
                    if algo_name == 'dmon':
                        handles.append(mlines.Line2D([], [], label='DMoN', color=col, marker="s", linestyle='None', markersize=7))
                    else:
                        handles.append(mlines.Line2D([], [], label=algo_name.upper(), color=col, marker="s", linestyle='None', markersize=7))

                ax.scatter(df_axis[df_axis['Algorithm'] == algorithms[a]]['A_Metric_Value'], df_axis[df_axis['Algorithm'] == algorithms[a]]['B_Metric_Value'], color=col, s=10, marker=markers[opt_in])

        handles.append(mlines.Line2D([], [], color=nature_colours[3], linestyle='dashed', linewidth=3, label='Quadratic Fit'))#, $R^2$' + f': {r_value_quad:.2f}'))
        handles.append(mlines.Line2D([], [], color=nature_colours[3], linewidth=3, label='Linear Fit'))
        ax.plot(x_space, y_line, color=nature_colours[3], linewidth=2)#, label=r'Linear Fit', $R^2$' + f': {r_value_line:.2f}')
        ax.plot(x_space, y_line_quad, color=nature_colours[3], linestyle='dashed', linewidth=2)#, label=r'Quadratic Fit')#, $R^2$' + f': {r_value_quad:.2f}', linewidth=2)
        
        if x_label == "Conductance":
            x_label = r"$\mathcal{C}$"
        if x_label == "Modularity":
            x_label = r"$\mathcal{M}$"
        if y_label == "NMI":
            y_label = "NMI"
        if y_label == "F1":
            y_label = "F1"
        
        if i == 0:
            ax.set_ylabel(y_label, fontsize=14)
        if i == 2:
            ax.set_ylabel(y_label, fontsize=14) 
            ax.set_xlabel(x_label, fontsize=14)
        if i == 3:
            ax.set_xlabel(x_label, fontsize=14)
        
        if i == 0 or i == 1:
            ax.set_ylim(0, 0.8)
        else:
            ax.set_ylim(0, 0.6)
        ax.set_title(x_label + r' $\rightarrow$ '+ y_label, fontsize=14)# + " (l-"+ r"$R^2$" + f": {r_value_line:.2f}, q-" + r"$R^2$"  + f": {r_value_quad:.2f}, " + r"$W$" + f": {W_order:.2f})", fontsize=9)
        #if i == 1:
        #    ax.legend(handles=handles, bbox_to_anchor=(1.10, 1.5), fontsize=8, ncols=1)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)
       
    print_algo_table(datasets, algorithms, folder, sfirst_folder)
    print_dataset_table(datasets, algorithms, folder, sfirst_folder)
    plt.tight_layout()
    if "Large" not in title:
        fig.suptitle(title, fontsize=18)
        plt.subplots_adjust(top=0.92, bottom=0.13)
        if pltlegend:
            # plt.subplots_adjust(top=0.85, bottom=0.3, hspace=0.42)
            # fig.add_subplot(111, frameon=False)
            # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            # plt.legend(handles=handles, bbox_to_anchor=(0, 0), fontsize=8, ncols=6)
            
            blank_ax = fig.add_axes([0, 0, 1, 1], frameon=False)
            blank_ax.axis('off')
            blank_ax.legend(handles=handles, loc='lower center', fontsize=10, ncols=5)
  
        if "66" in title:
            title_name = '66_data'
        elif "33" in title:
            title_name  = '33_data'
        else:
            title_name = title.replace(" ", "_")
            title_name = title_name.replace("\'", "")

        plt.savefig(f'./figures/postvivathesis/{title_name}.pdf')
    if pltlegend:
        return handles

def create_abs_performance_figure(datasets, algorithms, folder, sfirst_folder, title, plot_dims, figsize, plot_legend=False, default_algos=None, default_folder=None, default_sfirst_folder=None):
    plt.rcParams["hatch.linewidth"] = 0.3

    nature_colours = ["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"]
    my_colours2 = ["#FF9500", "gold", "#FF2C00", "#0C5DA5", "#B5DEFF", "#845B97"]
    my_colours1 = ['tab:red', 'tab:purple', 'tab:pink', 'tab:blue', 'tab:cyan', 'deepskyblue']
    my_colours = ['tab:red', 'darkorange', '#FDC010', 'tab:blue', 'tab:purple', '#78C7FF']
    error_colours = ["#D66363", "#F0A360", "#FCD668", "#68B8ED", "#C69DEB", "paleturquoise"]


    nrows, ncols = plot_dims[0], plot_dims[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    x_axis_names = [algo.split("_")[0] for algo in algorithms]
    x_axis_points = np.arange(len(x_axis_names))
    bar_width = 1/6

    if default_algos != None:
        include_defaults = True
    else:
        include_defaults = False

    for i, ax in enumerate(axes.flat):
        dataset = datasets[i]
        print(f'\n\n{dataset}')
        ################# ========== LOADED OBJECT ========== ###################
        mod_results, con_results = extract_results([dataset], algorithms, folder, sfirst_folder)
        if include_defaults:
            def_mod_results, def_con_results = extract_results([dataset], default_algos, default_folder, default_sfirst_folder)


        testnames = ["Modularity", "Modularity F1", "Modularity NMI", "Conductance", "Conductance F1", "Conductance NMI"]
        total_w_order = np.mean(compute_w_order_for_mod_and_con(mod_results, con_results, testnames, algorithms, [dataset]))
        if include_defaults:
            def_total_w_order = np.mean(compute_w_order_for_mod_and_con(def_mod_results, def_con_results, testnames, default_algos, [dataset]))

        if not include_defaults:
            if dataset == 'dblp':
                ax.set_title("DBLP" + ' (' + r'$\mathcal{W}^{m}$' + f': {total_w_order:.2f})', fontsize=15, y=1.01)
            elif 'synth' in dataset:
                dataset_name = r"$\textbf{A}$" + ": " + dataset.split("_")[1] + '  ' + r"$\textbf{X}$" ": " + dataset.split("_")[2]
                dataset_name = dataset_name.replace("disjoint", "Distinct")
                dataset_name = dataset_name.replace("random", "Random")
                dataset_name = dataset_name.replace("complete", "Null")
                ax.set_title(dataset_name + ' (' + r'$\mathcal{W}^{m}$' + f': {total_w_order:.2f})', fontsize=12, y=1.01)
            elif dataset == 'citeseer':
                ax.set_title('CiteSeer' + ' (' + r'$\mathcal{W}^{m}$' + f': {total_w_order:.2f})', fontsize=15, y=1.01)
            else:
                ax.set_title(dataset.capitalize() + ' (' + r'$\mathcal{W}^{m}$' + f': {total_w_order:.2f})', fontsize=15, y=1.01)
        else:
            if dataset == 'dblp':
                ax.set_title("DBLP" + ' (' + r'$\mathcal{W}^{m}_{hpo}$' + f': {total_w_order:.2f}, ' + r'$\mathcal{W}^{m}_{def}$' + f': {def_total_w_order:.2f}' + ')', fontsize=15, y=1.01)
            elif dataset == 'citeseer':
                ax.set_title('CiteSeer' + ' (' + r'$\mathcal{W}^{m}_{hpo}$' + f': {total_w_order:.2f}, ' + r'$\mathcal{W}^{m}_{def}$' + f': {def_total_w_order:.2f}' + ')', fontsize=15, y=1.01)
            else:
                ax.set_title(dataset.capitalize() + ' (' + r'$\mathcal{W}^{m}_{hpo}$' + f': {total_w_order:.2f}, ' + r'$\mathcal{W}^{m}_{def}$' + f': {def_total_w_order:.2f}' + ')', fontsize=15, y=1.01)


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
            ################# ========== LOADED OBJECT ========== ###################
            mod_results, con_results = extract_results([dataset], [algo], folder, sfirst_folder)


            # con_results[:, 0] = [1 - con_res for con_res in con_results[:, 0]]
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


        ax.bar(x_axis_points, mod, width=bar_width, linewidth=0, facecolor=my_colours[0])
        ax.errorbar(x_axis_points, mod, mod_std, ecolor=error_colours[0], elinewidth=1.5, linewidth=0)

        ax.bar(x_axis_points + 1/6, mod_f1, width=bar_width, linewidth=0, facecolor=my_colours[1])
        ax.errorbar(x_axis_points + 1/6, mod_f1, mod_f1_std, ecolor=error_colours[1], elinewidth=1.5, linewidth=0)

        ax.bar(x_axis_points + 1/3, mod_nmi, width=bar_width, linewidth=0, facecolor=my_colours[2])
        ax.errorbar(x_axis_points + 1/3, mod_nmi, mod_nmi_std, ecolor=error_colours[2], elinewidth=1.5, linewidth=0)

        ax.bar(x_axis_points + 1/2, con, width=bar_width, linewidth=0,  facecolor=my_colours[3])
        ax.errorbar(x_axis_points + 1/2, con, con_std, ecolor=error_colours[3], elinewidth=1.5, linewidth=0)
        
        ax.bar(x_axis_points + 2/3, con_f1, width=bar_width, linewidth=0, facecolor=my_colours[4])
        ax.errorbar(x_axis_points + 2/3, con_f1, con_f1_std, ecolor=error_colours[4], elinewidth=1.5, linewidth=0)
        
        ax.bar(x_axis_points + 5/6, con_nmi, width=bar_width, linewidth=0, facecolor=my_colours[5])
        ax.errorbar(x_axis_points + 5/6, con_nmi, con_nmi_std, ecolor=error_colours[5], elinewidth=1.5, linewidth=0)


        if include_defaults:
            blank_colours = np.zeros(4)
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

            for algo in default_algos:
                ################# ========== LOADED OBJECT ========== ###################
                mod_results, con_results = extract_results([dataset], [algo], default_folder, default_sfirst_folder)


                # con_results[:, 0] = [1 - con_res for con_res in con_results[:, 0]]
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


            ax.bar(x_axis_points, mod, width=bar_width, linewidth=1., facecolor=blank_colours, linestyle='--', edgecolor='black')
            ax.errorbar(x_axis_points, mod, mod_std, elinewidth=1., ecolor='black', linewidth=0.)

            ax.bar(x_axis_points + 1/6, mod_f1, width=bar_width, linewidth=1., facecolor=blank_colours, linestyle='--', edgecolor='black')
            ax.errorbar(x_axis_points + 1/6, mod_f1, mod_f1_std, elinewidth=1., ecolor='black', linewidth=0.)

            ax.bar(x_axis_points + 1/3, mod_nmi, width=bar_width, linewidth=1., facecolor=blank_colours, linestyle='--', edgecolor='black')
            ax.errorbar(x_axis_points + 1/3, mod_nmi, mod_nmi_std, elinewidth=1., ecolor='black', linewidth=0.)

            ax.bar(x_axis_points + 1/2, con, width=bar_width, linewidth=1.,  facecolor=blank_colours, linestyle='--', edgecolor='black')
            ax.errorbar(x_axis_points + 1/2, con, con_std, elinewidth=1., ecolor='black', linewidth=0.)
            
            ax.bar(x_axis_points + 2/3, con_f1, width=bar_width, linewidth=1., facecolor=blank_colours, linestyle='--', edgecolor='black')
            ax.errorbar(x_axis_points + 2/3, con_f1, con_f1_std, elinewidth=1., ecolor='black', linewidth=0.)
            
            ax.bar(x_axis_points + 5/6, con_nmi, width=bar_width, linewidth=1., facecolor=blank_colours, linestyle='--', edgecolor='black')
            ax.errorbar(x_axis_points + 5/6, con_nmi, con_nmi_std, elinewidth=1., ecolor='black', linewidth=0.)


        if "Large" not in title:
            # ax.set_xticks(x_axis_points - 0.5 * bar_width)

            ax.tick_params(which='major', length=0., color='black', axis='x', pad=7)
            ax.xaxis.set_major_locator(IndexLocator(base=1, offset=0.5))
            ax.set_xticklabels([name.upper() if name != 'dmon' else 'DMoN' for name in x_axis_names], ha='center', fontsize=12)

            # ax.tick_params(which='major', length=0., color='black', axis='x')
            ax.xaxis.set_major_locator(IndexLocator(base=1, offset=0.5))

            ax.tick_params(which='minor', length=7, color='black', width=1)
            ax.xaxis.set_minor_locator(IndexLocator(base=0.5, offset=0.))


 
            # # Shift the labels by +0.5 on the x-axis
            # #  Get current tick positions and labels
            # tick_positions = ax.get_xticks()
            # tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
            # # Shift the tick labels by +0.5 on the x-axis
            # new_tick_positions = tick_positions + 0.5
            # # Set the ticks and new positions, reusing the original labels
            # for mtl, myticklabel in enumerate(tick_labels):
            #     ax.set_xticklabels(tick_labels, position=(new_tick_positions[mtl], 0))

        else:
            ax.set_xticks(x_axis_points + 0.4)
            ax.set_xticklabels([name.upper() if name != 'dmon' else 'DMoN' for name in x_axis_names], ha='left', fontsize=15)
            
        
        ax.set_axisbelow(True)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0.0-(bar_width/2), len(x_axis_points)-(bar_width/2))


    if "Large" not in title:
        fig.suptitle(title, fontsize=18)
    else:
        fig.suptitle(title, fontsize=18) 
    plt.tight_layout()
    if plot_legend:
        if "Synth" in title:
           plt.subplots_adjust(top=0.92, bottom=0.12, hspace=0.45)
        else:
           plt.subplots_adjust(top=0.92, bottom=0.12, hspace=0.17)
        blank_ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        blank_ax.axis('off')
        handles = []
        handles.append(mpatches.Patch(linewidth=0, facecolor=my_colours[0], label=r'$\mathcal{M}$'))
        handles.append(mpatches.Patch(linewidth=0, facecolor=my_colours[1], label=r'$\mathcal{M} \rightarrow$ F1', edgecolor='black'))
        handles.append(mpatches.Patch(linewidth=0, facecolor=my_colours[2], label=r'$\mathcal{M} \rightarrow$ NMI', edgecolor='black'))

        handles.append(mpatches.Patch(linewidth=0, facecolor=my_colours[3], label=r'$\mathcal{C}$'))
        handles.append(mpatches.Patch(linewidth=0, facecolor=my_colours[4], label=r'$\mathcal{C} \rightarrow$ F1', edgecolor='black'))
        handles.append(mpatches.Patch(linewidth=0, facecolor=my_colours[5], label=r'$\mathcal{C} \rightarrow$ NMI',  edgecolor='black'))
        if include_defaults:
            handles.append(mlines.Line2D([], [], color=[0,0,0,0], linewidth=0, label=''))

            # # Define the font (Times New Roman)
            # font_properties = fm.FontProperties(family='DejaVu Sans', size=20)
            # # Create the text path for the custom symbol (U+26F6)
            # symbol_path = TextPath((0, 0), '$\U00002B1A$', prop=font_properties)
            # # Create a patch from the path
            # symbol_patch = PathPatch(symbol_path, transform=IdentityTransform(), lw=1.5, edgecolor='black', facecolor='none')
            # handles.append(mlines.Line2D([], [], color=[0,0,0,0], linewidth=1, markeredgewidth=1.5, marker=symbol_patch, markersize=10, linestyle=':', markeredgecolor='black', label='Default Hyperparameters'))
            # handles.append(mlines.Line2D([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], linestyle='--', color='black', linewidth=2, label='Default Hyperparameters'))
            # handles.append(mlines.Line2D([], [], color=[0,0,0,0], linewidth=1, markeredgewidth=1.5, marker="s", markersize=10, linestyle=':', markeredgecolor='black', label='Default Hyperparameters'))
            # handles.append(mlines.Line2D([], [], color=[0,0,0,0], linewidth=0, label='Default Hyperparameters'))
            # handles.append(mlines.Line2D([], [], color=[0,0,0,0], linewidth=0, label=''))
            class LabeledObject(object):
                def __init__(self, label):
                    self.label = label
                
                def get_label(self):
                    return self.label

            class CustomRectangleHandler(object):
                def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                    x0, y0 = handlebox.xdescent, handlebox.ydescent
                    width, height = handlebox.width, handlebox.height
                    patch = mpatches.Rectangle([x0, y0], width, height, facecolor='none',
                                            edgecolor='black', linestyle='--', lw=1,
                                            transform=handlebox.get_transform())
                    handlebox.add_artist(patch)
                    return patch

            custom_rect = LabeledObject('Default Hyperparameters')
            handles.append(custom_rect)

            blank_ax.legend(handles=handles, loc='lower center', fontsize=12, ncols=3, handler_map={LabeledObject: CustomRectangleHandler()})
        else:
            blank_ax.legend(handles=handles, loc='lower center', fontsize=12, ncols=3)

    save_name = f'{os.path.basename(os.path.normpath(folder))}'
    save_name = save_name.replace("\'", "")
    print(f'Saving: {save_name}')
    if 'citeseer' in datasets:
        plt.savefig(f'./figures/postvivathesis/{save_name}.pdf')
    elif 'texas' in datasets: 
        plt.savefig(f'./figures/postvivathesis/{save_name}_1.pdf')    
    elif 'wisc' in datasets: 
        plt.savefig(f'./figures/postvivathesis/{save_name}_2.pdf')
    if plot_legend:
        return handles

def calculate_framework_comparison_rank(datasets, algorithms, folder, default_algorithms, dfolder, sfirst_folder, sfirst_dfolder):
    # get results for both
    mod_results, con_results = extract_results(datasets, algorithms, folder, sfirst_folder)
    dmod_results, dcon_results = extract_results(datasets, default_algorithms, dfolder, sfirst_dfolder)
    dcon_results[:, 0] = dcon_results[:, 0]
    con_results[:, 0] = con_results[:, 0]
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

    means_hpo_std = np.std(rankings, axis=0)[0]
    means_def_std = np.std(rankings, axis=0)[1]
    print(f'HPO FCR: {means_hpo:.3f}$\pm$ {means_hpo_std:.2f}')
    print(f'Default FCR: {means_def:.3f}$\pm${means_def_std:.2f}')
    return 


if __name__ == "__main__":
    matplotlib.use("macosx")
    make_ugle = False
    make_big_figure = True
    make_dist_figure = True
    make_presentation_figures = True
    make_paper_figures =  True
    make_rankings_table = True

    make_unsuper = True
    calc_increases = False
    calc_synth_increases = False

    make_abs = False
    make_corr = False
    make_synth = False

    if make_ugle:
        algorithms = ['daegc', 'dgi', 'dmon', 'grace', 'mvgrl', 'sublime', 'bgrl', 'vgaer']
        datasets = ['cora', 'citeseer', 'dblp', 'bat', 'eat', 'texas', 'wisc', 'cornell', 'uat', 'amap']
        metrics = ['f1', 'nmi', 'modularity', 'conductance']
        folder = './results/legacy_results/progress_results/'
        search_first_post_viva = './post_viva_results/cm/hpo/'

        seeds = [42, 24, 976, 12345, 98765, 7, 856, 90, 672, 785]
        default_algos = ['daegc_default', 'dgi_default', 'dmon_default', 'grace_default', 'mvgrl_default',
                        'sublime_default', 'bgrl_default', 'vgaer_default']
        default_folder = './results/legacy_results/default_results/'
        search_first_post_viva_default = './post_viva_results/cm/default/'

        if make_presentation_figures: 
            create_rand_dist_comparison(['cora'], algorithms, metrics, seeds, folder, default_algos, default_folder,
                                        search_first_post_viva, search_first_post_viva_default)

        if make_paper_figures: 
            if make_big_figure:
                create_big_figure(['cora', 'citeseer'], algorithms, folder, default_algos, default_folder, search_first_post_viva, search_first_post_viva_default)
                create_big_figure(['amap', 'dblp'], algorithms, folder, default_algos, default_folder, search_first_post_viva, search_first_post_viva_default)
                create_big_figure(['texas', 'wisc', 'cornell'], algorithms, folder, default_algos, default_folder, search_first_post_viva, search_first_post_viva_default)
                create_big_figure(['bat', 'eat', 'uat'], algorithms, folder, default_algos, default_folder, search_first_post_viva, search_first_post_viva_default)



            # =========== DO THE FOLLOWING + OUTPUT RANKINGS FOR ALL DIFFERENT TESTS ========= 
            # fetch absolute results

            if 'uadgn' in algorithms: 
                exit()
            print('\n\n\n==================== PRINTING RANKING TABLE INFO ==========================\n\n\n')
            ################# ========== LOADED OBJECT ========== ###################
            result_object = make_test_performance_object(datasets, algorithms, metrics, seeds, folder, search_first_post_viva)
            default_result_object = make_test_performance_object(datasets, default_algos, metrics, seeds, default_folder, search_first_post_viva_default)


            ## OVERALL MATHCAL R 
            ## RANK FOR EACH METHOD
            if make_rankings_table: 
                result_object_fcr = result_object.flatten()
                default_result_object_fcr = default_result_object.flatten()
                
                # make comparisons
                n_comparisons = result_object_fcr.shape[0]
                rankings = np.zeros((n_comparisons, 2))
                for i in range(n_comparisons):
                    if result_object_fcr[i] > default_result_object_fcr[i]:
                        rankings[i] = [1, 2]
                    elif default_result_object_fcr[i] < result_object_fcr[i]:
                        rankings[i] = [2, 1]
                    else:
                        rankings[i] = [1.5, 1.5]
                means_hpo = np.mean(rankings, axis=0)[0]
                stds_hpo = np.std(rankings, axis=0)[0]
                means_def = np.mean(rankings, axis=0)[1]
                stds_def = np.std(rankings, axis=0)[1]

            # calculate ranking of each metric
            ranking_object = calculate_ranking_performance(result_object, datasets, metrics, seeds, calc_ave_first=False)
            default_ranking_object = calculate_ranking_performance(default_result_object, datasets, metrics, seeds, calc_ave_first=False)

            # print the average rank 
            if make_rankings_table: 
                print('all')
                
                rank_order = np.argsort(np.around(np.mean(ranking_object, axis=(0, 2, 3)), 1)) 
                
                hpo_ranks = np.around(np.mean(ranking_object, axis=(0, 2, 3)), 1)[rank_order]
                hpo_std = np.around(np.std(ranking_object, axis=(0, 2, 3)), 0)[rank_order]

                def_rank_order = np.argsort(np.around(np.mean(default_ranking_object, axis=(0, 2, 3)), 1)) 
                default_ranks = np.around(np.mean(default_ranking_object, axis=(0, 2, 3)), 1)[def_rank_order]
                default_std = np.around(np.std(default_ranking_object, axis=(0, 2, 3)), 0)[def_rank_order]

                print('HPO')
                print(f'{means_hpo:.1f}$\pm${stds_hpo:.1f}', end=' ')
                for air, _ in enumerate(hpo_ranks):
                    print(f'& {hpo_ranks[air]}$\pm${int(hpo_std[air])}', end=' ')
                print('\nDEFAULT')
                print(f'{means_def:.1f}$\pm${stds_def:.1f}', end=' ')
                for air, _ in enumerate(default_ranks):
                    print(f'& {default_ranks[air]}$\pm${int(default_std[air])}', end=' ')


                print('')
                print(f'MATCAL HPO: {means_hpo:.1f}$\pm${stds_hpo:.1f}')
                print(f'MATCAL DEF: {means_def:.1f}$\pm${stds_def:.1f}')
                print(np.array(algorithms)[rank_order])
                print(np.array(algorithms)[def_rank_order])
                print('\n')


                ## PER METRIC MATHCAL R
                ## (11, 8, 4, 10)
                for m, metric in enumerate(metrics):
                    copyresult_object = np.expand_dims(deepcopy(result_object[:, :, m, :]), axis=2)
                    copydefault_result_object = np.expand_dims(deepcopy(default_result_object[:, :, m, :]), axis=2)

                    result_object_fcr = copyresult_object.flatten()
                    default_result_object_fcr = copydefault_result_object.flatten()
                    
                    # make comparisons
                    n_comparisons = result_object_fcr.shape[0]
                    rankings = np.zeros((n_comparisons, 2))
                    for i in range(n_comparisons):
                        if result_object_fcr[i] > default_result_object_fcr[i]:
                            rankings[i] = [1, 2]
                        elif default_result_object_fcr[i] < result_object_fcr[i]:
                            rankings[i] = [2, 1]
                        else:
                            rankings[i] = [1.5, 1.5]
                    means_hpo = np.mean(rankings, axis=0)[0]
                    stds_hpo = np.std(rankings, axis=0)[0]
                    means_def = np.mean(rankings, axis=0)[1]
                    stds_def = np.std(rankings, axis=0)[1]

                    # calculate ranking of each metric
                    ranking_object = calculate_ranking_performance(copyresult_object, datasets, [metric], seeds, calc_ave_first=False)
                    default_ranking_object = calculate_ranking_performance(copydefault_result_object, datasets, [metric], seeds, calc_ave_first=False)

                    # print the average rank 
                    print(f'{metric}')
                    hpo_ranks = np.around(np.mean(ranking_object, axis=(0, 2, 3)), 1)[rank_order]
                    hpo_std = np.around(np.std(ranking_object, axis=(0, 2, 3)), 0)[rank_order]


                    default_ranks = np.around(np.mean(default_ranking_object, axis=(0, 2, 3)), 1)[def_rank_order]
                    default_std = np.around(np.std(default_ranking_object, axis=(0, 2, 3)), 0)[def_rank_order]

                    print('HPO')
                    print(f'{means_hpo:.1f}$\pm${stds_hpo:.1f}', end=' ')
                    for air, _ in enumerate(hpo_ranks):
                        print(f'& {hpo_ranks[air]}$\pm${int(hpo_std[air])}', end=' ')
                    print('\nDEFAULT')
                    print(f'{means_def:.1f}$\pm${stds_def:.1f}', end=' ')
                    for air, _ in enumerate(default_ranks):
                        print(f'& {default_ranks[air]}$\pm${int(default_std[air])}', end=' ')
                    print('\n')

                print('\n')
                for d, dataset in enumerate(datasets):
            
                    copyresult_object = np.expand_dims(deepcopy(result_object[d, :, :, :]), axis=0)
                    copydefault_result_object = np.expand_dims(deepcopy(default_result_object[d, :, :, :]), axis=0)

                    result_object_fcr = copyresult_object.flatten()
                    default_result_object_fcr = copydefault_result_object.flatten()
                    
                    # make comparisons
                    n_comparisons = result_object_fcr.shape[0]
                    rankings = np.zeros((n_comparisons, 2))
                    for i in range(n_comparisons):
                        if result_object_fcr[i] > default_result_object_fcr[i]:
                            rankings[i] = [1, 2]
                        elif default_result_object_fcr[i] < result_object_fcr[i]:
                            rankings[i] = [2, 1]
                        else:
                            rankings[i] = [1.5, 1.5]
                    means_hpo = np.mean(rankings, axis=0)[0]
                    stds_hpo = np.std(rankings, axis=0)[0]
                    means_def = np.mean(rankings, axis=0)[1]
                    stds_def = np.std(rankings, axis=0)[1]

                    # calculate ranking of each metric
                    ranking_object = calculate_ranking_performance(copyresult_object, [dataset], metrics, seeds, calc_ave_first=False)
                    default_ranking_object = calculate_ranking_performance(copydefault_result_object, [dataset], metrics, seeds, calc_ave_first=False)

                    # print the average rank 
                    print(f'{dataset}')
                    hpo_ranks = np.around(np.mean(ranking_object, axis=(0, 2, 3)), 1)[rank_order]
                    hpo_std = np.around(np.std(ranking_object, axis=(0, 2, 3)), 0)[rank_order]
                    default_ranks = np.around(np.mean(default_ranking_object, axis=(0, 2, 3)), 1)[def_rank_order]
                    default_std = np.around(np.std(default_ranking_object, axis=(0, 2, 3)), 0)[def_rank_order]
                    
                    print('HPO')
                    print(f'{means_hpo:.1f}$\pm${stds_hpo:.1f}', end=' ')
                    for air, _ in enumerate(hpo_ranks):
                        print(f'& {hpo_ranks[air]}$\pm${int(hpo_std[air])}', end=' ')
                    print('\nDEFAULT')
                    print(f'{means_def:.1f}$\pm${stds_def:.1f}', end=' ')
                    for air, _ in enumerate(default_ranks):
                        print(f'& {default_ranks[air]}$\pm${int(default_std[air])}', end=' ')
                    print('\n')


                for a, algorithm in enumerate(algorithms):
                    copyresult_object = np.expand_dims(deepcopy(result_object[:, a, :, :]), axis=0)
                    copydefault_result_object = np.expand_dims(deepcopy(default_result_object[:, a, :, :]), axis=0)

                    result_object_fcr = copyresult_object.flatten()
                    default_result_object_fcr = copydefault_result_object.flatten()
                    
                    # make comparisons
                    n_comparisons = result_object_fcr.shape[0]
                    rankings = np.zeros((n_comparisons, 2))
                    for i in range(n_comparisons):
                        if result_object_fcr[i] > default_result_object_fcr[i]:
                            rankings[i] = [1, 2]
                        elif default_result_object_fcr[i] < result_object_fcr[i]:
                            rankings[i] = [2, 1]
                        else:
                            rankings[i] = [1.5, 1.5]
                    means_hpo = np.mean(rankings, axis=0)[0]
                    stds_hpo = np.std(rankings, axis=0)[0]
                    means_def = np.mean(rankings, axis=0)[1]
                    stds_def = np.std(rankings, axis=0)[1]

                    print(f'{algorithm}')
                    print(f'MATCAL HPO: {means_hpo:.1f}$\pm${stds_hpo:.1f}')
                    print(f'MATCAL DEF: {means_def:.1f}$\pm${stds_def:.1f}')

                # for s, seed in enumerate(seeds):
                #     copyresult_object = np.expand_dims(deepcopy(result_object[:, :, :, s]), axis=0)
                #     copydefault_result_object = np.expand_dims(deepcopy(default_result_object[:, :, :, s]), axis=0)

                #     result_object_fcr = copyresult_object.flatten()
                #     default_result_object_fcr = copydefault_result_object.flatten()
                    
                #     # make comparisons
                #     n_comparisons = result_object_fcr.shape[0]
                #     rankings = np.zeros((n_comparisons, 2))
                #     for i in range(n_comparisons):
                #         if result_object_fcr[i] > default_result_object_fcr[i]:
                #             rankings[i] = [1, 2]
                #         elif default_result_object_fcr[i] < result_object_fcr[i]:
                #             rankings[i] = [2, 1]
                #         else:
                #             rankings[i] = [1.5, 1.5]
                #     means_hpo = np.mean(rankings, axis=0)[0]
                #     means_def = np.mean(rankings, axis=0)[1]

                #     print(f'{seed} - MATCAL HPO/DEF: {means_hpo:.1f}/{means_def:.1f}')

            # calculate ranking of each metric
            ranking_object = calculate_ranking_performance(result_object, datasets, metrics, seeds, calc_ave_first=False)
            default_ranking_object = calculate_ranking_performance(default_result_object, datasets, metrics, seeds, calc_ave_first=False)

            ranking_object = reshape_ranking_to_test_object(ranking_object)
            default_ranking_object = reshape_ranking_to_test_object(default_ranking_object)

            og_w = og_randomness(ranking_object, print_draws=True)
            print(f"OG W HPO: {og_w:.3f}")
            og_w_def = og_randomness(default_ranking_object, print_draws=True)
            print(f"OG W Default: {og_w_def:.3f}")

            # result_object = np.concatenate((result_object[:, :, 0:3, :], con_out.reshape((result_object.shape[0], result_object.shape[1], 1, result_object.shape[3]))), axis=2)
            # default_result_object = np.concatenate((default_result_object[:, :, 0:3, :], dcon_out.reshape((result_object.shape[0], result_object.shape[1], 1, result_object.shape[3]))), axis=2)
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
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 5))

                test_interval = 1
                n_tests = result_object.shape[0]
                n_repeats = 100
                input_idxs = range(0, n_tests)
                titles = ['Original ' + r'$\mathcal{W}$' + ' Randomness', 'Mean Ties ' + r'$\mathcal{W}^m$' + ' Randomness', 'Mean Ties ' + r'$\mathcal{W}^t$' + ' Randomness', r'$\mathcal{W}^w$' + ' Wasserstein Randomness']
                w_fns = ['og_randomness', 'og_newOld_randomness', 'ties_randomness', 'wasserstein_randomness']
                
                wshpo_cov = np.zeros((4, n_tests-1))
                wsdef_cov = np.zeros((4, n_tests-1))

                for a, ax in enumerate(axes.flat):
                    print(f'axis: {a}')
                    if w_fns[a] == 'og_randomness':
                        temp_object = ranking_object
                        default_temp_object = default_ranking_object
                    else:
                        temp_object = result_object
                        default_temp_object = default_result_object
                    cur_max = 0.
                    cur_min = 1.
                    for test_interval in range(1, n_tests):
                        whpo = []
                        wdef = []
                        for i in range(n_repeats):
                            test_idx = random.sample(input_idxs, test_interval)

                            temp_rankings = temp_object[test_idx, :, :]
                            WW_HPO = locals()[w_fns[a]](temp_rankings)
                            ax.plot(test_interval, WW_HPO, 'x', c='C4', markersize=2)

                            temp_rankings = default_temp_object[test_idx, :, :]
                            WW_DEF = locals()[w_fns[a]](temp_rankings)
                            ax.plot(test_interval, WW_DEF, 'x', c='C3', markersize=2)

                            cur_max = max(cur_max, WW_HPO)
                            cur_max = max(cur_max, WW_DEF)

                            cur_min = min(cur_min, WW_HPO)
                            cur_min = min(cur_min, WW_DEF)

                            # FOR COVARIANCE STUFF
                            whpo.append(WW_HPO)
                            wdef.append(WW_DEF)
                        
                        wshpo_cov[a, test_interval-1] = np.std(whpo) / np.mean(whpo)
                        wsdef_cov[a, test_interval-1] = np.std(wdef) / np.mean(wdef)

                    ax.set_xticks([1, 10, 20, 30, n_tests-1])
                    ax.set_xlim(0.5, n_tests-0.5)

                    
                    # PLOTTING FOR W RANDOMNESS FIGURE
                    handles = []
                    handles.append(mlines.Line2D([], [], label='HPO', color="C4", marker='x', linestyle='None', markersize=7, linewidth=2))
                    handles.append(mlines.Line2D([], [], label='Default', color="C3", marker='x', linestyle='None', markersize=7, linewidth=2))
                    if a != 1 and a != 3:
                        ax.set_ylabel(r'$\mathcal{W}$', fontsize=10)
                    if a != 0 and a != 1:
                        ax.set_xlabel(r'$\vert \mathcal{T} \vert $', fontsize=10)

                    ax.set_title(titles[a], fontsize=12)
                    ax.set_ylim(max(0, cur_min-0.1), min(1., cur_max+0.1))

                
                fig.tight_layout()
                blank_ax = fig.add_axes([0, 0, 1, 1], frameon=False)
                blank_ax.axis('off')
                #blank_ax.legend(handles=handles, bbox_to_anchor=(0.975, 0.2), fontsize=14)
                blank_ax.legend(handles=handles, loc='lower center', fontsize=10, ncols=2)
                fig.subplots_adjust(bottom=0.15)
                
                fig.savefig(f"{ugle_path}/figures/postvivathesis/w_distribution_entropy.pdf", bbox_inches='tight')


                fig2, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(6, 5))
                # PLOTTING CURVES
                def decaying_exponential(t, a, b, c):
                    return a * np.exp(-b * t) + c

                # Initial guess for parameters a, b, c
                initial_guess = [10, 0.5, 1]
                b_fits = []
                b_fitshpos = []

                for a2, ax2 in enumerate(axes2.flat):
                    t = np.array(list(range(1, n_tests)))
                    wdef = wsdef_cov[a2]
                    whpo = wshpo_cov[a2]

                    # Fit the curve
                    popt, pcov = curve_fit(decaying_exponential, t, wdef, p0=initial_guess)
                    popt_hpo, pcov_hpo = curve_fit(decaying_exponential, t, whpo, p0=initial_guess)

                    # Extract fitted coefficients: a, b, c
                    a_fit, b_fit, c_fit = popt
                    a_fithpo, b_fithpo, c_fithpo = popt_hpo

                    print(f"Fitted parameters Default {w_fns[a2]}: a = {a_fit:.2f}, b = {b_fit:.2f}, c = {c_fit:.2f}")
                    print(f"Fitted parameters HPO {w_fns[a2]}: a = {a_fithpo:.2f}, b = {b_fithpo:.2f}, c = {c_fithpo:.2f}")

                    # Use the fitted coefficients to calculate the fitted w values
                    w_fit = decaying_exponential(t, *popt)
                    w_fithpo = decaying_exponential(t, *popt_hpo)

                    # Plot the original data and the fitted curve
                    ax2.scatter(t, wdef, color='C3', label="Default Covariance", linewidth=2, marker='.')
                    ax2.plot(t, w_fit, color='salmon', label=f"Fitted Curve (b={b_fit:.2f}", linewidth=2)

                    ax2.scatter(t, whpo, color='C4', label="HPO Covariance", linewidth=2, marker='.')
                    ax2.plot(t, w_fithpo, color='plum', label=f"Fitted HPO Curve (b={b_fithpo:.2f}", linewidth=2)

                    b_fits.append(b_fit)
                    b_fitshpos.append(b_fithpo)
                    
                    if a2 != 1 and a2 != 3:
                        ax2.set_ylabel(r'$CV$', fontsize=10)
                    if a2 != 0 and a2 != 1:
                        ax2.set_xlabel(r'$\vert \mathcal{T} \vert $', fontsize=10)

                    ax2.set_title(titles[a2], fontsize=12)

                    ax2.set_xticks([1, 10, 20, 30, n_tests-1])
                    ax2.set_xlim(0.5, n_tests-0.5)
                    ax2.set_ylim(0, None)

                handles = []
                handles.append(mlines.Line2D([], [], label='HPO Coefficient of Variance', color="C4", marker='.', linestyle='None', markersize=7, linewidth=2))
                handles.append(mlines.Line2D([], [], label='Default Coefficient of Variance', color="C3", marker='.', linestyle='None', markersize=7, linewidth=2))
                handles.append(mlines.Line2D([], [], label='HPO Fitted Negative Exponetial', color="plum", marker=None, linestyle='-', markersize=7, linewidth=2))
                handles.append(mlines.Line2D([], [], label='Default Fitted Negative Exponetial', color="salmon", marker=None, linestyle='-', markersize=7, linewidth=2))
                # PLOTTING / FITTING NEGATIVE EXPONETIAL CURVE
                fig2.tight_layout()
                blank_ax = fig2.add_axes([0, 0, 1, 1], frameon=False)
                blank_ax.axis('off')
                #blank_ax.legend(handles=handles, bbox_to_anchor=(0.975, 0.2), fontsize=14)
                blank_ax.legend(handles=handles, loc='lower center', fontsize=10, ncols=2)
                fig2.subplots_adjust(bottom=0.2)
                for a2, ax2 in enumerate(axes2.flat):
                    ax2.text(s=r'Default $\omega$=' + f'{b_fits[a2]:.2f}', x=0.25, y=0.9, transform=ax2.transAxes)
                    ax2.text(s=r'HPO $\omega$=' + f'{b_fitshpos[a2]:.2f}', x=0.25, y=0.8, transform=ax2.transAxes)

                fig2.savefig(f"{ugle_path}/figures/postvivathesis/w_dist_covariance.pdf", bbox_inches='tight')
                # PLOTTING LOOK NICE FUNCTIONS


    if make_unsuper:
        q1_folder = './results/unsupervised_limit/default_q1/'
        q2_folder = './results/unsupervised_limit/hpo_q2/'
        qlarge_folder = './results/unsupervised_limit/hpo_large/'
        # q5_folder = './results/unsupervised_limit/synth_default_q5/'
        q5_folder1 = './results/unsupervised_limit/33_default_q4/'
        q5_folder2 = './results/unsupervised_limit/66_default_q4/'


        sfirst_q1_folder = './post_viva_results/ul/default/'
        sfirst_q2_folder = './post_viva_results/ul/hpo/'
        sfirst_qlarge_folder = './post_viva_results/ul/hpo_large/'
        sfirst_q5_folder1 = './post_viva_results/ul/33_train/'
        sfirst_q5_folder2 = './post_viva_results/ul/66_train/'
   

        seeds = [42, 24, 976, 12345, 98765, 7, 856, 90, 672, 785]
        datasets = ['citeseer', 'cora', 'texas', 'dblp', 'wisc', 'cornell']
        default_algorithms = ['dgi_default', 'daegc_default', 'dmon_default', 'grace_default', 'sublime_default', 'bgrl_default', 'vgaer_default']
        algorithms = ['dgi', 'daegc', 'dmon', 'grace', 'sublime', 'bgrl', 'vgaer']

       
        q1sup_folder = './results/unsupervised_limit/default_sup_select/'
        q2sup_folder = './results/unsupervised_limit/hpo_q2_sup/'
        qlargesup_folder = './results/unsupervised_limit/hpo_large_sup/'
        # synthsup_folder = './results/unsupervised_limit/synth_default_q5_sup/'
        q5sup_33 = './results/unsupervised_limit/default_33_sup/'
        q5sup_66 = './results/unsupervised_limit/default_66_sup/'


        sfirst_q1sup_folder = './post_viva_results/ul/default_sup/'
        sfirst_q2sup_folder = './post_viva_results/ul/hpo_sup/'
        sfirst_qlargesup_folder = './post_viva_results/ul/hpo_large_sup/'
        sfirst_q5sup_folder1 = './post_viva_results/ul/33_train_sup/'
        sfirst_q5sup_folder2 = './post_viva_results/ul/66_train_sup/'

        extract_infos = zip([q1_folder, q2_folder, qlarge_folder, qlarge_folder, q5_folder1, q5_folder2], 
                            [q1sup_folder, q2sup_folder, qlargesup_folder, qlargesup_folder, q5sup_33, q5sup_66],
                            [default_algorithms, algorithms, ['dmon'], ['dmon'], default_algorithms, default_algorithms],
                            [datasets, datasets, ['Photo'], ['Computers'], datasets, datasets],
                            [sfirst_q1_folder, sfirst_q2_folder, sfirst_qlarge_folder, sfirst_qlarge_folder, sfirst_q5_folder1, sfirst_q5_folder2],
                            [sfirst_q1sup_folder, sfirst_q2sup_folder, sfirst_qlargesup_folder, sfirst_qlargesup_folder, sfirst_q5sup_folder1, sfirst_q5sup_folder2])

        if calc_increases:
            for norm, sup, algos, dset, normsfirst, supfirst in extract_infos:
                # calculate the percentage drops
                print(sup.split("/")[-2])
                ################# ========== LOADED OBJECT ========== ###################
                f1_nmi_results = extract_supervised_results(dset, algos, sup, supfirst)
                print(norm.split("/")[-2])
                ################# ========== LOADED OBJECT ========== ###################
                dmod_results, dcon_results = extract_results(dset, algos, norm, normsfirst)
                calc_percent_increase(f1_nmi_results, dmod_results, dcon_results)


            calculate_framework_comparison_rank(datasets, algorithms, q2_folder, default_algorithms, q1_folder, sfirst_q2_folder, sfirst_q1_folder)

        if make_abs: 
            handles = create_abs_performance_figure(['citeseer', 'cora'], algorithms, q2_folder, sfirst_q2_folder, title="HPO vs Default HP Performance", plot_dims=[2, 1], figsize=(9, 11), plot_legend=True, 
                                                    default_algos=default_algorithms, default_folder=q1_folder, default_sfirst_folder=sfirst_q1_folder)
            create_abs_performance_figure(['texas', 'dblp'], algorithms, q2_folder, sfirst_q2_folder, title="HPO vs Default HP Performance", plot_dims=[2, 1], figsize=(9, 11), plot_legend=True,
                                          default_algos=default_algorithms, default_folder=q1_folder, default_sfirst_folder=sfirst_q1_folder)
            create_abs_performance_figure(['wisc', 'cornell'], algorithms, q2_folder, sfirst_q2_folder, title="HPO vs Default HP Performance", plot_dims=[2, 1], figsize=(9, 11), plot_legend=True,
                                          default_algos=default_algorithms, default_folder=q1_folder, default_sfirst_folder=sfirst_q1_folder)
          
            #create_abs_performance_figure(['Computers', 'Photo'], ['dmon'], qlarge_folder, title="DMoN Performance Large Datasets with HPO", plot_dims=[1, 2], figsize=(9, 10), plot_legend=True)
            
            create_abs_performance_figure(['citeseer', 'cora'], default_algorithms, q5_folder2, sfirst_q5_folder2 , title="Default Hyperparameters with 66\% of Training Data", plot_dims=[2, 1], figsize=(9, 11), plot_legend=True)
            create_abs_performance_figure(['texas', 'dblp'], default_algorithms, q5_folder2, sfirst_q5_folder2 , title="Default Hyperparameters with 66\% of Training Data", plot_dims=[2, 1], figsize=(9, 11), plot_legend=True)
            create_abs_performance_figure(['wisc', 'cornell'], default_algorithms, q5_folder2, sfirst_q5_folder2 , title="Default Hyperparameters with 66\% of Training Data", plot_dims=[2, 1], figsize=(9, 11), plot_legend=True)

            create_abs_performance_figure(['citeseer', 'cora'], default_algorithms, q5_folder1, sfirst_q5_folder1 , title="Default Hyperparameters with 33\% of Training Data", plot_dims=[2, 1], figsize=(9, 11), plot_legend=True)
            create_abs_performance_figure(['texas', 'dblp'], default_algorithms, q5_folder1, sfirst_q5_folder1 , title="Default Hyperparameters with 33\% of Training Data", plot_dims=[2, 1], figsize=(9, 11), plot_legend=True)
            create_abs_performance_figure(['wisc', 'cornell'], default_algorithms, q5_folder1, sfirst_q5_folder1 , title="Default Hyperparameters with 33\% of Training Data", plot_dims=[2, 1], figsize=(9, 11), plot_legend=True)



        if make_corr:
            print("\n\n\n LARGE \n")
            unsupervised_prediction_graph(['Computers', 'Photo'], ['dmon'], qlarge_folder, qlarge_folder, title="Large Dataset HPO")
            print("\n\n\n 66% \n")
            unsupervised_prediction_graph(datasets, default_algorithms, q5_folder2, sfirst_q5_folder2, title="q4: 66\% of the data")
            print("\n\n\n 33% \n")
            unsupervised_prediction_graph(datasets, default_algorithms, q5_folder1,sfirst_q5_folder1, title="q4: 33\% of the data")
            print("\n\n\n Default \n")
            unsupervised_prediction_graph(datasets, default_algorithms, q1_folder, sfirst_q1_folder, title="Default Hyperparameters Correlation", pltlegend=True)
            print("\n\n\n HPO \n")
            handles2 = unsupervised_prediction_graph(datasets, algorithms, q2_folder, sfirst_q2_folder  ,title="Hyperparameter Optimisation Correlation", pltlegend=True)


        make_corr_tables = True
        if make_corr_tables:
                
                hpo_res = print_algo_table(datasets, algorithms, q2_folder, sfirst_q2_folder)
                def_res = print_algo_table(datasets, default_algorithms, q1_folder, sfirst_q1_folder)

                print('ALGO TABLE')
                for a in range(hpo_res.shape[0]):
                    print(f'\n{algorithms[a]}')
                    for idx in range(hpo_res.shape[1]):
                        if hpo_res[a, idx] > def_res[a, idx]:
                            print(f'{def_res[a, idx]}(\\textbf{{{hpo_res[a, idx]}}})', end =' & ')
                        else:
                            print(f'\\textbf{{{def_res[a, idx]}}}({hpo_res[a, idx]})', end =' & ')

 
                hpo_res = print_dataset_table(datasets, algorithms, q2_folder, sfirst_q2_folder)
                def_res = print_dataset_table(datasets, default_algorithms, q1_folder, sfirst_q1_folder)


                print('\n\nDATASET TABLE')
                for a in range(hpo_res.shape[0]):
                    print(f'\n{datasets[a]}')
                    for idx in range(hpo_res.shape[1]):
                        if hpo_res[a, idx] > def_res[a, idx]:
                            print(f'{def_res[a, idx]}(\\textbf{{{hpo_res[a, idx]}}})', end =' & ')
                        else:
                            print(f'\\textbf{{{def_res[a, idx]}}}({hpo_res[a, idx]})', end =' & ')
                print('\n')


        def create_handles_image(handles, name):
            if name == 'corr':
                fig, axes = plt.subplots(figsize=(9, 0.75))
            else:
                fig, axes = plt.subplots(figsize=(5, 0.5))
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_xticklabels([])
            axes.set_yticklabels([])
            axes.axis('off')
            blank_ax = fig.add_axes([0, 0, 1, 1], frameon=False)
            blank_ax.axis('off')
            if name == 'corr':
                blank_ax.legend(handles=handles, loc='lower center', fontsize=15, ncols=5)
            else:
                blank_ax.legend(handles=handles, loc='lower center', fontsize=12, ncols=3)
            
            plt.savefig(f'./figures/postvivathesis/handles_{name}.png', format='png')
        # if make_abs: 
        #     create_handles_image(handles, name='abs')
        # if make_corr:
        #     create_handles_image(handles, name='corr')
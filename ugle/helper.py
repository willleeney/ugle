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
import pandas as pd

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
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 5))

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
    ax.bar(x_axis_names, f1, yerr=f1_std,
           width=bar_width, facecolor=alt_colours[0], alpha=0.9, linewidth=0, label='f1')
    ax.bar(x_axis_names + bar_width, nmi, yerr=nmi_std,
           width=bar_width, facecolor=alt_colours[1], alpha=0.9, linewidth=0, label='nmi')
    ax.bar(x_axis_names + (2 * bar_width), modularity, yerr=modularity_std,
           width=bar_width, facecolor=alt_colours[2], alpha=0.9, linewidth=0, label='modularity')
    ax.bar(x_axis_names + (3 * bar_width), conductance, yerr=conductance_std,
           width=bar_width, facecolor=alt_colours[3], alpha=0.9, linewidth=0, label='conductance')

    # plot default parameters bars in dashed lines
    blank_colours = np.zeros(4)
    ax.bar(x_axis_names, default_f1, width=bar_width,
           facecolor=blank_colours, edgecolor='black', linewidth=2, linestyle='--', label='default values')
    ax.bar(x_axis_names + bar_width, default_nmi, width=bar_width,
           facecolor=blank_colours, edgecolor='black', linewidth=2, linestyle='--')
    ax.bar(x_axis_names + (2 * bar_width), default_modularity, width=bar_width,
           facecolor=blank_colours, edgecolor='black', linewidth=2, linestyle='--')
    ax.bar(x_axis_names + (3 * bar_width), default_conductance,
           facecolor=blank_colours, width=bar_width, edgecolor='black', linewidth=2, linestyle='--')

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
    ax.set_title(dataset_name, y=0.95, fontsize=98)
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(42)

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


def create_rand_dist_fig(ax, title, datasets, algorithms, metrics, seeds, folder, calc_ave_first=False, set_legend=False):
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
    if calc_ave_first:
        ave_axis = [0, 2]
    else:
        ave_axis = [0, 2, 3]

    n_ranks = 1
    for axis in ave_axis:
        n_ranks *= ranking_object.shape[axis]

    # average rank over dataset and metric + sensitivity over seed
    all_ranks_per_algo = np.zeros(shape=(len(algorithms), n_ranks))
    for i, algo in enumerate(algorithms):
        all_ranks_per_algo[i] = ranking_object[:, i, :].flatten()

    # with plt.xkcd():
    # plot the lines
    for j, algo_ranks in enumerate(all_ranks_per_algo):
        kde = gaussian_kde(algo_ranks)
        y_axis = kde.evaluate(x_axis)
        ax.plot(x_axis, y_axis, label=algorithms[j], zorder=10)

    # ww_coeff = kendall_w(all_ranks_per_algo.T)

    if set_legend:
        #ax.legend(loc='best', fontsize=16, ncol=3)
        ax.legend(loc='upper center', fontsize=15, ncol=3, bbox_to_anchor=(0.5, -0.35))
        #ax.set_xlabel('algorithm rank distribution over all tests', fontsize=18)

    #ax.set_ybound(0, 3)
    ax.set_xbound(1, 10)
    ax.set_xlabel(r'$r$' + ' : algorithm ranking', fontsize=20)
    ax.set_ylabel(r'$f_{j}(r)$', fontsize=20) #kde estimatation of rank distribution

    #ax.text(0.4, 0.85, ave_overlap_text, fontsize=20, transform=ax.transAxes, zorder=1000)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=15)

    # ax.set_title(title + f'{ww_coeff:.3f}', fontsize=20)
    """
    return ax


def create_big_figure(datasets, algorithms, folder, default_algos, default_folder):
    """
    creates figure for all datasets tested comparing default and hpo results
    """
    # create holder figure
    nrows, ncols = 4, 3
    col_n, row_n = 0, 0
    fig, axs = plt.subplots(nrows, ncols, figsize=(54, 78))

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

    axs[0, 0].legend()
    for item in axs[0, 0].get_legend().get_texts():
        item.set_fontsize(48)

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


def create_rand_dist_comparison(datasets: list, algorithms: list, metrics: list, seeds: list, folder: str,
                                default_algos: list, default_folder: str, titles: list):

    # create holder figure
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 7.5))

    titles[0] = r'$W(\mathcal{T}_{(default)})$: '
    titles[1] = r'$W(\mathcal{T}_{(hpo)})$: '

    ax[0] = create_rand_dist_fig(ax[0], titles[0], datasets, default_algos, metrics, seeds, default_folder)
    ax[1] = create_rand_dist_fig(ax[1], titles[1], datasets, algorithms, metrics, seeds, folder, set_legend=True)

    result_object = make_test_performance_object(datasets, algorithms, metrics, seeds, folder)
    default_result_object = make_test_performance_object(datasets, default_algos, metrics, seeds, default_folder)
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


algorithms = ['daegc', 'dgi', 'dmon', 'grace', 'mvgrl', 'selfgnn', 'sublime', 'bgrl', 'vgaer', 'cagc']
datasets = ['cora', 'citeseer', 'dblp', 'bat', 'eat', 'texas', 'wisc', 'cornell', 'uat', 'amac', 'amap']
metrics = ['f1', 'nmi', 'modularity', 'conductance']
folder = './results/legacy_results/progress_results/'
seeds = [42, 24, 976, 12345, 98765, 7, 856, 90, 672, 785]
default_algos = ['daegc_default', 'dgi_default', 'dmon_default', 'grace_default', 'mvgrl_default', 'selfgnn_default',
                 'sublime_default', 'bgrl_default', 'vgaer_default', 'cagc_default']
default_folder = './results//legacy_resultsdefault_results/'

#create_all_paper_figures(datasets, algorithms, metrics, seeds, folder, default_folder, default_algos)

# creates a tikz figure to show someone
#fig, ax = plt.subplots(1, 1, figsize=(15, 15))
#ax = create_rand_dist_fig(ax, r'Framework Rank Distinction Coefficient $\Omega(\mathcal{\hat{T}}_{(hpo)}) : $', datasets, algorithms, metrics, seeds, folder, calc_ave_first=True, set_legend=True)
#fig.tight_layout()
#fig.savefig(f'{ugle_path}/figures/tkiz_fig.png', bbox_inches='tight')


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



    mod_results = np.asarray(mod_results)
    con_results = np.asarray(con_results)
    if return_df:
        return mod_results, con_results, df
    else: 
        return mod_results, con_results


def print_dataset_table(datasets, algorithms, folder):
    # extract results
    for dataset in datasets:
        print(dataset, end = ' ')
        mod_results, con_results = extract_results([dataset], algorithms, folder)
        for x, y in [[mod_results[:, 0], mod_results[:, 1]], [mod_results[:, 0], mod_results[:, 2]], [con_results[:, 0], con_results[:, 1]], [con_results[:, 0], con_results[:, 2]]]:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            print(f'& {r_value:.2f}', end=' ')
        print('')

def print_algo_table(datasets, algorithms, folder):
    for algorithm in algorithms:
        print(algorithm, end = ' ')
        mod_results, con_results = extract_results(datasets, [algorithm], folder)
        for x, y in [[mod_results[:, 0], mod_results[:, 1]], [mod_results[:, 0], mod_results[:, 2]], [con_results[:, 0], con_results[:, 1]], [con_results[:, 0], con_results[:, 2]]]:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            print(f'& {r_value:.2f}', end=' ')
        print('')


def kendall_w(expt_ratings):
    if expt_ratings.ndim!=2:
        raise 'ratings matrix must be 2-dimensional'
    m = expt_ratings.shape[0] # raters
    n = expt_ratings.shape[1] # items rated
    denom = m**2*(n**3-n)
    rating_sums = np.sum(expt_ratings, axis=0)
    S = n*np.var(rating_sums)
    return 12*S/denom


def rank_values(scores):
    argsort_array = scores.argsort().argsort()
    ranks_array = np.empty_like(argsort_array)
    ranks_array[argsort_array] = np.arange(len(scores))
    return ranks_array


def unsupervised_prediction_graph(datasets, algorithms, seeds, folder, title, marker_style=None):
    # extract results 
    if "q4" in title:
        extract_validation = True
        return_df = False
    else: 
        extract_validation = False
        if marker_style: 
            return_df = True
        else: 
            return_df = False
    if return_df:
        mod_results, con_results, df = extract_results(datasets, algorithms, folder, extract_validation=extract_validation, return_df=return_df)
    else:
        mod_results, con_results = extract_results(datasets, algorithms, folder, extract_validation=extract_validation, return_df=return_df)
    
    # compute W order
    different_comparisons = [mod_results[:, 0], mod_results[:, 1], mod_results[:, 2], con_results[:, 0], con_results[:, 1], con_results[:, 2]]
    testnames = ["Modularity", "Modularity F1", "Modularity NMI", "Conductance", "Conductance F1", "Conductance NMI"]
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
    print(f"Overall W Order: {np.mean(total_w_order):.3f} +- {np.std(total_w_order):.3f}")
    # the W order of each comparison 
    indv_tests = np.array([[0, 1], [0, 2], [3, 4], [3, 5]])
    moretestnames = [testnames[i] for i in indv_tests[:, 1]]
    more_w_orders = [np.mean(np.asarray(total_w_order[i])) for i in indv_tests]
    for n, w in zip(moretestnames, more_w_orders):
        print(f"W Order {n}: {w:.2f}")
   
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
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
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        print(f"Coefficient of Determination (R^2): {r_value:.2f}")
        x_space = np.linspace(np.min(x), np.max(x), 200)
        y_line = (x_space * slope) + intercept

        # Fit a quadratic 
        coefficients = np.polyfit(x, y, 2)
        poly = np.poly1d(coefficients)
        # Calculate predicted values
        predicted_y = poly(x)
        # Calculate total sum of squares
        total_var = np.sum((y - np.mean(y)) ** 2)
        # Calculate residual sum of squares
        residual_var = np.sum((y - predicted_y) ** 2)
        # Calculate variance explained
        variance_explained = 1 - (residual_var / total_var)
        variance_explained = np.round(variance_explained, 3)
        print(f"Variance Explained by Quadratic: {variance_explained}")
        y_line_quad = poly(x_space)

        # spearmans  
        spearman_stats = spearmanr(a=x, b=y)
        print(f"Spearmans Correlation Coefficient: {spearman_stats.correlation:.2f}")
        
        if marker_style == 'algorithms':
            markers = ['o', 's', '^', 'D', 'p', '*', 'H', '+', 'X', '$f$']
            markers = markers[:len(algorithms)]
            for opt_in in range(len(markers)):
                df_axis = df[(df['Algorithm'] == algorithms[opt_in]) & (df['A_Metric'] == x_label) & (df['B_Metric'] == y_label)]
                ax.scatter(df_axis['A_Metric_Value'], df_axis['B_Metric_Value'], color='tab:blue', s=15, marker=markers[opt_in], label=algorithms[opt_in])

        elif marker_style == 'datasets':
            markers =  ['$ci$', '$co$', '$tx$', '$db$', '$wc$', '$cn$']
            for opt_in in range(len(markers)): 
                df_axis = df[(df['Dataset'] == datasets[opt_in]) & (df['A_Metric'] == x_label) & (df['B_Metric'] == y_label)]
                ax.scatter(df_axis['A_Metric_Value'], df_axis['B_Metric_Value'], color='tab:blue', s=15, marker=markers[opt_in], label=datasets[opt_in])

        elif marker_style == 'seeds':
            markers = ['o', 's', '^', 'D', 'p', '*', 'H', '+', 'X', '$f$']
            markers = markers[:len(seeds)]
            for opt_in in range(len(markers)): 
                df_axis = df[(df['Seed'] == seeds[opt_in]) & (df['A_Metric'] == x_label) & (df['B_Metric'] == y_label)]
                ax.scatter(df_axis['A_Metric_Value'], df_axis['B_Metric_Value'], color='tab:blue', s=15, marker=markers[opt_in], label=seeds[opt_in])
        else:
            ax.scatter(x, y, color='tab:blue', s=15, label='Test Data')

        ax.plot(x_space, y_line, color='tab:red', label=r'Linear Fit, $R^2:$' + f'{r_value:.2f}', linewidth=3)
        ax.plot(x_space, y_line_quad, color="tab:orange", label=r'Quadratic Fit, $R^2:$' + f'{variance_explained:.2f}', linewidth=3)
        if i == 0:
            ax.set_ylabel(y_label, fontsize=20)
        if i == 2:
            ax.set_ylabel(y_label, fontsize=20) 
            ax.set_xlabel(x_label, fontsize=20)
        if i == 3:
            ax.set_xlabel(x_label, fontsize=20)
        
        ax.set_title(f"{x_label}" + r' $\rightarrow$ '+ f"{y_label} W: {W_order:.2f}", fontsize=20)
        ax.legend(loc='best', fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=18)
       
    
    print_algo_table(datasets, algorithms, folder)
    print_dataset_table(datasets, algorithms, folder)

    fig.suptitle(title, fontsize=24)
    plt.tight_layout()
    plt.savefig(f'./figures/unsupervised_limit/{os.path.basename(os.path.normpath(folder))}.png')


def synthetic_evaluation(datasets, algorithms, folder):
    x_axis_names = [algo.split("_")[0] for algo in algorithms]
    x_axis_points = np.arange(len(x_axis_names))
    bar_width = 1/4
    # extract results
    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        dataset = datasets[i]
        dataset_name = r"$A:$" + dataset.split("_")[1] + ' ' + r"$X:$" + dataset.split("_")[2]
        ax.set_title(dataset_name)

        mod_f1 = []
        mod_nmi = []
        con_f1 = []
        con_nmi = []

        for algo in algorithms:
            mod_results, con_results = extract_results([dataset], [algo], folder)
            mod_f1.append(np.mean(mod_results[:, 1]))
            mod_nmi.append(np.mean(mod_results[:, 2]))
            con_f1.append(np.mean(con_results[:, 1]))
            con_nmi.append(np.mean(con_results[:, 2]))

        ax.bar(x_axis_points, mod_f1, width=bar_width, facecolor="C3", linewidth=0, label='mod_f1')
        ax.bar(x_axis_points + 1/4, mod_nmi, width=bar_width, facecolor="C3", linewidth=0, alpha=0.5, label='mod_nmi')
        ax.bar(x_axis_points + 1/2, con_f1, width=bar_width, facecolor="C0", linewidth=0, label='con_f1')
        ax.bar(x_axis_points + 3/4, con_nmi, width=bar_width, facecolor="C0", linewidth=0, alpha=0.5, label='con_nmi')
               
        ax.set_xticks(x_axis_points - 0.5 * bar_width)
        ax.set_xticklabels(x_axis_names, ha='left', rotation=-45, position=(-0.5, 0.0))
        ax.set_axisbelow(True)
        ax.set_ylim(0.0, 1.0)
        ax.axhline(y=0.5, color='k', linestyle='-')
        if i == 5:
            ax.legend(loc='best')


    fig.suptitle("Default Hyperparameters on Synthetic Data")
    plt.tight_layout()
    plt.savefig(f'./figures/unsupervised_limit/{os.path.basename(os.path.normpath(folder))}.png')


q1_folder = './results/unsupervised_limit/default_q1/'
q2_folder = './results/unsupervised_limit/hpo_q2/'
q5_folder = './results/unsupervised_limit/synth_default_q5/'
q5_folder1 = './results/unsupervised_limit/33_default_q4/'
q5_folder2 = './results/unsupervised_limit/66_default_q4/'

seeds = [42, 24, 976, 12345, 98765, 7, 856, 90, 672, 785]
datasets = ['citeseer', 'cora', 'texas', 'dblp', 'wisc', 'cornell']
synth_datasets = ['synth_disjoint_disjoint_2', 'synth_disjoint_random_2', 'synth_disjoint_complete_2',
                    'synth_random_disjoint_2', 'synth_random_random_2', 'synth_random_complete_2',
                    'synth_complete_disjoint_2', 'synth_complete_random_2', 'synth_complete_complete_2']

default_algorithms = ['dgi_default', 'daegc_default', 'dmon_default', 'grace_default', 'sublime_default', 'bgrl_default', 'vgaer_default']
algorithms = ['dgi', 'daegc', 'dmon', 'grace', 'sublime', 'bgrl', 'vgaer']


#unsupervised_prediction_graph(datasets, default_algorithms, seeds, q1_folder, title="Default Hyperparameters")

#unsupervised_prediction_graph(datasets, default_algorithms, seeds, q1_folder, title="Default Hyperparameters")
#unsupervised_prediction_graph(datasets, default_algorithms, seeds, q5_folder2, title="q4: 66% of the data")
#unsupervised_prediction_graph(datasets, default_algorithms, seeds, q5_folder1, title="q4: 33% of the data")

#synthetic_evaluation(synth_datasets, default_algorithms, q5_folder)

#unsupervised_prediction_graph(datasets, algorithms, q2_folder, title="Hyperparameter Optimisation")
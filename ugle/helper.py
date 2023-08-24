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


def calculate_ranking_performance(result_object, datasets, metrics, seeds, scale_metrics=False, calc_ave_first=False):
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
                if scale_metrics:
                    ranking_of_algorithms = scale_metric_values(ranking_of_algorithms, metric_values, metric_name)
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
                    if scale_metrics:
                        ranking_of_algorithms = scale_metric_values(ranking_of_algorithms, metric_values, metric_name)
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


def create_dataset_landscape_fig(ax, title, datasets, algorithms, metrics, seeds, folder, scale_metrics=False, calc_ave_first=False):
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
    ranking_object = calculate_ranking_performance(result_object, datasets, metrics, seeds, scale_metrics=scale_metrics, calc_ave_first=calc_ave_first)
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
    cbar.ax.set_yticklabels(best_algo_list, fontsize=14, ha='left', rotation=-40)
    # add dataset locations
    ax.scatter(clustering, closeness, marker='x', s=20, color='black', label='datasets')
    # add nice visual elements
    ax.set_xlabel('Clustering Coefficient', fontsize=16)
    ax.set_ylabel('Closeness Centrality', fontsize=16)
    ax.set_title(title, fontsize=22, pad=15)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18)
    return ax


def create_ranking_charts(datasets: list, algorithms: list, metrics: list, seeds: list, folder: str, title_name: str, 
    scale_metrics: bool=True, calc_ave_first: bool=False, set_legend: bool=True, ax=None, ax1=None, fig=None, fig1=None):

    # init axis
    if not ax and not fig:
        fig, ax = plt.subplots(figsize=(6, 6))
    elif (ax and not fig) or (fig and not ax):
        print('please provide both fig and ax or neither')

    if not ax1:
        fig1, ax1 = plt.subplots(figsize=(6, 6))
    elif (ax1 and not fig1) or (fig1 and not ax1):
        print('please provide both fig1 and ax1 or neither')

    # fetch results
    result_object = make_test_performance_object(datasets, algorithms, metrics, seeds, folder)
    # calculate ranking 
    ranking_object = calculate_ranking_performance(result_object, datasets, metrics, seeds, scale_metrics=scale_metrics, calc_ave_first=calc_ave_first)
    # calculate overrall ranking and sensitivity + order
    if calc_ave_first:
        overrall_ranking = np.mean(ranking_object, axis=(0, 2))
        # calculate the ranking averages 
        ranking_over_metrics = np.mean(ranking_object, axis=0)
        ranking_over_datasets = np.mean(ranking_object, axis=2)
    else:
        overrall_ranking = np.mean(ranking_object, axis=(0, 2, 3))
        # calculate the ranking averages 
        ranking_over_metrics = np.mean(ranking_object, axis=(0, 3))
        ranking_over_datasets = np.mean(ranking_object, axis=(2, 3))

    algorithms = np.array(algorithms)
    ranking_order = np.argsort(overrall_ranking)
    algo_labels = algorithms[ranking_order]
    algo_labels = [albel.split("_")[0] for albel in algo_labels]

    # plot the by metric ranking
    for rank_on_metric_ave_over_dataset, metric in zip(ranking_over_metrics.T, metrics):
        ax.plot(algo_labels, rank_on_metric_ave_over_dataset[ranking_order],
                marker="o", label=metric, alpha=0.5, zorder=20)

    # plot the overrall ranking
    ax.scatter(algo_labels, overrall_ranking[ranking_order],
               marker="x", c='black', s=20, label='average \noverall rank', zorder=100)

    # set legend
    if set_legend:
        ax.legend(loc='upper center', fontsize=14, ncol=2, bbox_to_anchor=(0.5, -0.35))
    # configure y axis
    ax.set_ylim(0, 10.5)
    ax.set_ylabel('Algorithm ranking\naveraged over dataset', fontsize=14)
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
    ax.set_title(f'{title_name}', fontsize=20)
    plt.tight_layout()

    # plot the by dataset ranking
    colours = plt.get_cmap('tab20').colors
    for rank_on_dataset_ave_over_metric, dataset, c in zip(ranking_over_datasets, datasets, colours):
        ax1.plot(algo_labels, rank_on_dataset_ave_over_metric[ranking_order],
                 marker="o", label=dataset, alpha=0.5, zorder=20, c=c)
    # plot the overrall ranking
    ax1.scatter(algo_labels, overrall_ranking[ranking_order],
                marker="x", c='black', s=20, label='average \noverall rank', zorder=100)
    # set legend
    if set_legend:
        ax1.legend(loc='upper center', fontsize=14, ncol=4, bbox_to_anchor=(0.5, -0.35))
    # set y axis
    ax1.set_ylim(0, 10.5)
    ax1.set_ylabel('Algorithm ranking\naveraged over metric', fontsize=14)
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
    ax1.set_title(f'{title_name}', fontsize=20)
    plt.tight_layout()

    return ax, ax1


def create_rand_dist_fig(ax, title, datasets, algorithms, metrics, seeds, folder, scale_metrics=False, calc_ave_first=False, set_legend=False):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    x_axis = np.arange(0, len(algorithms), 0.001)

    result_object = make_test_performance_object(datasets, algorithms, metrics, seeds, folder)
    ranking_object = calculate_ranking_performance(result_object, datasets, metrics, seeds, calc_ave_first=calc_ave_first, scale_metrics=scale_metrics)

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

    # calculate amount of overlap
    out = 0.
    for i, _ in enumerate(algorithms):
        for j, _ in enumerate(algorithms):
            if i != j:
                out += wasserstein_distance(all_ranks_per_algo[i], all_ranks_per_algo[j])
                
    # calculate averages
    div_out = (len(algorithms) * (len(algorithms) -1))/2
    ave_overlap = str(round(out/div_out, 3))

    with plt.xkcd():
        # plot the lines
        for j, algo_ranks in enumerate(all_ranks_per_algo):
            kde = gaussian_kde(algo_ranks)
            y_axis = kde.evaluate(x_axis)
            ax.plot(x_axis, y_axis, label=algorithms[j], zorder=10)

        if set_legend:
            ax.legend(loc='best', fontsize=16, ncol=3)
            #ax.legend(loc='upper center', fontsize=14, ncol=3, bbox_to_anchor=(0.5, -0.35))
            #ax.set_xlabel('algorithm rank distribution over all tests', fontsize=18)

        #ax.set_ybound(0, 3)
        ax.set_xbound(1, 10)
        ax.set_ylabel('probability density', fontsize=18)
        ax.set_xlabel(r'kde estimatation of rank distribution $f_{j}(r)$', fontsize=18)

        #ax.text(0.4, 0.85, ave_overlap_text, fontsize=20, transform=ax.transAxes, zorder=1000)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        ax.set_title(title + ave_overlap, fontsize=20)
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
    fig.savefig(f"{ugle_path}/figures/new_hpo_investigation.png", bbox_inches='tight')
    return


def create_algo_selection_on_dataset_landscape(datasets, algorithms, default_algos, metrics, seeds, folder, default_folder, titles):

    fig, ax = plt.subplots(2, 2, figsize=(15, 7.5))

    titles[0] += r' $(\mathcal{\hat{T}}_{(dhp)})$'
    titles[1] += r' $(\mathcal{\hat{T}}_{(hpo)})$'
    titles[2] += r' $(\mathcal{T}_{(dhp)})$'
    titles[3] += r' $(\mathcal{T}_{(hpo)})$'

    ax[0, 0] = create_dataset_landscape_fig(ax[0, 0], titles[0], datasets, default_algos, metrics, seeds, default_folder, calc_ave_first=True)
    ax[0, 1] = create_dataset_landscape_fig(ax[0, 1], titles[1], datasets, algorithms, metrics, seeds, folder, calc_ave_first=True)
    ax[1, 0] = create_dataset_landscape_fig(ax[1, 0], titles[2], datasets, default_algos, metrics, seeds, default_folder)
    ax[1, 1] = create_dataset_landscape_fig(ax[1, 1], titles[3], datasets, algorithms, metrics, seeds, folder)
    
    fig.tight_layout()
    fig.savefig(f"{ugle_path}/figures/new_dataset_landscape_comparison.png")
    return


def create_comparison_figures(datasets: list, algorithms: list, metrics: list, seeds: list, folder: str,
                              default_algos: list, default_folder: str, titles: list):
    # create holder figure
    nrows, ncols = 2, 2
    fig0, axs0 = plt.subplots(nrows, ncols, figsize=(15, 7.5))

    # create holder figure
    nrows, ncols = 2, 2
    fig1, axs1 = plt.subplots(nrows, ncols, figsize=(15, 7.5))

    titles[0] += r' $(\mathcal{\hat{T}}_{(dhp)})$'
    titles[1] += r' $(\mathcal{\hat{T}}_{(hpo)})$'
    titles[2] += r' $(\mathcal{T}_{(dhp)})$'
    titles[3] += r' $(\mathcal{T}_{(hpo)})$'

    axs0[0, 0], axs1[0, 0] = create_ranking_charts(datasets, default_algos, metrics, seeds, default_folder,
                                           title_name=titles[0],
                                           scale_metrics=False, calc_ave_first=True, set_legend=False, ax=axs0[0, 0], ax1=axs1[0, 0], fig=fig0,
                                           fig1=fig1)
    axs0[0, 1], axs1[0, 1] = create_ranking_charts(datasets, algorithms, metrics, seeds, folder, title_name=titles[1],
                                           scale_metrics=False, calc_ave_first=True, set_legend=False, ax=axs0[0, 1], ax1=axs1[0, 1], fig=fig0,
                                           fig1=fig1)
    axs0[1, 0], axs1[1, 0] = create_ranking_charts(datasets, default_algos, metrics, seeds, default_folder,
                                           title_name=titles[2],
                                           scale_metrics=False, set_legend=False, ax=axs0[1, 0], ax1=axs1[1, 0], fig=fig0,
                                           fig1=fig1)
    axs0[1, 1], axs1[1, 1] = create_ranking_charts(datasets, algorithms, metrics, seeds, folder, title_name=titles[3],
                                           scale_metrics=False, set_legend=True, ax=axs0[1, 1], ax1=axs1[1, 1], fig=fig0,
                                           fig1=fig1)

    fig0.tight_layout()
    fig1.tight_layout()

    fig0.savefig(f"{ugle_path}/figures/new_ranking_comparison_metrics.png", bbox_inches='tight')
    fig1.savefig(f"{ugle_path}/figures/new_ranking_comparison_datasets.png", bbox_inches='tight')
    return


def create_rand_dist_comparison(datasets: list, algorithms: list, metrics: list, seeds: list, folder: str,
                                default_algos: list, default_folder: str, titles: list):

    # create holder figure
    nrows, ncols = 2, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 7.5))

    titles[0] += r' $\Omega(\mathcal{\hat{T}}_{(dhp)})$ : '
    titles[1] += r' $\Omega(\mathcal{\hat{T}}_{(hpo)})$ : '
    titles[2] += r' $\Omega(\mathcal{T}_{(dhp)})$ : '
    titles[3] += r' $\Omega(\mathcal{T}_{(hpo)})$ : '

    ax[0, 0] = create_rand_dist_fig(ax[0, 0], titles[0], datasets, default_algos, metrics, seeds, default_folder, calc_ave_first=True)
    ax[0, 1] = create_rand_dist_fig(ax[0, 1], titles[1], datasets, algorithms, metrics, seeds, folder, calc_ave_first=True)
    ax[1, 0] = create_rand_dist_fig(ax[1, 0], titles[2], datasets, default_algos, metrics, seeds, default_folder)
    ax[1, 1] = create_rand_dist_fig(ax[1, 1], titles[3], datasets, algorithms, metrics, seeds, folder, set_legend=True)

    fig.tight_layout()
    fig.savefig(f'{ugle_path}/figures/new_rand_dist_comparison.png', bbox_inches='tight')
    return


def create_all_paper_figures(datasets, algorithms, metrics, seeds, folder, default_folder, default_algos):
    titles = ['a) Default HPs w/ AveSeed', 'b) HPO w/ AveSeed', 'c) Default HPs w/ SeedRanking', 'd) HPO w/ SeedRanking']
    create_rand_dist_comparison(datasets, algorithms, metrics, seeds, folder, default_algos, default_folder, deepcopy(titles))
    create_algo_selection_on_dataset_landscape(datasets, algorithms, default_algos, metrics, seeds, folder, default_folder, deepcopy(titles))
    create_comparison_figures(datasets, algorithms, metrics, seeds, folder, default_algos, default_folder, deepcopy(titles))
    create_big_figure(datasets, algorithms, folder, default_algos, default_folder)

    return


#algorithms = ['daegc', 'dgi', 'dmon', 'grace', 'mvgrl', 'selfgnn', 'sublime', 'bgrl', 'vgaer', 'cagc']
#datasets = ['cora', 'citeseer', 'dblp', 'bat', 'eat', 'texas', 'wisc', 'cornell', 'uat', 'amac', 'amap']
#metrics = ['nmi', 'modularity', 'f1', 'conductance']
#folder = './progress_results/'
#seeds = [42, 24, 976, 12345, 98765, 7, 856, 90, 672, 785]
#default_algos = ['daegc_default', 'dgi_default', 'dmon_default', 'grace_default', 'mvgrl_default', 'selfgnn_default',
#                 'sublime_default', 'bgrl_default', 'vgaer_default', 'cagc_default']
#default_folder = './new_default/'

#create_all_paper_figures(datasets, algorithms, metrics, seeds, folder, default_folder, default_algos)

# creates a tikz figure to show someone
#fig, ax = plt.subplots(1, 1, figsize=(15, 15))
#ax = create_rand_dist_fig(ax, r'Framework Rank Distinction Coefficient $\Omega(\mathcal{\hat{T}}_{(hpo)}) : $', datasets, algorithms, metrics, seeds, folder, calc_ave_first=True, set_legend=True)
#fig.tight_layout()
#fig.savefig(f'{ugle_path}/figures/tkiz_fig.png', bbox_inches='tight')

def rank_values(scores):
    ranks = []
    for score in scores:
       ranks.append(scores.index(score) + 1)
    return ranks


def calculate_ranking_coefficient_over_dataset(result_object, algorithms, dataset_idx, metrics):
    ranking_object = np.mean(result_object, axis=-1)[dataset_idx, :, :]
    n_ranks = len(metrics)
    all_ranks_per_algo = np.zeros(shape=(len(algorithms), n_ranks))

    for m, metric_name in enumerate(metrics):
        if metric_name != 'conductance':
            all_ranks_per_algo[:, m] = rank_values(np.flip(np.sort(ranking_object[:, m])).tolist())
        else:
            all_ranks_per_algo[:, m] = rank_values(np.sort(ranking_object[:, m]).tolist())
       
    coeff = 0.
    for i, _ in enumerate(algorithms):
        for j, _ in enumerate(algorithms):
            if i != j:
                coeff += wasserstein_distance(all_ranks_per_algo[i], all_ranks_per_algo[j])
    
    # calculate averages
    div_out = (len(algorithms) * (len(algorithms) -1 ))/2
    coeff = round(coeff/div_out, 3)  

    return coeff 


def create_synth_figure():
    synth_datasets_2 = ['synth_disjoint_disjoint_2', 'synth_disjoint_random_2', 'synth_disjoint_complete_2',
                    'synth_random_disjoint_2', 'synth_random_random_2', 'synth_random_complete_2',
                    'synth_complete_disjoint_2', 'synth_complete_random_2', 'synth_complete_complete_2']
    
    synth_datasets = ['synth_disjoint_disjoint_4', 'synth_disjoint_random_4', 'synth_disjoint_complete_4',
                    'synth_random_disjoint_4', 'synth_random_random_4', 'synth_random_complete_4',
                    'synth_complete_disjoint_4', 'synth_complete_random_4', 'synth_complete_complete_4']
    synth_algorithms = ['dgi_default', 'daegc_default', 'dmon_default', 'grace_default', 'sublime_default', 'bgrl_default', 'vgaer_default']
    synth_folder = './synth_defaults/'
    metrics = ['nmi', 'modularity', 'f1', 'conductance']
    seeds = [42, 24, 976, 12345, 98765, 7, 856, 90, 672, 785]
    # fetch results
    result_object = make_test_performance_object(synth_datasets, synth_algorithms, metrics, seeds, synth_folder)

    # create graph subfigures 
    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        coeff = calculate_ranking_coefficient_over_dataset(result_object, synth_algorithms, i, metrics)

        title = (" ").join(synth_datasets[i].split("_")[1:3])
        title = 'adj: ' + synth_datasets[i].split("_")[1] + " feat: " + synth_datasets[i].split("_")[2]
        ax.set_title(f"{title}: {str(coeff)}" )

        alt_colours = ['C3', 'C0', 'C2', 'C1']

        bar_width = 1 / 4
        x_axis_names = np.arange(len(synth_algorithms))

        # plot results
        for m, metric in enumerate(metrics):
            metric_vals = []
            metric_std = []
            for a, algo in enumerate(synth_algorithms):
                metric_vals.append(np.mean(result_object[i, a, m, :]))
                metric_std.append(np.std(result_object[i, a, m, :]))
            ax.bar(x_axis_names + (m * bar_width), metric_vals, yerr=metric_std, 
                   width=bar_width, facecolor=alt_colours[m], alpha=0.9, linewidth=0, label=metric)
            
        ax.set_xticks(x_axis_names - 0.5 * bar_width)
        ax.set_xticklabels(synth_algorithms, ha='left', rotation=-45, position=(-0.5, 0.0))
        ax.set_axisbelow(True)
        ax.set_ylim(0.0, 1.0)
        ax.axhline(y=0.5, color='k', linestyle='-') 


    plt.tight_layout()
    plt.savefig('synth_4_default.png', bbox_inches='tight')
    print('done')



def create_trainpercent_figure():
    seeds = [42, 24, 976, 12345, 98765, 7, 856, 90, 672, 785]
    datasets = ['citeseer', 'cora', 'dblp', 'texas', 'wisc', 'cornell', 'amac']
    algorithms = ['dgi', 'daegc', 'dmon', 'grace', 'sublime', 'bgrl', 'vgaer']
    percents = ['01', '03', '05', '07', '09']
    percents_labels = ['0.1', '0.3', '0.5', '0.7', '0.9']
    metrics = ['nmi', 'modularity', 'f1', 'conductance']
    folder = './revamp_train_percent/'

    nrows, ncols = len(algorithms), len(datasets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5, 10))

    res_objs = []
    for pct in percents:
        res_objs.append(make_test_performance_object(datasets, algorithms, metrics, seeds, folder + pct + '/'))
    res_objs = np.stack(res_objs, axis=0)

    for d, dataset in enumerate(datasets):
        for a, algo in enumerate(algorithms):
            ax = axes[a, d]
            ax.set_title(f'{algo} -- {dataset}')

            alt_colours = ['C3', 'C0', 'C2', 'C1']
            bar_width = 1 / 4
            x_axis_names = np.arange(len(percents))

            # plot results
            for m, metric in enumerate(metrics):
                metric_vals = []
                metric_std = []
                for p in range(len(percents)):
                    metric_vals.append(np.mean(res_objs[p, d, a, m, :]))
                    metric_std.append(np.std(res_objs[p, d, a, m, :]))
                ax.bar(x_axis_names + (m * bar_width), metric_vals, yerr=metric_std, 
                    width=bar_width, facecolor=alt_colours[m], alpha=0.9, linewidth=0, label=metric)
                
            ax.set_xticks(x_axis_names - 0.5 * bar_width)
            ax.set_xticklabels(percents_labels, ha='left', rotation=-45, position=(-0.5, 0.0))
            ax.set_axisbelow(True)
            ax.set_ylim(0.0, 1.0)
                

    plt.tight_layout()
    plt.show()
    print('done')


#create_trainpercent_figure()
#create_synth_figure()

def calc_correlation():
    seeds = [42, 24, 976, 12345, 98765, 7, 856, 90, 672, 785]
    datasets = ['citeseer', 'cora', 'dblp', 'texas', 'wisc', 'cornell']
    algorithms = ['dgi_default', 'daegc_default', 'dmon_default', 'grace_default', 'sublime_default', 'bgrl_default', 'vgaer_default']
    metrics = ['nmi', 'modularity', 'f1', 'conductance']
    folder = './results/q1_default_predict_super/'

    mod_results = []
    con_results = []

    for dataset in datasets:
        for algo in algorithms:
            filename = f"{dataset}_{algo}.pkl"
            file_found = search_results(folder, filename)
            if file_found:
                result = pickle.load(open(file_found, "rb"))
            
            for seed_result in result.results:
                for metric_result in seed_result.study_output:
                    if 'modularity' in metric_result.metrics:
                       mod_results.append([metric_result.results['modularity'], metric_result.results['f1'], metric_result.results['nmi']])

                    if 'conductance' in metric_result.metrics:
                       con_results.append([metric_result.results['conductance'], metric_result.results['f1'], metric_result.results['nmi']])

    mod_results = np.asarray(mod_results)
    con_results = np.asarray(con_results)


    
    mod_f1 = np.corrcoef(mod_results[:, 0], mod_results[:, 1])[0,1]
    mod_nmi = np.corrcoef(mod_results[:, 0], mod_results[:, 2])[0,1]
    con_f1 = np.corrcoef(con_results[:, 0], con_results[:, 1])[0,1]
    con_nmi = np.corrcoef(con_results[:, 0], con_results[:, 2])[0,1]
    print('Correlation Coefficients UnsuperPredict: ')
    print(f'Modularity --> F1: {mod_f1:.3f}')
    print(f'Modularity --> NMI: {mod_nmi:.3f}')
    print(f'Conductance --> F1: {con_f1:.3f}')
    print(f'Conductance --> NMI: {con_nmi:.3f}')


    x = mod_results[:, 0]
    y = mod_results[:, 1]

    # Fit a 2nd-degree polynomial
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

    print("Coefficients:", coefficients)
    print("Variance Explained:", variance_explained)

    # Plotting
    plt.scatter(x, y, label='Data Points')
    x_space = np.linspace(np.min(x), np.max(x), 200)
    predicted_y = poly(x_space)
    plt.plot(x_space, predicted_y, label='Fitted Polynomial', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Fitted 2nd-Degree Polynomial')
    plt.legend()
    plt.grid()
    plt.show()


    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    x  = mod_results[:, 0].reshape(-1, 1)
    y = mod_results[:, 1].reshape(-1, 1)
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    gaussian_process.fit(x, y)
    X_new = np.linspace(np.min(x), np.max(x), 150)[:, np.newaxis]
    mean_prediction, std_prediction = gaussian_process.predict(X_new, return_std=True)
    plt.scatter(x, y, label="data", c="blue")
    plt.plot(X_new, mean_prediction, label="Mean prediction", c='orange')
    plt.fill_between(
        X_new.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
        c='orange'
    )
    plt.show()


    X_new = np.linspace(0, 5, 100)[:, np.newaxis]
    Y_pred, sigma = gaussian_process.predict(X_new, return_std=True)
    plt.figure()
    plt.scatter(x, y, c='r', label='Data')
    plt.plot(X_new, Y_pred, 'b', label='GP Prediction')
    plt.fill_between(X_new.ravel(), Y_pred - 1.96 * sigma, Y_pred + 1.96 * sigma, alpha=0.2, color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    plt.scatter(x, y, label='Data Points')
    plt.plot(x, cc * x, color='red', label='Correlation Line')
    plt.xlabel('mod')
    plt.ylabel('f1')
    plt.title('Correlation and Line of Best Fit between mod and f1')
    plt.legend()
    plt.grid(True)
    plt.show()
    print('stop')

calc_correlation()


def calc_synth_results():
    seeds = [42, 24, 976, 12345, 98765, 7, 856, 90, 672, 785]
    datasets = ['synth_disjoint_disjoint_2', 'synth_disjoint_random_2', 'synth_disjoint_complete_2',
                    'synth_random_disjoint_2', 'synth_random_random_2', 'synth_random_complete_2',
                    'synth_complete_disjoint_2', 'synth_complete_random_2', 'synth_complete_complete_2']
    algorithms = ['dgi_default', 'daegc_default', 'dmon_default', 'grace_default', 'sublime_default', 'bgrl_default', 'vgaer_default']
    metrics = ['nmi', 'modularity', 'f1', 'conductance']
    folder = './results/q2_synth_default/'
    
    mod_results = []
    con_results = []

    for dataset in datasets:
        for algo in algorithms:
            filename = f"{dataset}_{algo}.pkl"
            file_found = search_results(folder, filename)
            if file_found:
                result = pickle.load(open(file_found, "rb"))

            for seed_result in result.results:
                for metric_result in seed_result.study_output:
                    if 'modularity' in metric_result.metrics:
                       mod_results.append([metric_result.results['modularity'], metric_result.results['f1'], metric_result.results['nmi']])

                    if 'conductance' in metric_result.metrics:
                       con_results.append([metric_result.results['conductance'], metric_result.results['f1'], metric_result.results['nmi']])

    mod_results = np.asarray(mod_results)
    con_results = np.asarray(con_results)
    mod_f1 = np.corrcoef(mod_results[:, 0], mod_results[:, 1])[0,1]
    mod_nmi = np.corrcoef(mod_results[:, 0], mod_results[:, 2])[0,1]
    con_f1 = np.corrcoef(con_results[:, 0], con_results[:, 1])[0,1]
    con_nmi = np.corrcoef(con_results[:, 0], con_results[:, 2])[0,1]
    print('Correlation Coefficients SYNTH: ')
    print(f'Modularity --> F1: {mod_f1:.3f}')
    print(f'Modularity --> NMI: {mod_nmi:.3f}')
    print(f'Conductance --> F1: {con_f1:.3f}')
    print(f'Conductance --> NMI: {con_nmi:.3f}')


calc_synth_results()
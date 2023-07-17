import subprocess
import os
ugle_path = os.path.dirname(os.path.realpath(__file__))


def search_results(folder, filename):

    for root, dirs, files in os.walk(f'{ugle_path}/{folder}'):
        if filename in files:
            return os.path.join(root, filename)
    return None


def attempt_file_transfer(filename, folder):
    command = f"scp -r blue_server:~/ugle/{folder}{filename} {folder}"

    res = subprocess.call(command, shell=True)
    if res != 1:
        print(f'File Transferred: {filename}')
    else:
        print(f'No Result: {filename}')

    return


def update_progress_results(datasets, algorithms, folder):
    for dataset in datasets:
        for algo in algorithms:
            filename = f"{dataset}_{algo}.pkl"
            result = search_results(folder, filename)
            if result:
                print(f'Found: {filename}')
            else:
                attempt_file_transfer(filename, folder)

    return


algorithms = ['daegc', 'dgi', 'dmon', 'grace', 'mvgrl', 'selfgnn', 'sublime', 'bgrl', 'vgaer', 'cagc']
datasets = ['cora', 'citeseer', 'dblp', 'bat', 'eat', 'texas', 'wisc', 'cornell', 'uat', 'amac', 'amap']
folder = './updated_results/'
update_progress_results(datasets, algorithms, folder)

default_algos = ['daegc_default', 'dgi_default', 'dmon_default', 'grace_default', 'mvgrl_default', 'selfgnn_default',
                 'sublime_default', 'bgrl_default', 'vgaer_default', 'cagc_default']
default_folder = './new_default/'
update_progress_results(datasets, default_algos, default_folder)
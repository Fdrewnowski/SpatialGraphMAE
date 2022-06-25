import argparse
import copy

from pathlib import Path
import networkx as nx
import osmnx as ox
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='visualise_attr')
    parser.add_argument('-d', '--directory', type=str, default='./data_raw/')
    parser.add_argument('-n', '--name', type=str, default='attr_plot.png')
    return parser.parse_args()

def count_key_attributes(directory):
    found_files = list(Path(directory).glob('*.xml'))
    graph_counter = []
    for path in tqdm(found_files):
        selected_keys = {'oneway': 0, 'lanes': 0, 'highway': 0, 'maxspeed': 0,
             'length': 0, 'access': 0, 'bridge': 0, 'junction': 0,
             'width': 0, 'service': 0, 'tunnel': 0}
        iterator = 0
        graph_nx = ox.io.load_graphml(path)
        for edge in graph_nx.edges():
            for connection in graph_nx[edge[0]][edge[1]].keys():
                iterator += 1
                for key, val in graph_nx[edge[0]][edge[1]][connection].items():
                    if key in selected_keys.keys():
                        selected_keys[key] += 1
        selected_keys['number_of_conections'] = iterator
        graph_counter.append(selected_keys)
    return graph_counter

def calculate_for_all_graphs(graph_counted):
    all_graphs = {}
    for counted_graph in graph_counted:
        for key, val in counted_graph.items():
            if key not in all_graphs:
                all_graphs[key] = val
            else:
                all_graphs[key] += val

    normalized_count = all_graphs.copy()

    for key, val in all_graphs.items():
            normalized_count[key] = round(val / all_graphs['number_of_conections'], 4)*100
    number_of_conections = all_graphs['number_of_conections']
    normalized_count.pop('number_of_conections')
    normalized_count = {k: v for k, v in sorted(normalized_count.items(), key=lambda item: item[1], reverse=True)}
    return normalized_count, number_of_conections

def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (np.asarray(values) - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

def plot_graph(normalized_count, name, number_of_conections):
    with sns.axes_style("whitegrid"):
        sns.set(rc={'figure.figsize':(12,6)})
        fig, ax = plt.subplots()
        vcenter = 0
        vmin, vmax = np.asarray(list(normalized_count.values())).min(), np.asarray(list(normalized_count.values())).max()
        vcenter = vmin + ((vmax - vmin)/2)
        normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
        colormap = cm.viridis

        ax1 = sns.barplot(x=list(normalized_count.keys()),
                    y=list(normalized_count.values()),
                    palette=colors_from_values(list(normalized_count.values()), "viridis"),
                    ax=ax)

        ax1.bar_label(ax.containers[0])
        ax1.set_title("Udział całościowy cech w {} krawędziach zbioru danych".format(number_of_conections), fontsize = 16)
        ax1.set_xlabel("Nazwa atrybutu", fontsize = 14)
        ax1.set_ylabel("Procent wystąpienia w danych", fontsize = 14)

        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        scalarmappaple.set_array(list(normalized_count.values()))
        fig.colorbar(scalarmappaple)
        fig.savefig("./visu/" + name)
        plt.show()

        
if __name__ == "__main__":
    args = build_args()
    key_for_graph_count = count_key_attributes(args.directory)
    normalized_count, number_of_conections = calculate_for_all_graphs(key_for_graph_count)
    plot_graph(normalized_count, args.name, number_of_conections)
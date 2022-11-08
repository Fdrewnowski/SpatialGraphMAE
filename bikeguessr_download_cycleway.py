import argparse
import logging

import networkx as nx
import osmnx as ox
from tqdm import tqdm


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='bikeguessr_download_cycleway')
    data_to_download = parser.add_mutually_exclusive_group(required=True)
    data_to_download.add_argument('-a', '--all', action='store_true')
    data_to_download.add_argument('-w', '--wroclaw', action='store_true')
    data_to_download.add_argument('-g', '--gdansk', action='store_true')
    data_to_download.add_argument('-n', '--nysa', action='store_true')
    return parser.parse_args()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    

def download_graph(place: str):
    place_parts = place.split(',')
    assert len(place_parts) > 1
    output = place_parts[0] + "_" + place_parts[-1]+"_recent"
    output = output.replace(' ', "")
    gdf = ox.geocoder.geocode_to_gdf(place)
    polygon = gdf['geometry'][0]
    filters = ['["highway"~"cycleway"]', '["bicycle"~"designated"]', '["bicycle"~"permissive"]', '["bicycle"~"yes"]','["cycleway"~"lane"]']

    #print("Downloading graphs")
    graphs_with_cycle = [ox.graph.graph_from_polygon(
        polygon, network_type='bike', custom_filter=cf, retain_all=True) for cf in filters]
    graph_without_cycle = ox.graph.graph_from_polygon(
        polygon, network_type='drive', retain_all=True)

    # print("Merging")
    previous = graph_without_cycle
    nx.set_edge_attributes(previous, 0, "label")
    for bike_graph in graphs_with_cycle:
        nx.set_edge_attributes(bike_graph, 1, "label")
        merged_graph = nx.compose(previous, bike_graph)
        previous = merged_graph

    merged_graph_copy = merged_graph.copy()

    edge_id = 0
    for edge in merged_graph.edges():
        for connection in merged_graph[edge[0]][edge[1]].keys():
            for key, val in merged_graph[edge[0]][edge[1]][connection].items():
                graph_edge = merged_graph_copy[edge[0]][edge[1]][connection]
                graph_edge['idx'] = edge_id
        edge_id += 1

    # print("Saving")
    merged_graph = ox.utils_graph.remove_isolated_nodes(merged_graph_copy)
    merged_graph.name = output
    ox.save_graphml(merged_graph, filepath="./data_raw/{}.xml".format(output))


if __name__ == "__main__":
    args = build_args()
    places_to_download = []
    if args.all:
        places_to_download = ["Antwerpia, Flanders, Belgia",
            "Ghent, Gent, Flandria Wschodnia, Flanders, Belgia",
            "Charleroi, Hainaut, Walonia, Belgia",
            "Liège, Walonia, 4000, Belgia",
            "Bruksela, Brussels, Brussels-Capital, Belgia",
            "Schaerbeek - Schaarbeek, Brussels-Capital, Belgia",
            "Anderlecht, Brussels-Capital, 1070, Belgia",
            "Brugia, Flandria Zachodnia, Flanders, Belgia",
            "Namur, Walonia, Belgia"
            ]
            
    if args.wroclaw:
        places_to_download = ["Wrocław, województwo dolnośląskie, Polska"]
    if args.gdansk:
        places_to_download = ["Gdańsk, województwo pomorskie, Polska"]
    if args.nysa:
        places_to_download = ["Wałbrzych, województwo dolnośląskie, Polska"]
    place_iter = tqdm(places_to_download, total=len(places_to_download))
    for place in place_iter:
        place_iter.set_description(
            f"# {place}")
        try:
            download_graph(place)
        except:
            logging.warning(f'{place} was corrupted. Skipping...')

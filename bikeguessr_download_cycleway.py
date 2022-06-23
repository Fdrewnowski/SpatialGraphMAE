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
    filters = ['["highway"~"cycleway"]', '["bicycle"~"designated"]', '["cycleway"]']

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
        places_to_download = ["Wrocław, województwo dolnośląskie, Polska",
            "Gdańsk, województwo pomorskie, Polska",
            "Poznań, województwo wielkopolskie, Polska",
            "Warszawa, województwo mazowieckie, Polska",
            "Kraków, województwo małopolskie, Polska",
            "Berlin, Niemcy",
            "Mediolan, Lombardia, Włochy",
            "Amsterdam, Holandia Północna, Niderlandy, Holandia",
            #"Londyn, Greater London, Anglia, Wielka Brytania", # too big
            "Budapeszt, Środkowe Węgry, Węgry",
            "Sztokholm, Solna kommun, Stockholm County, Szwecja",
            "Oslo, Norwegia",
            "Wilno, Samorząd miasta Wilna, Okręg wileński, Litwa",
            "Bruksela, Brussels-Capital, Belgia",
            "Rzym, Roma Capitale, Lacjum, Włochy",
            "Florencja, Metropolitan City of Florence, Toskania, Włochy",
            "Bolonia, Emilia-Romania, Włochy",
            "Lizbona, Lisbon, Portugalia",
            "Madryt, Área metropolitana de Madrid y Corredor del Henares, Wspólnota Madrytu, Hiszpania",
            "Sewilla, Sevilla, Andaluzja, Hiszpania",
            "Walencja, Comarca de València, Walencja, Wspólnota Walencka, Hiszpania",
            "Barcelona, Barcelonès, Barcelona, Katalonia, 08001, Hiszpania",
            "Bilbao, Biscay, Kraj Basków, Hiszpania",
            "Saragossa, Zaragoza, Saragossa, Aragonia, Hiszpania",
            "Marsylia, Marseille, Bouches-du-Rhône, Prowansja-Alpy-Lazurowe Wybrzeże, Francja metropolitalna, 13000, Francja",
            "Lyon, Métropole de Lyon, Departemental constituency of Rhône, Owernia-Rodan-Alpy, Francja metropolitalna, Francja",
            "Bordeaux, Żyronda, Nowa Akwitania, Francja metropolitalna, Francja",
            "Paryż, Ile-de-France, Francja metropolitalna, Francja",
            "Rennes, Ille-et-Vilaine, Brittany, Francja metropolitalna, Francja",
            "Lille, Nord, Hauts-de-France, Francja metropolitalna, Francja ",
            "Amiens, Somme, Hauts-de-France, Francja metropolitalna, Francja",
            "Dublin, Dublin 1, Leinster, Irlandia",
            "Rotterdam, Holandia Południowa, Niderlandy, Holandia",
            "Haga, Holandia Południowa, Niderlandy, Holandia",
            "Dordrecht, Holandia Południowa, Niderlandy, Holandia",
            "Antwerpia, Flanders, Belgia",
            "Essen, Nadrenia Północna-Westfalia, Niemcy",
            "Hanower, Region Hannover, Dolna Saksonia, Niemcy",
            "Monachium, Bawaria, Niemcy",
            "Berno, Bern-Mittelland administrative district, Bernese Mittelland administrative region, Berno, Szwajcaria",
            "Zurych, District Zurich, Zurych, Szwajcaria",
            "Bazylea, Basel-City, Szwajcaria",
            "Salzburg, 5020, Austria",
            "Wiedeń, Austria",
            "Praga, Czechy",
            "Malmo, Malmö kommun, Skåne County, Szwecja",
            "Central Region, Malta",
            "Ljubljana, Upravna Enota Ljubljana, Słowenia",
            "Zagrzeb, City of Zagreb, Chorwacja",
            "Budapeszt, Środkowe Węgry, Węgry",
            "Bukareszt, Rumunia",
            "Helsinki, Helsinki sub-region, Uusimaa, Southern Finland, Mainland Finland, Finlandia",
            "Wenecja, Venezia, Wenecja Euganejska, Włochy",
            "Arnhem, Geldria, Niderlandy, Holandia",
            "Bratysława, Kraj bratysławski, Słowacja",
            "Tallinn, Prowincja Harju, Estonia",
            "Ryga, Liwonia, Łotwa",
            "Neapol, Napoli, Kampania, Włochy",
            "Bari, Apulia, Włochy",
            "Cardiff, Walia, CF, Wielka Brytania",
            "Birmingham, Attwood Green, West Midlands Combined Authority, Anglia, Wielka Brytania",
            "Lwów, Lviv Urban Hromada, Rejon lwowski, Obwód lwowski, Ukraina"]
            
    if args.wroclaw:
        places_to_download = ["Wrocław, województwo dolnośląskie, Polska"]
    if args.gdansk:
        places_to_download = ["Gdańsk, województwo pomorskie, Polska"]
    place_iter = tqdm(places_to_download, total=len(places_to_download))
    for place in place_iter:
        place_iter.set_description(
            f"# {place}")
        try:
            download_graph(place)
        except:
            logging.warn(f'{place} was corrupted. Skipping...')

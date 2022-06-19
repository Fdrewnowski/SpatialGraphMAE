import osmnx as ox
import networkx as nx
from tqdm import tqdm
import copy

def download_graph(place: str):
    place_parts = place.split(',')
    assert len(place_parts) > 1
    output =  place_parts[0] + "_" + place_parts[-1]+"_recent"
    output = output.replace(' ', "")
    gdf = ox.geocoder.geocode_to_gdf(place)
    polygon = gdf['geometry'][0]
    filters = ['["highway"~"cycleway"]', '["bicycle"]', '["cycleway"]']
    
    #print("Downloading graphs")
    graphs_with_cycle = [ox.graph.graph_from_polygon(polygon, network_type='bike', custom_filter=cf, retain_all=True) for cf in filters]
    graph_without_cycle = ox.graph.graph_from_polygon(polygon, network_type='bike', retain_all=True)
   
    #print("Merging")
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

    #print("Saving") 
    merged_graph = ox.utils_graph.remove_isolated_nodes(merged_graph_copy)
    merged_graph.name = output
    ox.save_graphml(merged_graph, filepath="./data_raw/{}.xml".format(output))




if __name__ == "__main__":
    places_to_download = ["Wrocław, województwo dolnośląskie, Polska"]
                        #  "Gdańsk, województwo pomorskie, Polska",
                        #  "Berlin, Niemcy",
                        #  "Mediolan, Lombardia, Włochy"] #,
                        #   "Amsterdam, Holandia Północna, Niderlandy, Holandia",
                        #   "Poznań, województwo wielkopolskie, Polska",
                        #   "Warszawa, województwo mazowieckie, Polska",
                        #   "Kraków, województwo małopolskie, Polska",
                        #   "Londyn, Greater London, Anglia, Wielka Brytania",
                        #   "Budapeszt, Środkowe Węgry, Węgry",
                        #   "Sztokholm, Solna kommun, Stockholm County, Szwecja",
                        #   "Oslo, Norwegia",
                        #   "Wilno, Samorząd miasta Wilna, Okręg wileński, Litwa",
                        #   "Bruksela, Brussels-Capital, Belgia"'
                        #   "Paryż, Ile-de-France, Francja metropolitalna, Francja",
                        #   "Rzym, Roma Capitale, Lacjum, Włochy",
                        #   "Florencja, Metropolitan City of Florence, Toskania, Włochy",
                        #   "Bolonia, Emilia-Romania, Włochy"
                        #   ]
    for place in tqdm(places_to_download, total = len(places_to_download)):
        download_graph(place)
import osmnx as ox
import networkx as nx
import folium
from tqdm import tqdm

from networkx.classes.multidigraph import MultiDiGraph
from dgl.heterograph import DGLHeteroGraph
from torch import Tensor
from folium.folium import Map
from dgl.data.utils import load_graphs
import random
import copy


def _visualise_masked_roads(grapf_networkx: MultiDiGraph, mask: Tensor, name: str):
    assert grapf_networkx.number_of_edges() == mask.shape[0]
    
    mask_ids = ((mask == True).nonzero(as_tuple=True)[0]).tolist()
    keys_to_remove = ['oneway', 'lanes', 'ref', 'name', 'highway', 'maxspeed', 'length', 'label']

    year = "2022"

    dif_masked_cycle = nx.create_empty_copy(grapf_networkx)
    dif_masked_road = dif_masked_cycle.copy()
    diff_unmasked = dif_masked_cycle.copy()
    
    
    for idx, x in enumerate(tqdm(set(grapf_networkx.edges()), total = len(set(grapf_networkx.edges())))):
        edge = grapf_networkx[x[0]][x[1]][0]
        if edge['idx'] in mask_ids:
            dif_attributes = edge.copy()

            if dif_attributes['label'] == 1: #if cycle
                vis_data = dict(
                href=f"https://www.openstreetmap.org/way/{edge['osmid']}", 
                years=['cycle', 'masked'], 
                data=dict()
                )
                vis_data['data'] = {year:[dif_attributes['label'],True]}
                for key in keys_to_remove:
                    if key in dif_attributes.keys():
                        dif_attributes.pop(key)
                dif_attributes['vis_data'] = vis_data
                dif_masked_cycle.add_edges_from([(x[0], x[1], dif_attributes)])
            else:
                vis_data = dict(
                href=f"https://www.openstreetmap.org/way/{edge['osmid']}", 
                years=['cycle', 'masked'], 
                data=dict()
                )
                vis_data['data'] = {year:[dif_attributes['label'],True]}
                for key in keys_to_remove:
                    if key in dif_attributes.keys():
                        dif_attributes.pop(key)
                dif_attributes['vis_data'] = vis_data
                dif_masked_road.add_edges_from([(x[0], x[1], dif_attributes)])
        else:
            vis_data = dict(
            href=f"https://www.openstreetmap.org/way/{edge['osmid']}", 
            years=['cycle', 'masked'], 
            data=dict()
            )
            dif_attributes = edge.copy()
            vis_data['data'] = {year:[dif_attributes['label'],False]}
            for key in keys_to_remove:
                if key in dif_attributes.keys():
                    dif_attributes.pop(key)
            dif_attributes['vis_data'] = vis_data
            diff_unmasked.add_edges_from([(x[0], x[1], dif_attributes)])

            
    m = ox.plot_graph_folium(diff_unmasked, popup_attribute='vis_data', color="blue", edge_width=1)
    try:
        m = ox.plot_graph_folium(dif_masked_cycle, popup_attribute='vis_data', graph_map=m, color="green", edge_width=2)
    except Exception as e:
        print(str(e))
    try:
        m = ox.plot_graph_folium(dif_masked_road, popup_attribute='vis_data', graph_map=m, color="red", edge_width=2)
    except Exception as e:
        print(str(e))
    m.save(f"./visu/{name}_masks.html")

if __name__ == "__main__":
    #options
    ox_graph_name = "Wrocław_Polska_recent.xml"
    dgl_graph_name = "Wroclaw_Polska_recent_masks.graph"
    mask_to_visualise = 'test_mask' # 3 options train, dev, test

    graph_ox = ox.io.load_graphml("./data_raw/" + ox_graph_name)
    dgl_graph = load_graphs("./data_transformed/" + dgl_graph_name)[0][0]

    _visualise_masked_roads(graph_ox, dgl_graph.ndata[mask_to_visualise] , ox_graph_name.split('.')[0])






















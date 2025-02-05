import osmnx as ox
import networkx as nx
import folium
from tqdm import tqdm
import torch
from networkx.classes.multidigraph import MultiDiGraph
from dgl.heterograph import DGLHeteroGraph
from torch import Tensor
from folium.folium import Map
from dgl.data.utils import load_graphs
import pickle


def _show_preds(grapf_networkx: MultiDiGraph, mask: Tensor, preds: Tensor, name: str, popup: bool):
    assert grapf_networkx.number_of_edges() == mask.shape[0]
    
    mask_ids = ((mask == True).nonzero(as_tuple=True)[0]).tolist()
    pred_ids = ((preds == True).nonzero(as_tuple=True)[0]).tolist()

    year = str(2022)
    dif_cycle = nx.create_empty_copy(grapf_networkx)
    dif_cycle_new = dif_cycle.copy()
    
    
    for x in tqdm(set(grapf_networkx.edges()), total = len(set(grapf_networkx.edges()))):
        edge = grapf_networkx[x[0]][x[1]][0]
        #print(edge)
        dif_attributes = edge.copy()
        if int(dif_attributes['label']) == 1: #if cycle
            vis_data = dict(
            href=f"https://www.openstreetmap.org/way/{edge['osmid']}", 
            years=['cycle', 'masked'], 
            data=dict()
            )
            vis_data['data'] = {year:[dif_attributes['label'],True]}
            dif_attributes['vis_data'] = vis_data
            if not popup:
                dif_attributes = None
            dif_cycle.add_edges_from([(x[0], x[1], dif_attributes)])
        elif int(dif_attributes['label']) == 0 and (int(edge['idx']) not in pred_ids or int(edge['idx']) not in mask_ids):
            vis_data = dict(
            href=f"https://www.openstreetmap.org/way/{edge['osmid']}", 
            years=['cycle', 'masked'], 
            data=dict()
            )
            vis_data['data'] = {year:[dif_attributes['label'],True]}
            dif_attributes['vis_data'] = vis_data
            if not popup:
                dif_attributes = None
            dif_cycle_new.add_edges_from([(x[0], x[1], dif_attributes)])

    if not popup:
        m = ox.plot_graph_folium(dif_cycle, color="blue", edge_width=1)
    else:
        m = ox.plot_graph_folium(dif_cycle, popup_attribute='vis_data', color="blue")
    try:
        if not popup:
            m = ox.plot_graph_folium(dif_cycle_new, graph_map=m, color="red", edge_width=1)
        else:
            m = ox.plot_graph_folium(dif_cycle_new, popup_attribute='vis_data', graph_map=m, color="red")
    except Exception as e:
        print("Error in pred as true cycle" + str(e))

    m.save(f"./visu/{name}_{str(popup)}_not_cycle.html")

    #m1 = ox.plot_graph_folium(dif_cycle, popup_attribute='vis_data', color="blue")
    #m1.save(f"./visu/{name}_{str(popup)}_now.html")



if __name__ == "__main__":
    #options
    ox_graph_name = "Wrocław_Polska_recent.xml"
    dgl_graph_name = "wro_xd.bin"
    prediction_file = "Wrocław_Polska_recent.pkl"
    mask_to_visualise = 'test_mask' # 3 options train, dev, test

    graph_ox = ox.io.load_graphml("./data_raw/" + ox_graph_name)
    #dgl_graph = load_graphs("./final_results/" + dgl_graph_name)[0][0]

    with open("./preds/" + prediction_file, 'rb') as handle:
        predictions = pickle.load(handle).cpu()
        predictions_bycycle = predictions.max(1)[1]#.type_as(dgl_graph.ndata[mask_to_visualise])

    top_preds = predictions[predictions_bycycle].reshape(2, -1).topk(1000)[1][1].tolist()
    print(predictions_bycycle.unique(return_counts =True))
    all_elem = torch.ones(predictions_bycycle.shape[0])

    _show_preds(graph_ox,  all_elem, predictions_bycycle , prediction_file.split('.')[0], True)
import argparse
import pickle
from pathlib import Path
from typing import List

import torch
from dgl import DGLHeteroGraph
from dgl.data.utils import load_graphs

from bikeguessr_transform import load_transform_single_bikeguessr
from graphmae.models.edcoder import PreModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO:
#   Load ready model (argparse)
#   Load linegraph for prediction
#       OR load and transform graphml file
#   Predict the labels
#   Save predictions


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='SpatialGraphMAE')
    parser.add_argument('-m', '--model', type=str, default='sgmae.pt')
    parser.add_argument('-t', '--transform_data', action='store_true')
    parser.add_argument('-d', '--data', type=str,
                        default='data_transformed/bikeguessr.bin')
    parser.add_argument('-s', '--save', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = build_args()
    sgmae: PreModel = torch.load(args.model)

    graph = None
    if args.transform_data:
        graph = load_transform_single_bikeguessr(args.data, save=False)
    else:
        graphs: List[DGLHeteroGraph] = load_graphs(args.data)[0]
        graph = graphs[0]

    X: torch.Tensor = graph.ndata['feat']

    sgmae = sgmae.to(device)
    graph = graph.to(device)
    X = X.to(device)

    pred = sgmae(graph, X)
    pred_1 = sgmae(None, X)
    if args.save:
        pred_file = Path('preds/' + Path(args.data).stem + '.pkl')
        pred_file.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_file, 'wb+') as handle:
            pickle.dump(pred, handle, protocol=pickle.HIGHEST_PROTOCOL)

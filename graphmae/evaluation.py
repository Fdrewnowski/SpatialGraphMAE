import copy
import logging
import pickle
import warnings
from typing import Dict, Tuple

import torch
import torch.nn as nn
from dgl.heterograph import DGLHeteroGraph
from tqdm import tqdm

from graphmae.models.edcoder import PreModel
from graphmae.utils import accuracy, create_optimizer, f1, recall

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def prepare_eval_dict(f1s, recalls) -> Dict[str, float]:
    out = {}
    out['F1/Test'] = f1s[0]
    out['F1/Val'] = f1s[1]
    out['F1/Train'] = f1s[2]
    out['Recall/Test'] = recalls[0]
    out['Recall/Val'] = recalls[1]
    out['Recall/Train'] = recalls[2]
    return out


def node_classification_evaluation(
        model: PreModel,
        graph: DGLHeteroGraph,
        x: torch.Tensor,
        num_classes: int,
        lr_f: float,
        weight_decay_f: float,
        max_epoch_f: float,
        device: torch.device,
        linear_prob: bool = True,
        mute: bool = False) -> Tuple[torch.nn.Module, Dict[str, float]]:
    model.eval()
    if linear_prob:
        with torch.no_grad():
            x = model.embed(graph.to(device), x.to(device))
            in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes)

    num_finetune_params = [p.numel()
                           for p in encoder.parameters() if p.requires_grad]
    if not mute:
        logging.info(
            f"num parameters for finetuning: {sum(num_finetune_params)}")

    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    clf, clf_scores = linear_probing_for_transductive_node_classiifcation(
        encoder, graph, x, optimizer_f, max_epoch_f, device, mute)
    return clf, clf_scores


def linear_probing_for_transductive_node_classiifcation(
        model: PreModel,
        graph: DGLHeteroGraph,
        feat: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        max_epoch: int,
        device: torch.device,
        mute: bool = False) -> Tuple[torch.nn.Module, Dict[str, float]]:
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = feat.to(device)

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]

    best_val_f1, best_val_recall = 0, 0
    best_val_f_epoch, best_val_r_epoch = 0, 0
    best_model_f1 = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(graph, x)
            # validation metrics
            val_f1 = f1(pred[val_mask], labels[val_mask])
            val_recall = recall(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            # test metrics
            test_f1 = f1(pred[test_mask], labels[test_mask])
            test_recall = recall(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_val_f_epoch = epoch
            best_model_f1 = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_f1:{val_f1}, test_loss:{test_loss.item(): .4f}, test_f1:{test_f1: .4f}")

    best_model_f1.eval()
    with torch.no_grad():
        pred = best_model_f1(graph, x)

        f1_scores = f1(pred[test_mask], labels[test_mask]), \
            f1(pred[val_mask], labels[val_mask]), \
            f1(pred[train_mask], labels[train_mask])

        recall_scores = recall(pred[test_mask], labels[test_mask]), \
            recall(pred[val_mask], labels[val_mask]), \
            recall(pred[train_mask], labels[train_mask])

    if not mute:
        logging.info(
            f"--- TestF1: {test_f1:.4f}, early-stopping-TestF1: {f1_scores[0]:.4f}," +
            f"Best ValF1: {best_val_f1:.4f} in epoch {best_val_f_epoch} --- ")
        logging.info(
            f"--- TestRecall: {test_recall:.4f}, early-stopping-TestRecall: {recall_scores[0]:.4f}," +
            f"Best ValRecall: {best_val_recall:.4f} in epoch {best_val_r_epoch} --- ")

    with open('./preds/wro_xd_v2.pkl', 'wb+') as handle:
        pickle.dump(pred.cpu(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    # (final_acc, es_acc, best_acc)
    clf_eval_dict = prepare_eval_dict(f1_scores, recall_scores)

    return best_model_f1, clf_eval_dict


def linear_probing_for_inductive_node_classiifcation(model, x, labels, mask, optimizer, max_epoch, device, mute=False):
    if len(labels.shape) > 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    train_mask, val_mask, test_mask = mask

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

        best_val_acc = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(None, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(None, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(None, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if not mute:
        logging.info(
            f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}")

    return test_acc, estp_test_acc


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)
        self.linear2 = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        #out = self.linear2(logits)
        return logits

import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from dgl.data.utils import load_graphs
from dgl.heterograph import DGLHeteroGraph
from tqdm import tqdm

from graphmae.evaluation import node_classification_evaluation
from graphmae.models import build_model
from graphmae.models.edcoder import PreModel
from graphmae.utils import (TBLogger, build_args, create_optimizer,
                            get_current_lr, load_best_configs, set_random_seed)
from main_transform_raw_bikeguessr import DATA_OUTPUT, _sizeof_fmt

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def load_bikeguessr_dataset(filename: str, directory: str = None) -> Tuple[List[DGLHeteroGraph], Tuple[int, int]]:
    logging.info('load bikeguessr dataset')
    if directory is None:
        directory = os.path.join(os.getcwd(), DATA_OUTPUT)
    file = Path(directory, filename)

    logging.info('processing: ' + str(file.stem) +
                 ' size: ' + _sizeof_fmt(os.path.getsize(file)))
    graphs, _ = load_graphs(str(file))
    num_features, num_classes = [], []
    for i in range(len(graphs)):
        graphs[i] = graphs[i].remove_self_loop()
        graphs[i] = graphs[i].add_self_loop()
    num_features = graphs[i].ndata["feat"].shape[1]
    num_classes = 2

    return graphs, (num_features, num_classes)


def train_transductive(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    logging.info("using device: {}".format(device))
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    graphs, (num_features, num_classes) = load_bikeguessr_dataset(
        r'C:\Users\jbelter\VisualCodeProjects\SpatialGraphMAE\data_transformed\14af828816524a98b4ec85f4cae30389_masks.graph')
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(
                name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")

            def scheduler(epoch): return (
                1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        X = [g.ndata['feat'] for g in graphs]
        if not load_model:
            model = pretrain(model, graphs, X, optimizer, max_epoch, device, scheduler,
                             num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        for g, x in zip(graphs, X):
            final_acc, estp_acc, _ = node_classification_evaluation(
                model, g, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
            acc_list.append(final_acc)
            estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    logging.info(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    logging.info(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")


def pretrain(model: PreModel,
             graphs: List[DGLHeteroGraph],
             feats: List[torch.Tensor],
             optimizer: torch.optim.Optimizer,
             max_epoch: int,
             device: torch.device,
             scheduler: torch.optim.lr_scheduler.LambdaLR,
             num_classes: int,
             lr_f: float,
             weight_decay_f: float,
             max_epoch_f: int,
             linear_prob: bool,
             logger: TBLogger = None):
    logging.info("start training..")
    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        epoch_f1_scores_train, epoch_f1_scores_val, epoch_f1_scores_test, epoch_loss = [], [], [], []
        for graph, feat in zip(graphs, feats):
            g = graph.to(device)
            x = feat.to(device)
            model.train()

            loss, loss_dict = model(g, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_iter.set_description(
                f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
            epoch_loss.append(loss.cpu().detach().numpy())

            if (epoch + 1) % 4 == 0:
                _, _, f1_scores = node_classification_evaluation(
                    model, g, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, epoch, linear_prob, mute=True)
                epoch_f1_scores_train.append(f1_scores[0])
                epoch_f1_scores_val.append(f1_scores[1])
                epoch_f1_scores_test.append(f1_scores[2])
        if logger is not None:
            logging_dict = {}
            logging_dict['Loss/train'] = np.mean(epoch_loss)
            if (epoch + 1) % 4 == 0:
                logging_dict['F1/train'] = np.mean(epoch_f1_scores_train)
                logging_dict['F1/test'] = np.mean(epoch_f1_scores_test)
                logging_dict['F1/val'] = np.mean(epoch_f1_scores_val)
            logger.note(logging_dict, step=epoch)

    # return best_model
    return model


if __name__ == '__main__':
    args = build_args()
    args.dataset = 'bikeguessr'
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    args.save_model = True
    args.load_model = False
    print(args)
    train_transductive(args)
    # TENSORBOARD_WRITER.close()

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from dgl.data.utils import load_graphs
from dgl.heterograph import DGLHeteroGraph
from tqdm import tqdm

from bikeguessr_transform import DATA_OUTPUT, _sizeof_fmt
from graphmae.evaluation import node_classification_evaluation
from graphmae.models import build_model
from graphmae.models.edcoder import PreModel
from graphmae.utils import (TBLogger, build_args, create_optimizer,
                            get_current_lr, load_best_configs, set_random_seed)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def load_bikeguessr_dataset(filepath: str) -> Tuple[List[DGLHeteroGraph], Tuple[int, int]]:
    logging.info('load bikeguessr dataset')
    if filepath is None:
        filepath = str(Path(DATA_OUTPUT, 'bikeguessr.bin'))
    file = Path(filepath)

    logging.info('processing: ' + str(file.absolute()) +
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
    dataset_path = args.path
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f

    optim_type = args.optimizer

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    graphs, (num_features, num_classes) = load_bikeguessr_dataset(dataset_path)
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            # logger = TBLogger(
            #    name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            logger = TBLogger(name=f"{dataset_name}_{current_time}")
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
            model = torch.load("bikeguessr.model")


        model = model.to(device)
        model.eval()

        for g, x in zip(graphs, X):
            best_model, f1_scores = node_classification_evaluation(
                model, g, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)

        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model, "bikeguessr.model")

        logging.info(f'final f1 score on test: {f1_scores[2]}')

        if logger is not None:
            logger.finish()

    return best_model


def _is_same_model(model: PreModel, other_model: PreModel):
    for p1, p2 in zip(model.parameters(), other_model.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


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
             logger: TBLogger = None,
             eval_epoch: int = 10):
    logging.info("start training..")
    epoch_iter = tqdm(range(max_epoch))
    best_test_f1_score = 0.0
    best_model = None
    for epoch in epoch_iter:
        epoch_f1_scores_train, epoch_f1_scores_val, epoch_f1_scores_test, epoch_loss = [], [], [], []
        for graph, feat in zip(graphs, feats):
            g = graph.to(device)
            x = feat.to(device)
            model.train()

            loss, _ = model(g, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_iter.set_description(
                f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
            epoch_loss.append(loss.cpu().detach().numpy())

            if (epoch + 1) % eval_epoch == 0:
                model_with_classifier, f1_scores = node_classification_evaluation(
                    model, g, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, epoch, linear_prob, mute=True)
                epoch_f1_scores_train.append(f1_scores[0])
                epoch_f1_scores_val.append(f1_scores[1])
                epoch_f1_scores_test.append(f1_scores[2])
        if (epoch + 1) % eval_epoch == 0:
            if best_test_f1_score < np.mean(epoch_f1_scores_test):
                best_test_f1_score = np.mean(epoch_f1_scores_test)
                torch.save(model_with_classifier, "best_f1_score_model.pt")
                best_model = torch.load("best_f1_score_model.pt")

        if logger is not None:
            logging_dict = {}
            logging_dict['Loss/train'] = np.mean(epoch_loss)
            if (epoch + 1) % eval_epoch == 0:
                logging_dict['F1/train'] = np.mean(epoch_f1_scores_train)
                logging_dict['F1/test'] = np.mean(epoch_f1_scores_test)
                logging_dict['F1/val'] = np.mean(epoch_f1_scores_val)
            logger.note(logging_dict, step=epoch)

    if _is_same_model(best_model, model_with_classifier):
        logging.warn('Best model and currently trained model are identical')
    # return best_model
    return best_model


if __name__ == '__main__':
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    train_transductive(args)
    # TENSORBOARD_WRITER.close()

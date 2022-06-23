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
    eval_epoch = args.eval_epoch
    eval_repeats = args.eval_repeats

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
        if load_model:
            logging.info("Loading Model ... ")
            model = torch.load("sgmae.model")
        else:
            gae_early_stopping, gae_full, (f1_early_stopping, f1_full) = \
                pretrain(model, graphs, X, optimizer, max_epoch, device, scheduler,
                         num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger,
                         eval_epoch, eval_repeats)
            model = gae_full
            if save_model:
                logging.info("Saving trained Graph Masked Auto Encoders...")
                torch.save(gae_full, "gmae_full.model")
                torch.save(gae_early_stopping, "gmae_early_stopping.model")
            logging.info(
                f'early stopping f1 score on test (pretrain): {f1_early_stopping}')
            logging.info(
                f'end of training f1 score on test (pretrain): {f1_full}')

        model = model.to(device)
        model.eval()

        for g, x in zip(graphs, X):
            clf_eot, scores_eot = node_classification_evaluation(
                model, g, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        # if load_model:
        #     logging.info(f'early stopping f1 score on test: {f1_early_stopping}')
        #     logging.info(f'end of training f1 score on test: {f1_full}')

        # best_model = sgmae_full if f1_score_early_stopping > f1_score_eot[2] else sgmae_full_eot

        # TODO: training to create sgmae
        if save_model:
            logging.info("Saving Model ...")
            torch.save(model, "sgmae.model")

        if logger is not None:
            logger.finish()

    return model


def _is_same_model(model: PreModel, other_model: PreModel):
    try:
        for p1, p2 in zip(model.parameters(), other_model.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                return False
        return True
    except:
        return False


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
             eval_epoch: int = 10,
             eval_repeats: int = 5) -> Tuple[PreModel, PreModel, Tuple[float]]:
    logging.info("start training..")
    epoch_iter = tqdm(range(max_epoch))
    best_val_f1_score, best_test_f1_score = 0.0, 0.0
    best_gae = None
    for epoch in epoch_iter:
        e_f1_test, e_f1_val, e_f1_train = [], [], []
        e_recall_test, e_recall_val, e_recall_train = [], [], []
        epoch_loss = []
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
                g_f1_test, g_f1_val, g_f1_train = [], [], []
                g_recall_test, g_recall_val, g_recall_train = [], [], []
                logging.info(f'start evaluating gmae at epoch {epoch}')
                for _ in tqdm(range(eval_repeats)):
                    _, scores = node_classification_evaluation(
                        model, g, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)
                    g_f1_test.append(scores['F1/Test'])
                    g_f1_val.append(scores['F1/Val'])
                    g_f1_train.append(scores['F1/Train'])
                    g_recall_test.append(scores['Recall/Test'])
                    g_recall_val.append(scores['Recall/Val'])
                    g_recall_train.append(scores['Recall/Train'])
                e_f1_test.append(np.mean(g_f1_test))
                e_f1_val.append(np.mean(g_f1_val))
                e_f1_train.append(np.mean(g_f1_train))
                e_recall_test.append(np.mean(g_recall_test))
                e_recall_val.append(np.mean(g_recall_val))
                e_recall_train.append(np.mean(g_recall_train))
        if (epoch + 1) % eval_epoch == 0:
            if best_val_f1_score < np.mean(e_f1_val):
                best_val_f1_score = np.mean(e_f1_val)
                best_test_f1_score = np.mean(e_f1_test)
                torch.save(model, "sgmae_early_stopping.model")
                best_gae = torch.load("sgmae_early_stopping.model")

        if logger is not None:
            logging_dict = {}
            logging_dict['Loss/train'] = np.mean(epoch_loss)
            if (epoch + 1) % eval_epoch == 0:
                logging_dict['F1/test'] = np.mean(e_f1_test)
                logging_dict['F1/val'] = np.mean(e_f1_val)
                logging_dict['F1/train'] = np.mean(e_f1_train)
                logging_dict['Recall/test'] = np.mean(e_recall_test)
                logging_dict['Recall/val'] = np.mean(e_recall_val)
                logging_dict['Recall/train'] = np.mean(e_recall_train)
            logger.note(logging_dict, step=epoch)

    early_stopping_f1_score = best_test_f1_score
    full_f1_score = np.mean(e_f1_test)

    if _is_same_model(model, best_gae):
        logging.warn('Best model and currently trained model are identical')

    return model, best_gae, (early_stopping_f1_score, full_f1_score)


if __name__ == '__main__':
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    train_transductive(args)
    # TENSORBOARD_WRITER.close()

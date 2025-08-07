import gzip
import pickle
import argparse
import json
import torch
import numpy as np
import random
import networkx as nx
from haversine import haversine
from rich.console import Console
from tqdm import tqdm

console = Console(record=True)


def dump_obj(obj, filename, protocol=-1, serializer=pickle):
    with gzip.open(filename, 'wb') as fout:
        serializer.dump(obj, fout, protocol)


def load_obj(filename, serializer=pickle):
    with gzip.open(filename, 'rb') as fin:
        obj = serializer.load(fin)
    return obj


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_root", type=str, default="./data_dump/", help="dump data")
    parser.add_argument("--dataset", type=str, default="cmu", help="dataset name")
    parser.add_argument("--model", type=str, default="gauann_2", help="model type")
    parser.add_argument("--model_dict", type=str, default="./model_state_dict/", help="model parameters")
    parser.add_argument("--epochs", type=int, default=50, help="how many epochs of training")
    parser.add_argument('--save', action='store_true', help='if exists save the model after training')
    parser.add_argument('--load', action='store_true', help='if exists load pretrained model from file')
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.2, help="the weight of location regularization loss")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--outsize', type=int, default=256, help="output size")
    parser.add_argument('--n_layers', type=int, default=2, help="the number of GNN layers to sample")
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader's num_workers")

    args = parser.parse_args()
    return args


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def breakpoint():
    import pdb
    pdb.set_trace()


def second_order_neighbor(g_raw, nodes):
    # g_raw: the whole graph including known nodes and mentioned nodes --large graph
    # nodes: the node_id of known nodes --small graph node
    nodes = set(nodes)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    all_nodes = set(g_raw.nodes())
    for m in all_nodes:
        neighbors = g_raw[m]
        neighbors_inlist = [t for t in neighbors if t in nodes]
        # add edge between known m and known n
        if m in nodes:
            for n in neighbors_inlist:
                if m < n:
                    if not g.has_edge(m, n):
                        g.add_edge(m, n)
        # add edge between known n1 and known n2
        # because n1 and n2 have relation to m, whether m is in the list or not
        for n1 in neighbors_inlist:
            for n2 in neighbors_inlist:
                if n1 < n2:
                    if not g.has_edge(n1, n2):
                        g.add_edge(n1, n2)
    return g




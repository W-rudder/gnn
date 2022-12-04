"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse

import torch
import pandas as pd
import numpy as np
#import numba

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder, get_index
from utils import EarlyStopMonitor, RandEdgeSampler

DATA = 'video'

### Load data and train val test split
g_df = pd.read_csv('../data/ml_{}.csv'.format(DATA))
e_feat = np.load('../data/ml_{}.npy'.format(DATA))
n_feat = np.load('../data/ml_{}_node.npy'.format(DATA))

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list)

# print(len(full_adj_list))
test_id_l = src_l[-5:]
test_t_l = ts_l[-5:]
print(test_id_l, test_t_l)
src_list, src_t_list, ngh_list, ngh_t_list, ngh_e_list = full_ngh_finder.subgragh_to_adj(2, test_id_l, test_t_l, 20)
print(src_list.shape, src_t_list.shape)
print(ngh_list.shape, ngh_t_list.shape, ngh_e_list.shape)
#  {(1854, 4397, 402278400.0, 1), (2497, 12705, 448761600.0, 2), (2559, 12715, 448761600.0, 1), (2497, 12705, 448761600.0, 1), (2515, 6124, 413683200.0, 2), (2556, 5911, 413078400.0, 1), (2515, 6124, 413683200.0, 1)}
for i in range(5):
    print(np.count_nonzero(src_list[i]))
    print(np.count_nonzero(ngh_list[i]))
# print(src_list.device)
# print(n_feat.shape, e_feat.shape)
# AGG_METHOD = 'attn'
# ATTN_MODE = 'prod'
# SEQ_LEN = 20
# tgan = TGAN(full_ngh_finder, n_feat, e_feat,
#             num_layers=2, use_time='time', agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
#             seq_len=SEQ_LEN, n_head=2, drop_out=0.1, node_dim=2, time_dim=100)
# res = tgan.sub_tem_conv(src_list, src_t_list, ngh_list, ngh_t_list, ngh_e_list, num_neighbors=20)
# print(res[0])

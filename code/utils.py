from __future__ import annotations
import argparse
import torch
import dgl
import csv
import time
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pickle
# from torch_sparse import spspmm
import os
import re
import copy
import networkx as nx
import numpy as np
# import scipy.sparse as sp
import torch as th
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import random
import dgl
import time
import pandas as pd


def feature_normalize(feature): 
    '''sum_norm'''
    feature = np.array(feature)
    rowsum = feature.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return feature / rowsum

def train_val_test_split(labels,train_ratio=0.5,val_ratio=0.25,seed=20,label_number=1000):
    import random
    random.seed(seed)
    label_idx_0 = np.where(labels==0)[0]  # 只要label为0和1的
    label_idx_1 = np.where(labels==1)[0]  # 
    random.shuffle(label_idx_0) 
    random.shuffle(label_idx_1)
    position1 = train_ratio
    position2 = train_ratio + val_ratio
    idx_train = np.append(label_idx_0[:min(int(position1 * len(label_idx_0)), label_number//2)], 
                          label_idx_1[:min(int(position1 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(position1 * len(label_idx_0)):int(position2 * len(label_idx_0))], 
                        label_idx_1[int(position1 * len(label_idx_1)):int(position2 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(position2 * len(label_idx_0)):],
                         label_idx_1[int(position2 * len(label_idx_1)):])
    print('train,val,test:',len(idx_train),len(idx_val),len(idx_test))
    return idx_train, idx_val, idx_test


def load_data(args: argparse.Namespace) -> tuple[dgl.DGLGraph, dict[str, torch.Tensor]]:
    def load_dataset(args):
        datapath = "../data/"
        dataname = args.dataset +'/'
        if args.dataset=='nba':
            # edge_df = pd.read_csv('../data/nba/' + 'nba_relationship.txt', sep='\t')
            edges_unordered = np.genfromtxt(datapath + dataname + 'nba_relationship.txt').astype('int')
            # node_df = pd.read_csv(os.path.join('../dataset/nba/', 'nba.csv'))
            idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'nba.csv'))
            print('load edge data')
            predict_attr = 'SALARY'
            labels = idx_features_labels[predict_attr].values
            header = list(idx_features_labels.columns)
            header.remove(predict_attr)
            sens_attr = "country"
            # labels = y
            adj_start = time.time()
            # feature = node_df[node_df.columns[2:]]
            feature = idx_features_labels[header]
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = ["country"])

            # idx = node_df['user_id'].values # for relations
            idx = np.array(idx_features_labels["user_id"], dtype=int)
            idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
            adj_end = time.time()
            print('create adj time is {:.3f}'.format((adj_end-adj_start)))
            # print('adj created!')
            sens = idx_features_labels[sens_attr].values.astype(int) 
            sens = torch.FloatTensor(sens) 
            feature = np.array(feature)
            feature = feature_normalize(feature)
            feature = torch.FloatTensor(feature)
            labels = torch.LongTensor(labels) 
            labels[labels >1] =1
            
            # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
            idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
            
            print('dataset:',args.dataset)
            print('sens:',sens_attr)
            print('feature:',feature.shape)
            print('labels:',torch.unique(labels))
            return adj, feature, labels, sens, idx_train, idx_val, idx_test # 不包含label [0,1(大于1的转成1)]以外的值的id

        elif args.dataset=='pokec_z':
            edges_unordered = np.genfromtxt(datapath + dataname + 'region_job_relationship.txt').astype('int')
            predict_attr = 'I_am_working_in_field'
            sens_attr = 'region'
            print('Loading {} dataset'.format(args.dataset))
            idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'region_job.csv'))
            header = list(idx_features_labels.columns)
            header.remove(predict_attr)

            # header.remove(sens_attr)
            # header.remove(predict_attr)
            feature = idx_features_labels[header]
            # feature=feature_normalize(idx_features_labels[header])
            labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
            #-----
            adj_start = time.time()
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = ["region"])

            # idx = node_df['user_id'].values # for relations
            idx = np.array(idx_features_labels["user_id"], dtype=int)
            idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
            adj_end = time.time()
            sens = idx_features_labels[sens_attr].values.astype(int) 
            sens = torch.FloatTensor(sens) 
            feature = np.array(feature)
            feature = feature_normalize(feature)
            feature = torch.FloatTensor(feature)
            # return feature
            labels = torch.LongTensor(labels) 
            labels[labels >1] =1
            # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
            idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
        
            print('dataset:',args.dataset)
            print('sens:',sens_attr)
            print('feature:',feature.shape)
            print('labels:',torch.unique(labels))
            return adj, feature, labels, sens, idx_train, idx_val, idx_test
        elif args.dataset=='pokec_n':
            edges_unordered = np.genfromtxt(datapath + dataname + 'region_job_2_relationship.txt').astype('int')
            predict_attr = 'I_am_working_in_field'
            sens_attr = 'region'
            print('Loading {} dataset'.format(args.dataset))
            idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'region_job_2.csv'))
            header = list(idx_features_labels.columns)
            header.remove(predict_attr)

            # header.remove(sens_attr)
            # header.remove(predict_attr)
            feature = idx_features_labels[header]
            # feature=feature_normalize(idx_features_labels[header])
            labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
            #-----
            adj_start = time.time()
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = ["region"])

            # idx = node_df['user_id'].values # for relations
            idx = np.array(idx_features_labels["user_id"], dtype=int)
            idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
            adj_end = time.time()
            sens = idx_features_labels[sens_attr].values.astype(int) 
            sens = torch.FloatTensor(sens) 
            feature = np.array(feature)
            feature = feature_normalize(feature)
            feature = torch.FloatTensor(feature)
            # return feature
            labels = torch.LongTensor(labels) 
            labels[labels >1] =1
            # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
            idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
        
            print('dataset:',args.dataset)
            print('sens:',sens_attr)
            print('feature:',feature.shape)
            print('labels:',torch.unique(labels))
            return adj, feature, labels, sens, idx_train, idx_val, idx_test
        elif args.dataset=='credit':
            idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'credit.csv'))
            edges_unordered = np.genfromtxt(datapath + dataname + 'credit_edges.txt').astype('int')
            sens_attr="Age"
            predict_attr="NoDefaultNextMonth"
            print('Loading {} dataset'.format(args.dataset))
            # header = list(idx_features_labels.columns)
            header = list(idx_features_labels.columns)
            header.remove('Single')
            header.remove(predict_attr)
            
            feature = idx_features_labels[header]
            labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = ["Age"])
            adj_start = time.time()
            idx = np.arange(feature.shape[0]) 
            idx_map = {j: i for i, j in enumerate(idx)}
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
            adj_end = time.time()
            
            sens = idx_features_labels[sens_attr].values.astype(int) 
            sens = torch.FloatTensor(sens) 
            feature = np.array(feature)
            feature = feature_normalize(feature)
            feature = torch.FloatTensor(feature)
            labels = torch.LongTensor(labels) 
            labels[labels >1] =1
            idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
        
            print('dataset:',args.dataset)
            print('sens:',sens_attr)
            print('feature:',feature.shape)
            print('labels:',torch.unique(labels))
            idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
            
            return adj, feature, labels, sens, idx_train, idx_val, idx_test
        
        elif args.dataset=='income':
            idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'income.csv'))
            edges_unordered = np.genfromtxt(datapath + dataname + 'income_edges.txt').astype('int')
            sens_attr="race"
            predict_attr="income"
            print('Loading {} dataset'.format(args.dataset))
            header = list(idx_features_labels.columns) #list将括号里的内容变为数组
            header.remove(predict_attr) #header.remove删除括号内的东西
            feature = idx_features_labels[header]
            labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = ["race"])
            adj_start = time.time()
            idx = np.arange(feature.shape[0]) 
            idx_map = {j: i for i, j in enumerate(idx)}
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
            adj_end = time.time()
            
            sens = idx_features_labels[sens_attr].values.astype(int) 
            sens = torch.FloatTensor(sens) 
            feature = np.array(feature)
            feature = feature_normalize(feature)
            feature = torch.FloatTensor(feature)
            labels = torch.LongTensor(labels) 
            labels[labels >1] =1
            
            print('dataset:',args.dataset)
            print('sens:',sens_attr)
            print('feature:',feature.shape)
            print('labels:',torch.unique(labels))
            idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
            
            return adj, feature, labels, sens, idx_train, idx_val, idx_test
        
        elif args.dataset=='german':
            idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'german.csv'))
            edges_unordered = np.genfromtxt(datapath + dataname + 'german_edges.txt').astype('int')
            print('Loading {} dataset'.format(args.dataset))
            sens_attr="Gender"
            predict_attr="GoodCustomer"
            header = list(idx_features_labels.columns)
            header.remove(predict_attr)
            header.remove('OtherLoansAtStore')
            header.remove('PurposeOfLoan')
            
            idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
            idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0
            feature = idx_features_labels[header]
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = ["Gender"])
            
            # features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
            feature = sp.csr_matrix(feature, dtype=np.float32)
            labels = idx_features_labels[predict_attr].values
            labels[labels == -1] = 0
            
            adj_start = time.time()
            idx = np.arange(feature.shape[0])
            idx_map = {j: i for i, j in enumerate(idx)}
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
            
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
            adj_end = time.time()
            
            sens = idx_features_labels[sens_attr].values.astype(int) 
            sens = torch.FloatTensor(sens) 
            feature = np.array(feature.todense())
            # feature = feature_normalize(feature)
            feature = torch.FloatTensor(feature)
            labels = torch.LongTensor(labels) 
            
            print('dataset:',args.dataset)
            print('sens:',sens_attr)
            print('feature:',feature.shape)
            print('labels:',torch.unique(labels))
            idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
            
            return adj, feature, labels, sens, idx_train, idx_val, idx_test
        
        elif args.dataset=='bail':
            idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'bail.csv'))
            edges_unordered = np.genfromtxt(datapath + dataname + 'bail_edges.txt').astype('int')
            print('Loading {} dataset'.format(args.dataset))
            sens_attr="WHITE"
            predict_attr="RECID"
            header = list(idx_features_labels.columns)
            header.remove(predict_attr)
            
            feature = idx_features_labels[header]
            labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
            
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = ["WHITE"])
            
            adj_start = time.time()
            idx = np.arange(feature.shape[0])
            idx_map = {j: i for i, j in enumerate(idx)}
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
            adj_end = time.time()
            
            sens = idx_features_labels[sens_attr].values.astype(int) 
            sens = torch.FloatTensor(sens) 
            feature = np.array(feature)
            feature = feature_normalize(feature)
            feature = torch.FloatTensor(feature)
            labels = torch.LongTensor(labels) 
            
            print('dataset:',args.dataset)
            print('sens:',sens_attr)
            print('feature:',feature.shape)
            print('labels:',torch.unique(labels))
            idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
            
            return adj, feature, labels, sens, idx_train, idx_val, idx_test
           
    adj, feature, labels, sens, idx_train, idx_val, idx_test = load_dataset(args)
    print(f"num edges: {adj.size}")
    g = dgl.graph(adj.nonzero())
    g.ndata.update({
        "feature": feature,
        "label": labels,
        "sensitive": sens,
    })
    index_split = {
        "train_index": idx_train,
        "val_index": idx_val,
        "test_index": idx_test
    }
    return g, index_split

# def load_data(dataset):
#     dataset = dataset.lower()
#     assert dataset in ['pokec_z','pokec_n', 'dblp']
    
#     glist, _ = dgl.load_graphs(f'../data/{dataset}.bin')
#     g = glist[0]

#     idx_train = torch.where(g.ndata['train_index'])[0]
#     idx_val = torch.where(g.ndata['val_index'])[0]
#     idx_test = torch.where(g.ndata['test_index'])[0]
#     # g.ndata.pop('train_index')
#     # g.ndata.pop('val_index')
#     # g.ndata.pop('test_index')
#     index_split = {'train_index': idx_train,
#                     'val_index': idx_val,
#                     'test_index': idx_test}
#     return g, index_split


def fair_matrix(pred, label, sens, index):

    SP = []
    EO = []

    idx_d = torch.where(sens[index]==0)[0]
    idx_a = torch.where(sens[index]==1)[0]
    for i in range(label.max()+1):
        # SP
        p_i0 = torch.where(pred[index][idx_d] == i)[0]
        p_i1 = torch.where(pred[index][idx_a] == i)[0]

        sp = (p_i1.shape[0]/idx_a.shape[0]) - (p_i0.shape[0]/idx_d.shape[0])
        SP.append(sp)
        
        # EO
        p_y0 = torch.where(label[index][idx_d] == i)[0]
        p_y1 = torch.where(label[index][idx_a] == i)[0]

        p_iy0 = torch.where(pred[index][idx_d][p_y0] == i)[0]
        p_iy1 = torch.where(pred[index][idx_a][p_y1] == i)[0]

        if p_y0.shape[0] == 0 or p_y1.shape[0] == 0:
            eo = 0
        else:
            eo = (p_iy1.shape[0]/p_y1.shape[0]) - (p_iy0.shape[0]/p_y0.shape[0])
        EO.append(eo)   
    return SP, EO

import torch
import sys
from utils.multi_task_dataloader import MultiTaskDataLoader
from utils.multi_task_data import BERT4RecDataset
from ColdGPT import ColdGPT
from utils.parser import parse_args
import os
import pickle
from random import shuffle
import numpy as np
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def load_new_embeddings(new_embedding_path, device):
    '''
    get the embeddings of the training and test (scs) items as well as attrs
    '''
    if os.path.exists(new_embedding_path):
        with open(new_embedding_path, 'rb') as fp:
            new_embeddings = pickle.load(fp)
    else:
        num_train_items = mt_dataloader.num_train_items
        num_train_attrs = len(mt_dataloader.train_attrs_dict)
        
        train_graph = mt_dataloader.train_graph
        attr_embedding = mt_dataloader.train_attrs_embedding
        item_embedding = mt_dataloader.train_items_embedding
        
        model = ColdGPT(args = args, num_items = num_train_items, num_attrs = num_train_attrs, \
                graph = train_graph, device = device, item_embedding = item_embedding, attr_embedding = attr_embedding, \
                t1 = True, t2 = args.t2, t3 = args.t3)

        model.load_model(args.model_save_path)

        model.num_items += mt_dataloader.num_test_items
        model.num_attrs += len(mt_dataloader.new_test_attrs_dict)
        model.task1.encoder.item_init_embeddings = torch.cat((model.task1.encoder.item_init_embeddings, mt_dataloader.test_items_embedding), 0)
        model.task1.encoder.attr_init_embeddings = torch.cat((model.task1.encoder.attr_init_embeddings, mt_dataloader.new_test_attrs_embedding), 0)
        model.task1.graph = mt_dataloader.test_graph.to(device)
        new_embeddings = model.task1.extract_embeddings().clone().detach()

        with open(new_embedding_path, 'wb') as fp:
            pickle.dump(new_embeddings, fp)

    item_embeddings = new_embeddings[:(mt_dataloader.num_train_items + mt_dataloader.num_test_items)]
    attr_embeddings = new_embeddings[(mt_dataloader.num_train_items + mt_dataloader.num_test_items):]

    assert len(attr_embeddings) == len(mt_dataloader.train_attrs_dict) + len(mt_dataloader.new_test_attrs_dict)
    return item_embeddings, attr_embeddings


def cal_dcg(result):
    '''
    calculate dcg
    '''
    discount = torch.tensor(range(len(result))).to(result.device) + 2
    discount = discount.log2()
    return (result / discount).sum()


def build_dataloader(args, data, device):
    dic_user = {}
    for line in data:
        user = line[0]
        item = line[1]
        rating = line[2]
        if user in dic_user:
            dic_user[user].append([item, rating])
        else:
            dic_user[user] = [[item, rating]]
    for user in dic_user:
        dic_user[user] = np.array(dic_user[user])

    users = torch.LongTensor(data[:, 0]).to(device)
    items = torch.LongTensor(data[:, 1]).to(device)
    ratings = torch.Tensor(data[:, 2]).to(device)
    data = TensorDataset(users, items, ratings)
    data_loader = DataLoader(data, batch_size = args.pretrain_batch_size, shuffle = True)

    return data_loader, dic_user


def pred_ranking_new(new_item_embedding, device):
    train_triples, test_triples, val_triples, pred_triples = mt_dataloader.get_rec_data()
    pred_triples[:, 1] -= mt_dataloader.num_val_items # map the triples to bypass the validation items

    user_embeddings = torch.mean(new_item_embedding[:mt_dataloader.num_train_items], 0)
    user_embeddings = user_embeddings.repeat(mt_dataloader.num_user, 1)
    _, train_test_dic = build_dataloader(args, np.concatenate((train_triples, test_triples), axis=0), device)
    train_test_user_neighbor_dict = {int(user): torch.tensor([int(item) for [item, rating] in ratings]).to(device) for user, ratings in train_test_dic.items()}
    train_test_user_neighbor_dict = {k: v for k, v in sorted(train_test_user_neighbor_dict.items(), key=lambda item: item[0])}
    train_test_user_embedding_dict = {user: torch.mean(torch.index_select(new_item_embedding, 0, neighbors), 0) for user, neighbors in train_test_user_neighbor_dict.items()}
    users = torch.tensor(list(train_test_user_embedding_dict.keys())).to(device)
    user_embeddings[users] = torch.stack(list(train_test_user_embedding_dict.values()), 0)

    result_dic_ndcg = {}
    result_dic_recall = {}
    for k in args.k_list:
        result_dic_ndcg[k] = 0
        result_dic_recall[k] = 0
    
    _, pre_dic = build_dataloader(args, pred_triples, device)
    for user in pre_dic.keys():
        gt = torch.zeros(len(new_item_embedding) - mt_dataloader.num_train_items)
        gt[pre_dic[user][:, 0] - mt_dataloader.num_train_items] = torch.tensor(pre_dic[user][:, 1], dtype = gt.dtype)
        items = [i+mt_dataloader.num_train_items for i in range(len(new_item_embedding) - mt_dataloader.num_train_items)]
        users = [user] * len(items)
        users = user_embeddings[users]
        items = new_item_embedding[items]
        ratings_pred = torch.mul(users, items).sum(1)
        values, indices = torch.topk(ratings_pred, k = max(args.k_list))
        gt_values, gt_indices = torch.topk(gt, k = max(args.k_list))
        for k in args.k_list:
            result = gt[indices]
            idcg = cal_dcg(gt_values[:k])
            dcg = cal_dcg(result[:k])
            ndcg = dcg / idcg
            recall = (result[:k] != 0).sum() / (gt != 0).sum()
            result_dic_ndcg[k] += ndcg
            result_dic_recall[k] += recall

    for key in result_dic_ndcg:
        result_dic_ndcg[key] = result_dic_ndcg[key] / len(pre_dic)
        result_dic_recall[key] = result_dic_recall[key] / len(pre_dic)

    return result_dic_ndcg, result_dic_recall


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cpu')
    mt_dataloader = MultiTaskDataLoader(path = args.data_path, \
        plm = args.plm, t1 = True, t2 = args.t2, t3 = args.t3)
    new_item_embedding, _ = load_new_embeddings(new_embedding_path = './new_embedding/new_embeddings.pkl', device = device)
    new_item_embedding = F.normalize(new_item_embedding, p=2.0, dim = 0)
    new_item_embedding = new_item_embedding.to(device)
    result_dic_ndcg, result_dic_recall = pred_ranking_new(new_item_embedding = new_item_embedding, device = device)
    print('result_dic_ndcg: ', result_dic_ndcg)
    print('result_dic_recall: ', result_dic_recall)


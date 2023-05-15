import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import *
import numpy as np
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from BERT4Rec import BERT4Rec
from torch_geometric.nn import GCNConv, GATConv
from utils.parser import parse_args
from utils.multi_task_data import NodePairDataset, BERT4RecDataset, MultiTaskDataset, MultiTaskBatchSampler
from utils.EarlyStop import EarlyStoppingCriterion
from utils.InfoNCE_loss import InfoNCE_loss
from utils.multi_task_dataloader import MultiTaskDataLoader
from tqdm import tqdm
import torch.optim as optim
import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

class Encoder(nn.Module):
    def __init__(self, args, item_init_embeddings, attr_init_embeddings, out_dim):
        super().__init__()
        self.item_init_embeddings = item_init_embeddings
        self.attr_init_embeddings = attr_init_embeddings
        self.in_dim = self.item_init_embeddings.size(1)
        self.out_dim = out_dim
        self.args = args
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.ELU(),
            nn.Linear(self.in_dim, self.in_dim),
            nn.ELU(),
            nn.Linear(self.in_dim, out_dim)
        )
        self.layers.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    
    def get_item_embeddings(self):
        item_init_embeddings = self.item_init_embeddings
        return self.layers(item_init_embeddings)
    
    def get_attr_embeddings(self):
        attr_init_embeddings = self.attr_init_embeddings
        return self.layers(attr_init_embeddings)

class Task1(nn.Module):
    def __init__(self, args, graph, device, item_embedding, attr_embedding):
        super(Task1, self).__init__()
        self.args = args
        self.encoder = Encoder(args = args, item_init_embeddings = item_embedding, attr_init_embeddings = attr_embedding, \
            out_dim = args.embed_dim)

        self.device = device
        self.graph = graph
        self.conv = self.build_gnn_layer()
        self.edge_dropout = args.edge_dropout
    
    def build_gnn_layer(self):
        return GCNConv(self.args.embed_dim, self.args.embed_dim)

    def extract_embeddings(self):
        edges = self.graph.edge_index
        node_embedding = torch.cat((self.encoder.get_item_embeddings(), self.encoder.get_attr_embeddings()), 0)
        result_embeddings = self.conv(node_embedding, edges)
        
        return result_embeddings

    def forward(self, inputs):
        inputs = inputs.to(self.device)
        edges = self.graph.edge_index
        embeddings = torch.cat((self.encoder.get_item_embeddings(), self.encoder.get_attr_embeddings()), 0)
        embeddings = self.conv(embeddings, edges)

        x = embeddings[inputs[:, 0]]
        y = embeddings[inputs[:, 1]]
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        loss = (x - y).norm(p=2, dim=1).pow(2).mean()
        return loss, embeddings

'''
The pre-training tasks:
task 1: self-supervised learning on attribute-item graph
task 2: predict masked items in users' purchase history
task 3: self-supervised learning on review term-item graph
'''
class ColdGPT(nn.Module):
    def __init__(self, args, num_items, num_attrs, graph, device, item_embedding, attr_embedding, \
            t1 = True, t2 = True, t3 = True, task3_graph = None, filtered_np_embedding = None):
        super(ColdGPT, self).__init__()

        # items and attributes embedding layers
        self.args = args
        self.device = device
        self.num_items = num_items
        self.num_attrs = num_attrs
        self.embed_dim = args.embed_dim

        self.task1 = Task1(args, graph, device, item_embedding, attr_embedding)
        # for task 2 (predict masked items in users' purchase history)
        if t2:
            self.BERT4Rec = BERT4Rec(args, self.task1, num_items)
        # for task 3 (predict items using reviews)
        if t3:
            self.task3_graph = task3_graph
            self.filtered_np_embedding = filtered_np_embedding
            self.task3_conv = GCNConv(self.args.embed_dim, self.args.embed_dim)

    def forward(self, task_type, inputs):
        if task_type == 1:
            loss, embeddings = self.task1(inputs)

        if task_type == 2:
            x, y = inputs
            x = x.to(self.device)
            y = y.to(self.device)
            
            logits, embeddings = self.BERT4Rec(x)
            y = y.view(-1)
            
            indices = torch.where(y != 0)[0]
            x = torch.index_select(logits, 0, indices)
            y = torch.index_select(y, 0, indices)
            y -= 1
            y = embeddings[y]
            
            x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
            loss = (x - y).norm(p=2, dim=1).pow(2).mean()

        if task_type == 3:
            edges = self.task3_graph.edge_index
            node_embedding = torch.cat((self.task1.encoder.get_item_embeddings(), self.task1.encoder.layers(self.filtered_np_embedding)), 0)

            embeddings = self.task3_conv(node_embedding, edges)
            x = embeddings[inputs[:, 0]]
            y = embeddings[inputs[:, 1]]
                
            x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
            loss = (x - y).norm(p=2, dim=1).pow(2).mean()
        
        return loss, embeddings
    
    def predict(self, new_graph, new_item_embedding, new_attr_embedding):
        self.num_items += new_item_embedding.size(0)
        self.num_attrs += new_attr_embedding.size(0)
        self.task1.encoder.item_init_embeddings = torch.cat((self.task1.encoder.item_init_embeddings, new_item_embedding), 0)
        self.task1.encoder.attr_init_embeddings = torch.cat((self.task1.encoder.attr_init_embeddings, new_attr_embedding), 0)
        self.graph = new_graph.to(device)
        return self.task1.extract_embeddings()[:self.num_items].clone().detach() # embeddings of all the items (including training and test items)

    def load_model(self, path):
        model_dict = torch.load(path)
        self.load_state_dict(model_dict)

    def regularization_loss(self, task_type, inputs, embeddings):

        if task_type == 1:
            inputs = inputs.to(self.device)
            item_ids = torch.unique(inputs[:, 0])
            attr_ids = torch.unique(inputs[:, 1])
            
            item_embedding_batch = torch.index_select(embeddings, 0, item_ids)
            attr_embedding_batch = torch.index_select(embeddings, 0, attr_ids)
            item_embedding_batch_normed = F.normalize(item_embedding_batch, dim=-1)
            attr_embedding_batch_normed = F.normalize(attr_embedding_batch, dim=-1)

            loss_reg = torch.pdist(item_embedding_batch_normed, p=2).pow(2).mul(-2).exp().mean().log()
            loss_reg += torch.pdist(attr_embedding_batch_normed, p=2).pow(2).mul(-2).exp().mean().log()
        elif task_type == 2:
            _, y = inputs
            y = set(y.view(-1).tolist())
            y.discard(0)
            y = [item-1 for item in list(y)]
            y = torch.LongTensor(y)
            y = y.to(self.device)
            
            item_embedding_batch = torch.index_select(embeddings, 0, y)
            item_embedding_batch_normed = F.normalize(item_embedding_batch, dim=-1)
            loss_reg = torch.pdist(item_embedding_batch_normed, p=2).pow(2).mul(-2).exp().mean().log()
            
        elif task_type == 3:
            inputs = inputs.to(self.device)
            item_ids = torch.unique(inputs[:, 0])
            np_ids = torch.unique(inputs[:, 1])
                
            item_embedding_batch = torch.index_select(embeddings, 0, item_ids)
            np_embedding_batch = torch.index_select(embeddings, 0, np_ids)
            item_embedding_batch_normed = F.normalize(item_embedding_batch, dim=-1)
            np_embedding_batch_normed = F.normalize(np_embedding_batch, dim=-1)

            loss_reg = torch.pdist(item_embedding_batch_normed, p=2).pow(2).mul(-2).exp().mean().log()
            loss_reg += torch.pdist(np_embedding_batch_normed, p=2).pow(2).mul(-2).exp().mean().log()

        return loss_reg


if __name__ == "__main__":
    args = parse_args()
    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + str(args.gpu))
    logging.basicConfig(filename=args.log_path,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
    logging.info('---------------- args ----------------')
    for arg in vars(args):
        print(str(arg) + ': ' + str(getattr(args, arg)))
        logging.info(str(arg) + ': ' + str(getattr(args, arg)))
    logging.info('--------------------------------------')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mt_dataloader = MultiTaskDataLoader(path = args.data_path, \
        max_len = args.BERT4Rec_max_len, plm = args.plm, t1 = True, t2 = args.t2, t3 = args.t3)
    num_train_items = mt_dataloader.num_train_items
    num_train_attrs = len(mt_dataloader.train_attrs_dict)
    train_graph = mt_dataloader.train_graph.to(device)
    attr_embedding = mt_dataloader.train_attrs_embedding.to(device)
    item_embedding = mt_dataloader.train_items_embedding.to(device)

    if args.t3:
        task3_graph = mt_dataloader.task3_graph.to(device)
        filtered_np_embedding = mt_dataloader.filtered_np_embedding.to(device)

    model = ColdGPT(args = args, num_items = num_train_items, num_attrs = num_train_attrs, \
        graph = train_graph, device = device, item_embedding = item_embedding, attr_embedding = attr_embedding, \
        t1 = True, t2 = args.t2, t3 = args.t3, task3_graph = task3_graph, filtered_np_embedding = filtered_np_embedding)

    if args.model_load_path != '':
        model.load_model(args.model_load_path)
    model.to(device)

    # Pre-training
    logging.info("---------------- Pre-training hete model ----------------")
    task1_dataset, task2_dataset, task3_dataset = None, None, None
    if args.t1:
        task1_dataset = NodePairDataset(mt_dataloader.task1_list)
    if args.t2:
        task2_dataset = BERT4RecDataset(args, mt_dataloader.task2_list, num_train_items)
    if args.t3:
        task3_dataset = NodePairDataset(mt_dataloader.task3_list)
    datasets = [task1_dataset, task2_dataset, task3_dataset]
    multi_task_train_dataset = MultiTaskDataset(datasets)
    multi_task_batch_sampler = MultiTaskBatchSampler(datasets, args)
    multi_task_train_data = torch.utils.data.DataLoader(multi_task_train_dataset, batch_sampler=multi_task_batch_sampler)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = args.weight_decay) # default eps=1e-08, weight_decay=0.01
    early_stop = EarlyStoppingCriterion(patience = args.early_stop, save_path = args.model_save_path, delta = args.early_stop_delta)
        
    epoch = 0
    model.train()
    model.training = True
    while True:
        epoch += 1
        loss_epoch = 0
        # to calculate the average loss of each task
        loss_type_epoch = [0, 0, 0] 
        num_batches_type = [0, 0, 0]

        tqdm_multi_task_train_data = tqdm(multi_task_train_data)
        for step, batch in enumerate(tqdm_multi_task_train_data):
            if step % 10000 == 0:
                logging.info("Epoch %d, step %d/%d." % (epoch, step, len(multi_task_train_data)))
            task_type, inputs = batch
            task_type = task_type[0].item()
            num_batches_type[task_type-1] += 1
            loss, embeddings = model(task_type, inputs)

            if task_type == 1:
                loss *= args.w_t1
            elif task_type == 2:
                loss *= args.w_t2
            elif task_type == 3:
                loss *= args.w_t3
            else:
                raise Exception('Wrong task type')

            loss_reg = args.lambda_reg * model.regularization_loss(task_type, inputs, embeddings)
            loss += loss_reg

            loss_epoch += loss.detach()
            loss_type_epoch[task_type-1] += loss.detach()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            tqdm_multi_task_train_data.set_description(
            'Epoch {}, loss {:.3f} '.format(epoch, loss_epoch/(step+1)))

        # Average loss of this epoch
        loss_epoch /= step
        logging.info("---------- Pre-training loss of epoch %d ----------" % (epoch))
        logging.info("Overall loss: %.6f" % (loss_epoch))
        # Average loss of each task in this epoch
        if args.t1:
            loss_task = loss_type_epoch[0]/num_batches_type[0]
            logging.info("Task %d loss: %.6f" % (1, loss_task))
        if args.t2:
            loss_task = loss_type_epoch[1]/num_batches_type[1]
            logging.info("Task %d loss: %.6f" % (2, loss_task))
        if args.t3:
            loss_task = loss_type_epoch[2]/num_batches_type[2]
            logging.info("Task %d loss: %.6f" % (3, loss_task))

        # early stop
        early_stop(loss_epoch, model)
        if early_stop.early_stop:
            logging.info("Pretrain Early stopping. Epochs:%d early_stop_loss:%.6f" % (epoch, early_stop.best_loss))
            break
        
    model.load_model(args.model_save_path)
    model.to(device)

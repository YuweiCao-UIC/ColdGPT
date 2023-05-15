import logging
logging.basicConfig(level=logging.DEBUG)
import random
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import sys
import torch_geometric as pyg
from torch_geometric.data import Data
import pickle
import pandas as pd
logging.basicConfig(stream = sys.stdout, level = logging.INFO)
import matplotlib.pyplot as plt
import math
from os.path import exists
import os
import json
from random import shuffle
from sentence_transformers import SentenceTransformer
import re

def SBERT_embed(s_list):
    '''
    Use Sentence-BERT to embed sentences.
    s_list: a list of sentences/ tokens to be embedded.
    output: the embeddings of the sentences/ tokens.
    '''
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(s_list, convert_to_tensor = True, normalize_embeddings = True)
    return embeddings.cpu()

def BERT_embed(sentences, batch_size = 5):
    '''
    Use BERT to embed sentences.
    sentences: a list of sentences/ tokens to be embedded.
    output: [CLS] (pooler_output) of the embedded sentences/ tokens.
    refer to the following post for an introduction of pooler/[CLS]/mean-pooling:
    https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
    '''
    # BERT
    tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
    BERT_model = AutoModel.from_pretrained("bert-large-cased")
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    #BERT_model = AutoModel.from_pretrained("bert-base-cased")
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #BERT_model = AutoModel.from_pretrained("bert-base-uncased")

    # mBERT
    #tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    #BERT_model = BertModel.from_pretrained("bert-base-multilingual-cased")

    output_pooler = []
    output_cls = []
    output_mean = []
    sentence_loader = torch.utils.data.DataLoader(sentences, batch_size = batch_size, shuffle = False)
    for batch_sentence in tqdm(sentence_loader):
        #print('batch_sentence: ', batch_sentence)
        inputs_ids = tokenizer(batch_sentence, padding=True, truncation=True, return_tensors="pt")
        BERT_output = BERT_model(**inputs_ids)
        output_pooler.append(BERT_output[1].detach())
        last_hidden_state = BERT_output[0].detach()
        #print('last_hidden_state: ', last_hidden_state)
        output_cls.append(last_hidden_state[:, 0])
        attention_mask = inputs_ids['attention_mask']
        #print('attention_mask: ', attention_mask)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        #print('mean_embeddings: ', mean_embeddings)
        output_mean.append(mean_embeddings)
    embeddings_pooler = torch.cat(output_pooler, dim=0)
    #print('embeddings_pooler: ', embeddings_pooler)
    embeddings_cls = torch.cat(output_cls, dim=0)
    #print('embeddings_cls: ', embeddings_cls)
    embeddings_mean = torch.cat(output_mean, dim=0)
    #print('embeddings_mean: ', embeddings_mean)
    return embeddings_pooler, embeddings_cls, embeddings_mean

def strip(alist):
    return [each.strip('\', \"') for each in alist]

class MultiTaskDataLoader():
    """ 
    Generate datasets for multi tasks
    """
    def __init__(self, path, filter_threshold = 20, max_len = 100, plm = 'BERT', \
            t1 = True, t2 = True, t3 = True):

        processed_data_path = path + 'processed_data/'
        raw_data_path = path + 'raw_data/'

        # raw data files
        self.item_path = raw_data_path + 'Atlanta_all_item.txt'
        self.ratings_path = raw_data_path + 'Atlanta_ratings.txt'
        self.attrs_path = raw_data_path + 'Atlanta_all_item_attrs.csv'
        self.reviews_path = raw_data_path + 'Atlanta_reviews.json'

        # pre-precessed files
        self.attrs_dict_path = processed_data_path + 'attrs_dict_graph/'
        self.attrs_embedding_path = processed_data_path + 'attrs_embedding/'
        self.items_embedding_path = processed_data_path + 'items_embedding/'
        self.task3_path = processed_data_path + 'task3_path/'


        # read in the ratings, filter them accoring to the sequence length, then
        # sort them according to sequence length
        # self.i_ratings format: {{mapped_item_id0: {mapped_user_id10: 3.0, mapped_user_id112: 4.0, ...}}, ...}
        # self.map_i_o2n format: {'NeXXwr0jU3t2KcywWRGcwg': 0, '61JT4z85HnhOPtvJfQpcFg': 1, ...}
        self.i_ratings, self.map_i_o2n, self.map_u_o2n = self.read_ratings(path = self.ratings_path, n = filter_threshold)
        self.map_i_n2o = {v:k for k,v in self.map_i_o2n.items()}
        self.map_u_n2o = {v:k for k,v in self.map_u_o2n.items()}
        self.num_user = len(self.map_u_o2n)
        self.num_item = len(self.map_i_o2n)

        # split self.i_ratings (already sorted by length in self.read_ratings) into train/val/test sets 
        # 90% as training, 10% as test. 5% of the training set are used for validation
        # self.train_i_ratings, self.val_i_ratings, self.test_i_ratings format: same as self.i_ratings
        # self.train_users: a list of users involved in the training set
        self.train_i_ratings, self.val_i_ratings, self.test_i_ratings, self.train_users, \
            self.num_train_items, self.num_val_items, self.num_test_items, \
            self.num_train_ratings, self.num_val_ratings, self.num_test_ratings = self.split_ratings()

        if t1:
            logging.info("task1: generate item-attribute graph")
            #self.all_attrs_dict format: {raw_item_id: [attr1, attr2, ...], ...}
            self.all_attrs_dict = self.read_attributes(self.attrs_path)
            self.train_item_attrs_dict, self.val_item_attrs_dict, self.test_item_attrs_dict, \
                self.train_attrs_dict, self.new_val_attrs_dict, self.new_test_attrs_dict, \
                self.train_attrs_embedding, self.new_val_attrs_embedding, self.new_test_attrs_embedding, \
                self.train_items_embedding, self.val_items_embedding, self.test_items_embedding, \
                self.train_graph, self.train_edges, self.val_graph, self.test_graph, self.task1_list = self.task1(plm = plm)

        if t2:
            logging.info("task2: generate user-item interaction sequence")
            self.task2_list, self.task2_pred_dic = self.task2(max_len = max_len)
        
        if t3:
            self.filtered_np_embedding, self.task3_graph, self.task3_list = \
                self.task3_phrase_senti(embedding_save_path = self.task3_path, plm = plm)


    def split_ratings(self):
        # 0~val_start: training items; val_start~train_end: val items; train_end~len(filtered_i_ratings_sorted): test items;
        train_end = math.ceil(0.9*len(self.i_ratings))
        val_start = math.ceil(0.95*train_end)

        # split the ratings
        train_i_ratings, val_i_ratings, test_i_ratings = {}, {}, {}
        for item in list(self.i_ratings.keys())[:val_start]:
            train_i_ratings[item] = self.i_ratings[item]
        for item in list(self.i_ratings.keys())[val_start:train_end]:
            val_i_ratings[item] = self.i_ratings[item]
        for item in list(self.i_ratings.keys())[train_end:]:
            test_i_ratings[item] = self.i_ratings[item]
        num_train_items = len(train_i_ratings)
        num_val_items = len(val_i_ratings)
        num_test_items = len(test_i_ratings)
        
        # get a set of unique users in the training set
        train_users = []
        for item, sequence in train_i_ratings.items():
            train_users += list(sequence.keys())
        train_users = set(train_users)
        if len(train_users) != self.num_user:
            print('------- '+ str(self.num_user - len(train_users)) + ' val/test users are not involved in the training set ------')

        # filter val_i_ratings and test_i_ratings to make sure that all the users involved 
        # in them are also in the train_i_ratings (i.e., are old users)
        # count the number of train, val, and test ratings
        num_train_ratings, num_val_ratings, num_test_ratings = 0, 0, 0
        for item, sequence in train_i_ratings.items():
            num_train_ratings += len(sequence)
        for item, sequence in val_i_ratings.items():
            val_i_ratings[item] = {user:rating for user, rating in sequence.items() if user in train_users}
            num_val_ratings += len(val_i_ratings[item])
        for item, sequence in test_i_ratings.items():
            test_i_ratings[item] = {user:rating for user, rating in sequence.items() if user in train_users}
            num_test_ratings += len(test_i_ratings[item])
        
        return train_i_ratings, val_i_ratings, test_i_ratings, train_users, \
            num_train_items, num_val_items, num_test_items, num_train_ratings, num_val_ratings, num_test_ratings

        
    def read_ratings(self, path, n = 20):
        print('Inside read_ratings')
        # all_u_ratings format: {{raw_user_id0: {raw_item_id8: 2.0, raw_item_id666: 5.0, ...}}, ...}
        all_u_ratings = {}
        # filtered and mapped rating sequences
        # i_ratings format: {{mapped_item_id0: {mapped_user_id10: 3.0, mapped_user_id112: 4.0, ...}}, ...}
        # u_ratings format: {{mapped_user_id0: {mapped_item_id8: 2.0, mapped_item_id666: 5.0, ...}}, ...}
        i_ratings, u_ratings = {}, {}

        # get the raw sequences
        with open(path, 'r') as f:
            lines = f.readlines()
            #lines = lines[:50] # testing
            for line in tqdm(lines):
                #print(line)
                line = line.strip().split('\t')
                [user, item, rating, timestamp] = line
                #print(user, item, rating)
                if user not in all_u_ratings.keys():
                    all_u_ratings[user] = {}
                all_u_ratings[user][item] = float(rating)

        # filter data based on sequence length
        if n is not None:
            # filter out users with too short sequences
            for user, sequence in all_u_ratings.items():
                if len(sequence) >= n:
                    u_ratings[user] = sequence
            
            # organize the filtered data by items
            unfiltered_i_ratings = {}
            for user, sequence in u_ratings.items():
                for item, rating in sequence.items():
                    if item not in unfiltered_i_ratings:
                        unfiltered_i_ratings[item] = {}
                    unfiltered_i_ratings[item][user] = rating
            
            for item, sequence in unfiltered_i_ratings.items():
                if len(sequence) >= n:
                    i_ratings[item] = sequence
        
            # sort i_ratings based on sequence length
            i_ratings_sorted = {}
            for k in sorted(i_ratings, key=lambda k: len(i_ratings[k]), reverse=True):
                i_ratings_sorted[k] = i_ratings[k]
            #print("len(i_ratings_sorted), should be the same as len(i_ratings): ", len(i_ratings_sorted))
            '''
            print("Two longest sequences:")
            for item in list(i_ratings_sorted.keys())[:2]:
                print(i_ratings_sorted[item])
            print("Two shortest sequences:")
            for item in list(i_ratings_sorted.keys())[-2:]:
                print(i_ratings_sorted[item])
            '''

            # generate id maps
            map_i_o2n = {old_id:new_id for new_id, old_id in enumerate(i_ratings_sorted)}
            map_u_o2n = {old_id:new_id for new_id, old_id in enumerate(u_ratings)}
            #print('# of items: ', len(map_i_o2n))
            #print('# of users: ', len(map_u_o2n))

            # mapping
            i_ratings_mapped = {}
            for item, sequence in i_ratings_sorted.items():
                i_ratings_mapped[map_i_o2n[item]] = {map_u_o2n[user]:rating for user, rating in i_ratings_sorted[item].items()}

        return i_ratings_mapped, map_i_o2n, map_u_o2n
        

    def read_attributes(self, path):
        '''
        Read in the attributes of all the items.
        Return 
        '''
        df = pd.read_csv(path)
        attrs = [strip(each.strip("{}'").split(", ")) if isinstance(each, str) and each != 'set()' else [] for each in df['attributes'].tolist()]
        cats = [strip(each.strip("{}'").split(", ")) if isinstance(each, str) and each != 'set()' else [] for each in df['categories'].tolist()]
        all_attrs = []
        for i in range(len(attrs)):
            temp = []
            temp.extend(attrs[i])
            temp.extend(cats[i])
            all_attrs.append(temp)
        df['all_attrs'] = all_attrs
        #attrs_df = df.loc[:, ['business_id', 'all_attrs']]
        #print(attrs_df.head(10))
        all_attrs_dict = {raw_item_id:attrs for raw_item_id, attrs in zip(df['business_id'], df['all_attrs'])}
        #print(attrs_dict)
        return all_attrs_dict


    def get_attrs_for_items(self, item_list):
        '''
        Take a list of (new, i.e., numerical) item ids and return a dict of item attributes
        item_attrs_dict format: {new_item_id: [attr1, attr2, ...], ...}
        '''
        item_attrs_dict = {new_item_id:self.all_attrs_dict[self.map_i_n2o[new_item_id]] for new_item_id in item_list}
        return item_attrs_dict


    def generate_item_attribute_graph(self, item_attrs_dict, attrs_dict, mode = 'train', train_edges = None):
        '''
        Construct and return item-attribute bipartite graph (pyg graph).
        item_attrs_dict: {2636: ['WheelchairAccessible', 'BusinessAcceptsCreditCards', ...], ...}
        attrs_dict: {'WheelchairAccessible': 0, 'BusinessAcceptsCreditCards': 1, ...}
        '''
        edges = [] 
        for item, attrs in item_attrs_dict.items():
            for attr in attrs:
                edges.append([item, attrs_dict[attr]])
        
        # an original list of all the [item_id, attr_id] before mapping
        edges = torch.tensor(edges, dtype = torch.long).t() 
        if mode == 'train':
            train_edges = edges.detach().clone()
        
        # map the atrribute nodes
        if mode == 'train': # generate the train graph
            edges[1] += self.num_train_items
        elif mode == 'valid': # generate the valid graph
            edges = torch.cat((train_edges, edges), 1)
            edges[1] += self.num_train_items
            edges[1] += self.num_val_items
        elif mode == 'test': # generate the test graph
            edges[0] -= self.num_val_items
            edges = torch.cat((train_edges, edges), 1)
            edges[1] += self.num_train_items
            edges[1] += self.num_test_items

        undirected_edges = pyg.utils.to_undirected(edges)
        graph = Data(edge_index = undirected_edges)

        if mode == 'train': 
            return graph, train_edges
        return graph, None


    def task1(self, plm = 'BERT', embed_type = 'pooler'): # embed_type can be 'pooler' or 'cls' or 'mean'

        #================== item_attrs_dict, attrs_dict, graph, edge ===================
        train_item_attrs_dict_path = self.attrs_dict_path + 'train_item_attrs_dict.pkl'
        val_item_attrs_dict_path = self.attrs_dict_path + 'val_item_attrs_dict.pkl'
        test_item_attrs_dict_path = self.attrs_dict_path + 'test_item_attrs_dict.pkl'

        train_attrs_dict_path = self.attrs_dict_path + 'train_attrs_dict.pkl'
        new_val_attrs_dict_path = self.attrs_dict_path + 'new_val_attrs_dict.pkl'
        new_test_attrs_dict_path = self.attrs_dict_path + 'new_test_attrs_dict.pkl'

        train_graph_path = self.attrs_dict_path + 'train_graph.pkl'
        train_edges_path = self.attrs_dict_path + 'train_edges.pkl'
        val_graph_path = self.attrs_dict_path + 'val_graph.pkl'
        test_graph_path = self.attrs_dict_path + 'test_graph.pkl'

        if os.path.exists(train_item_attrs_dict_path):
            print('Loading...')
            with open(train_item_attrs_dict_path, 'rb') as fp:
                train_item_attrs_dict = pickle.load(fp)
            with open(val_item_attrs_dict_path, 'rb') as fp:
                val_item_attrs_dict = pickle.load(fp)
            with open(test_item_attrs_dict_path, 'rb') as fp:
                test_item_attrs_dict = pickle.load(fp)

            with open(train_attrs_dict_path, 'rb') as fp:
                train_attrs_dict = pickle.load(fp)
            with open(new_val_attrs_dict_path, 'rb') as fp:
                new_val_attrs_dict = pickle.load(fp)
            with open(new_test_attrs_dict_path, 'rb') as fp:
                new_test_attrs_dict = pickle.load(fp)
            
            with open(train_graph_path, 'rb') as fp:
                train_graph = pickle.load(fp)
            with open(train_edges_path, 'rb') as fp:
                train_edges = pickle.load(fp)
            with open(val_graph_path, 'rb') as fp:
                val_graph = pickle.load(fp)
            with open(test_graph_path, 'rb') as fp:
                test_graph = pickle.load(fp)

        else:
            print('Generating...')
            # get the attributes of training, val, and test items
            train_item_attrs_dict = self.get_attrs_for_items(list(self.train_i_ratings.keys()))
            val_item_attrs_dict = self.get_attrs_for_items(list(self.val_i_ratings.keys()))
            test_item_attrs_dict = self.get_attrs_for_items(list(self.test_i_ratings.keys()))

            with open(train_item_attrs_dict_path, 'wb') as fp:
                pickle.dump(train_item_attrs_dict, fp)
            with open(val_item_attrs_dict_path, 'wb') as fp:
                pickle.dump(val_item_attrs_dict, fp)
            with open(test_item_attrs_dict_path, 'wb') as fp:
                pickle.dump(test_item_attrs_dict, fp)

            # get the attr dicts for training, val, and test
            train_attrs_set, val_attrs_set, test_attrs_set = [], [], []
            for item, attrs in train_item_attrs_dict.items():
                train_attrs_set += attrs
            train_attrs_set = set(train_attrs_set)
            train_attrs_dict = {attr:id for id, attr in enumerate(list(train_attrs_set))}

            for item, attrs in val_item_attrs_dict.items():
                val_attrs_set += attrs
            val_attrs_set = set(val_attrs_set)
            # a set of new val item attributes that are not in the training set
            new_val_attrs_set = val_attrs_set - train_attrs_set
            new_val_attrs_dict = {attr:id+len(train_attrs_dict) for id, attr in enumerate(list(new_val_attrs_set))}

            for item, attrs in test_item_attrs_dict.items():
                test_attrs_set += attrs
            test_attrs_set = set(test_attrs_set)
            # a set of new test item attributes that are not in the training set
            new_test_attrs_set = test_attrs_set - train_attrs_set
            new_test_attrs_dict = {attr:id+len(train_attrs_dict) for id, attr in enumerate(list(new_test_attrs_set))}

            with open(train_attrs_dict_path, 'wb') as fp:
                pickle.dump(train_attrs_dict, fp)
            with open(new_val_attrs_dict_path, 'wb') as fp:
                pickle.dump(new_val_attrs_dict, fp)
            with open(new_test_attrs_dict_path, 'wb') as fp:
                pickle.dump(new_test_attrs_dict, fp)

            # generate graphs for training, val, and test
            train_graph, train_edges = self.generate_item_attribute_graph(item_attrs_dict = train_item_attrs_dict, \
                attrs_dict = train_attrs_dict, mode = 'train')
            val_graph, _ = self.generate_item_attribute_graph(item_attrs_dict = val_item_attrs_dict, \
                attrs_dict = {**train_attrs_dict, **new_val_attrs_dict}, mode = 'valid', train_edges = train_edges)
            test_graph, _ = self.generate_item_attribute_graph(item_attrs_dict = test_item_attrs_dict, \
                attrs_dict = {**train_attrs_dict, **new_test_attrs_dict}, mode = 'test', train_edges = train_edges)

            with open(train_graph_path, 'wb') as fp:
                pickle.dump(train_graph, fp)
            with open(train_edges_path, 'wb') as fp:
                pickle.dump(train_edges, fp)
            with open(val_graph_path, 'wb') as fp:
                pickle.dump(val_graph, fp)
            with open(test_graph_path, 'wb') as fp:
                pickle.dump(test_graph, fp)

        #================== attrs_embedding ===================
        if plm == 'BERT':
            train_attrs_embedding_path = self.attrs_embedding_path + 'train_attrs_embedding_' + embed_type + '.pkl'
            new_val_attrs_embedding_path = self.attrs_embedding_path + 'new_val_attrs_embedding_' + embed_type + '.pkl'
            new_test_attrs_embedding_path = self.attrs_embedding_path + 'new_test_attrs_embedding_' + embed_type + '.pkl'

        elif plm == 'SBERT':
            train_attrs_embedding_path = self.attrs_embedding_path + 'train_attrs_embedding_sbert.pkl'
            new_val_attrs_embedding_path = self.attrs_embedding_path + 'new_val_attrs_embedding_sbert.pkl'
            new_test_attrs_embedding_path = self.attrs_embedding_path + 'new_test_attrs_embedding_sbert.pkl'

        if os.path.exists(train_attrs_embedding_path):
            with open(train_attrs_embedding_path, 'rb') as fp:
                train_attrs_embedding = pickle.load(fp)
            with open(new_val_attrs_embedding_path, 'rb') as fp:
                new_val_attrs_embedding = pickle.load(fp)
            with open(new_test_attrs_embedding_path, 'rb') as fp:
                new_test_attrs_embedding = pickle.load(fp)
        else:
            if plm == 'BERT':
                train_attrs_embedding_pooler_path = self.attrs_embedding_path + 'train_attrs_embedding_pooler.pkl'
                new_val_attrs_embedding_pooler_path = self.attrs_embedding_path + 'new_val_attrs_embedding_pooler.pkl'
                new_test_attrs_embedding_pooler_path = self.attrs_embedding_path + 'new_test_attrs_embedding_pooler.pkl'

                train_attrs_embedding_cls_path = self.attrs_embedding_path + 'train_attrs_embedding_cls.pkl'
                new_val_attrs_embedding_cls_path = self.attrs_embedding_path + 'new_val_attrs_embedding_cls.pkl'
                new_test_attrs_embedding_cls_path = self.attrs_embedding_path + 'new_test_attrs_embedding_cls.pkl'

                train_attrs_embedding_mean_path = self.attrs_embedding_path + 'train_attrs_embedding_mean.pkl'
                new_val_attrs_embedding_mean_path = self.attrs_embedding_path + 'new_val_attrs_embedding_mean.pkl'
                new_test_attrs_embedding_mean_path = self.attrs_embedding_path + 'new_test_attrs_embedding_mean.pkl'

                train_attrs_embeddings_pooler, train_attrs_embeddings_cls, train_attrs_embeddings_mean = \
                    BERT_embed(list(train_attrs_dict.keys()))
                new_val_attrs_embeddings_pooler, new_val_attrs_embeddings_cls, new_val_attrs_embeddings_mean = \
                    BERT_embed(list(new_val_attrs_dict.keys()))
                new_test_attrs_embeddings_pooler, new_test_attrs_embeddings_cls, new_test_attrs_embeddings_mean = \
                    BERT_embed(list(new_test_attrs_dict.keys()))
                
                with open(train_attrs_embedding_pooler_path, 'wb') as fp:
                    pickle.dump(train_attrs_embeddings_pooler, fp)
                with open(new_val_attrs_embedding_pooler_path, 'wb') as fp:
                    pickle.dump(new_val_attrs_embeddings_pooler, fp)            
                with open(new_test_attrs_embedding_pooler_path, 'wb') as fp:
                    pickle.dump(new_test_attrs_embeddings_pooler, fp)
                
                with open(train_attrs_embedding_cls_path, 'wb') as fp:
                    pickle.dump(train_attrs_embeddings_cls, fp)
                with open(new_val_attrs_embedding_cls_path, 'wb') as fp:
                    pickle.dump(new_val_attrs_embeddings_cls, fp)            
                with open(new_test_attrs_embedding_cls_path, 'wb') as fp:
                    pickle.dump(new_test_attrs_embeddings_cls, fp)
                
                with open(train_attrs_embedding_mean_path, 'wb') as fp:
                    pickle.dump(train_attrs_embeddings_mean, fp)
                with open(new_val_attrs_embedding_mean_path, 'wb') as fp:
                    pickle.dump(new_val_attrs_embeddings_mean, fp)            
                with open(new_test_attrs_embedding_mean_path, 'wb') as fp:
                    pickle.dump(new_test_attrs_embeddings_mean, fp)

                with open(train_attrs_embedding_path, 'rb') as fp:
                    train_attrs_embedding = pickle.load(fp)
                with open(new_val_attrs_embedding_path, 'rb') as fp:
                    new_val_attrs_embedding = pickle.load(fp)
                with open(new_test_attrs_embedding_path, 'rb') as fp:
                    new_test_attrs_embedding = pickle.load(fp)

            elif plm == 'SBERT':
                train_attrs_embedding = SBERT_embed(list(train_attrs_dict.keys()))
                new_val_attrs_embedding = SBERT_embed(list(new_val_attrs_dict.keys()))
                new_test_attrs_embedding = SBERT_embed(list(new_test_attrs_dict.keys()))

                with open(train_attrs_embedding_path, 'wb') as fp:
                    pickle.dump(train_attrs_embedding, fp)
                with open(new_val_attrs_embedding_path, 'wb') as fp:
                    pickle.dump(new_val_attrs_embedding, fp)            
                with open(new_test_attrs_embedding_path, 'wb') as fp:
                    pickle.dump(new_test_attrs_embedding, fp)

        #================== items_embedding ===================
        if plm == 'BERT':
            train_items_embedding_path = self.items_embedding_path + 'train_items_embedding_' + embed_type + '.pkl'
            val_items_embedding_path = self.items_embedding_path + 'val_items_embedding_' + embed_type + '.pkl'
            test_items_embedding_path = self.items_embedding_path + 'test_items_embedding_' + embed_type + '.pkl'

        elif plm == 'SBERT':
            train_items_embedding_path = self.items_embedding_path + 'train_items_embedding_sbert.pkl'
            val_items_embedding_path = self.items_embedding_path + 'val_items_embedding_sbert.pkl'
            test_items_embedding_path = self.items_embedding_path + 'test_items_embedding_sbert.pkl'
        
        if os.path.exists(train_items_embedding_path):
            print('Loading train/val/test_items_embedding...')
            with open(train_items_embedding_path, 'rb') as fp:
                train_items_embedding = pickle.load(fp)
            with open(val_items_embedding_path, 'rb') as fp:
                val_items_embedding = pickle.load(fp)
            with open(test_items_embedding_path, 'rb') as fp:
                test_items_embedding = pickle.load(fp)
        else:
            print('Generating train/val/test_items_embedding...')
            if plm == 'BERT':
                train_items_embedding_pooler_path = self.items_embedding_path + 'train_items_embedding_pooler.pkl'
                val_items_embedding_pooler_path = self.items_embedding_path + 'val_items_embedding_pooler.pkl'
                test_items_embedding_pooler_path = self.items_embedding_path + 'test_items_embedding_pooler.pkl'

                train_items_embedding_cls_path = self.items_embedding_path + 'train_items_embedding_cls.pkl'
                val_items_embedding_cls_path = self.items_embedding_path + 'val_items_embedding_cls.pkl'
                test_items_embedding_cls_path = self.items_embedding_path + 'test_items_embedding_cls.pkl'

                train_items_embedding_mean_path = self.items_embedding_path + 'train_items_embedding_mean.pkl'
                val_items_embedding_mean_path = self.items_embedding_path + 'val_items_embedding_mean.pkl'
                test_items_embedding_mean_path = self.items_embedding_path + 'test_items_embedding_mean.pkl'

                train_items_embedding_pooler, train_items_embedding_cls, train_items_embedding_mean = \
                    BERT_embed([' '.join(each) for each in train_item_attrs_dict.values()])
                val_items_embedding_pooler, val_items_embedding_cls, val_items_embedding_mean = \
                    BERT_embed([' '.join(each) for each in val_item_attrs_dict.values()])
                test_items_embedding_pooler, test_items_embedding_cls, test_items_embedding_mean = \
                    BERT_embed([' '.join(each) for each in test_item_attrs_dict.values()])

                with open(train_items_embedding_pooler_path, 'wb') as fp:
                    pickle.dump(train_items_embedding_pooler, fp)
                with open(val_items_embedding_pooler_path, 'wb') as fp:
                    pickle.dump(val_items_embedding_pooler, fp)            
                with open(test_items_embedding_pooler_path, 'wb') as fp:
                    pickle.dump(test_items_embedding_pooler, fp)
                            
                with open(train_items_embedding_cls_path, 'wb') as fp:
                    pickle.dump(train_items_embedding_cls, fp)
                with open(val_items_embedding_cls_path, 'wb') as fp:
                    pickle.dump(val_items_embedding_cls, fp)            
                with open(test_items_embedding_cls_path, 'wb') as fp:
                    pickle.dump(test_items_embedding_cls, fp)
                            
                with open(train_items_embedding_mean_path, 'wb') as fp:
                    pickle.dump(train_items_embedding_mean, fp)
                with open(val_items_embedding_mean_path, 'wb') as fp:
                    pickle.dump(val_items_embedding_mean, fp)            
                with open(test_items_embedding_mean_path, 'wb') as fp:
                    pickle.dump(test_items_embedding_mean, fp)

                with open(train_items_embedding_path, 'rb') as fp:
                    train_items_embedding = pickle.load(fp)
                with open(val_items_embedding_path, 'rb') as fp:
                    val_items_embedding = pickle.load(fp)
                with open(test_items_embedding_path, 'rb') as fp:
                    test_items_embedding = pickle.load(fp)

            elif plm == 'SBERT':
                train_items_embedding = SBERT_embed([' '.join(each) for each in train_item_attrs_dict.values()])
                val_items_embedding = SBERT_embed([' '.join(each) for each in val_item_attrs_dict.values()])
                test_items_embedding = SBERT_embed([' '.join(each) for each in test_item_attrs_dict.values()])

                with open(train_items_embedding_path, 'wb') as fp:
                    pickle.dump(train_items_embedding, fp)
                with open(val_items_embedding_path, 'wb') as fp:
                    pickle.dump(val_items_embedding, fp)
                with open(test_items_embedding_path, 'wb') as fp:
                    pickle.dump(test_items_embedding, fp)

        #================== task1_list ===================
        task1_list = train_edges
        task1_list[1] += self.num_train_items
        task1_list = task1_list.t()
        idx = torch.randperm(task1_list.size()[0])
        task1_list = task1_list[idx]

        #================== statistics ===================
        print('Inside multi_task_dataloader task 1')
        print('\tNumber of attributes in the training set: ', len(train_attrs_dict)) # 
        print('\tNumber of new attributes in the validation set: ', len(new_val_attrs_dict)) # 
        print('\tNumber of new attributes in the test set: ', len(new_test_attrs_dict)) # 

        return train_item_attrs_dict, val_item_attrs_dict, test_item_attrs_dict, \
                train_attrs_dict, new_val_attrs_dict, new_test_attrs_dict, \
                train_attrs_embedding, new_val_attrs_embedding, new_test_attrs_embedding, \
                train_items_embedding, val_items_embedding, test_items_embedding, \
                train_graph, train_edges, val_graph, test_graph, task1_list
        

    def get_train_user_item_list(self, path):
        '''
        Generate a dict of user purchase histories (only items in the training set are kept).
        u_ratings format: {{mapped_user_id0: {mapped_item_id15: 4.0, mapped_item_id200: 5.0, ...}}, ...}
        '''
        u_ratings = {}

        # get the raw sequences (which are sorted by timestamp, 
        # so we can keep the sequential info in the generated sequences)
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                #print(line)
                line = line.strip().split('\t')
                [user, item, rating, timestamp] = line
                #print(user, item, rating)
                if user not in self.map_u_o2n.keys() or item not in self.map_i_o2n.keys():
                    continue
                user_id = self.map_u_o2n[user]
                item_id = self.map_i_o2n[item]
                if user_id in self.train_users:
                    if item_id in self.train_i_ratings.keys():
                        if user_id not in u_ratings.keys():
                            u_ratings[user_id] = {}
                            #u_ratings[user_id][item_id] = float(rating)
                        u_ratings[user_id][item_id] = float(rating)
        
        return u_ratings


    def task2(self, max_len = 100, keep_liked = False):
        '''
        Generate sequences (users' purchase histories) of items
        If keep_liked = True, only keep the items that a user likes (rating = 4.0 or 5.0)
        task2_list: a list of lists, format: [[item10, item18, item300, ...], ...]
        '''
        u_ratings = self.get_train_user_item_list(path = self.ratings_path)
        task2_list = []
        task2_pred_dic = {}
        for user, sequence in u_ratings.items():
            if keep_liked:
                #task2_list.append([item for item, rating in sequence.items() if rating >= 4.0])
                seq = [item for item, rating in sequence.items() if rating >= 4.0]
            else:
                #task2_list.append([item for item, rating in sequence.items()])
                seq = [item for item, rating in sequence.items()]
            if len(seq) > max_len:
                for i in range(len(seq) - max_len + 1):
                    task2_list.append(seq[i: i + max_len])
            else:
                task2_list.append(seq)
            
            pred_seq = [item for item, rating in sequence.items()]
            if len(pred_seq) > (max_len - 1):
                pred_seq = pred_seq[-(max_len - 1):]
            task2_pred_dic[user] = pred_seq

        task2_pred_dic = {k: v for k, v in sorted(task2_pred_dic.items(), key=lambda item: item[0])}
        print('Inside multi_task_dataloader task 2')
        print('\tTotal number of user purchase sequences (only kept the training items): ', len(task2_list)) #
        #print('task2_list: ', task2_list)
        return task2_list, task2_pred_dic


    def task3_phrase_senti(self, embedding_save_path, plm = 'BERT'):
        filtered_pl_attr_item_dict_path = embedding_save_path + 'pl_attr_item_dict.pkl'
        if os.path.exists(filtered_pl_attr_item_dict_path):
            with open(filtered_pl_attr_item_dict_path, 'rb') as fp:
                filtered_pl_attr_item_dict = pickle.load(fp)
        else:
            # get filtered (by frequencies) phrase-level sentiment-based review attributes
            # load preprocessed reviews with phrase-level sentiments.
            # reviews is a list of dictionaries, while each dictionary has the following keys:
            # 'user'; 'item'; 'rating'; 'text'; 'sentence', a list of (feature, adj, sent, score).
            reviews = pickle.load(open('./English-Jar/lei_100/output/reviews.pickle', 'rb'))

            item_attrs = {} # {0:[(bread, 1), (service, -1), ...], 1:[...], ...}
            for r in reviews:
                item = int(r['item'])
                if item not in item_attrs:
                    item_attrs[item] = []
                if 'sentence' in r:
                    for each in r['sentence']:
                        attr = (each[0], each[-1])
                        if attr not in item_attrs[item]:
                            item_attrs[item].append(attr)
            #print('item_attrs: ', item_attrs)

            unique_attrs = [] # [bread, service, ...]
            for k,v in item_attrs.items():
                unique_attrs += [attr[0] for attr in v]
            #print(len(unique_attrs)) # 137230
            unique_attrs = list(set(unique_attrs))
            #print(len(unique_attrs)) # 2133
            #print(unique_attrs[:50]) 

            # count attrs, w/o considering sentiment
            unique_attrs_counts = {} # {bread: 105, service: 809, ...}
            for k,v in item_attrs.items():
                for attr in v:
                    if attr[0] not in unique_attrs_counts:
                        unique_attrs_counts[attr[0]] = 0
                    unique_attrs_counts[attr[0]] += 1
            #print('unique_attrs_counts: ', unique_attrs_counts)

            # {bbq: 1, cake, 1, ..., service: 809, ...}
            unique_attrs_counts_sorted = {k: v for k, v in sorted(unique_attrs_counts.items(), key=lambda item: item[1])}
            #print('unique_attrs_counts_sorted: ', unique_attrs_counts_sorted)

            # {sugar: 10, ..., pepper: 500}
            unique_attrs_counts_sorted_filtered = {k: v for k, v in unique_attrs_counts_sorted.items() if v >= 10 and v <= 500}
            #print('unique_attrs_counts_sorted_filtered: ', unique_attrs_counts_sorted_filtered)
            #print('len(unique_attrs_counts_sorted_filtered): ', len(unique_attrs_counts_sorted_filtered))
            # [10, 500]: 754

            # {0:['good bread', 'bad service', ...], 1:[...], ...}
            item_attrs_filtered = {}
            for k, v in item_attrs.items():
                #item_attrs_filtered[k] = [attr for attr in v if attr[0] in unique_attrs_counts_sorted_filtered.keys()]
                item_attrs = []
                for attr in v:
                    if attr[0] in unique_attrs_counts_sorted_filtered.keys():
                        if attr[1] == 1:
                            item_attrs.append('good ' + attr[0])
                        elif attr[1] == -1:
                            item_attrs.append('bad ' + attr[0])
                        else:
                            print('unrecognized sentiment')
                            continue
                item_attrs_filtered[k] = list(set(item_attrs))
            #print('item_attrs_filtered: ', item_attrs_filtered)

            # keys: phrase-level sentiment-based review attributes 
            # values: lists of items that has each attribute
            # {'good bread':[0, 12, 66, ...], 'bad bread':[...], ...}
            filtered_pl_attr_item_dict = {}
            for k, v in item_attrs_filtered.items():
                for attr in v:
                    if attr not in filtered_pl_attr_item_dict:
                        filtered_pl_attr_item_dict[attr] = []
                    filtered_pl_attr_item_dict[attr].append(k)

            with open(filtered_pl_attr_item_dict_path, 'wb') as fp:
                pickle.dump(filtered_pl_attr_item_dict, fp)

        for k in list(filtered_pl_attr_item_dict.keys())[:10]:
            print(k, ': ', filtered_pl_attr_item_dict[k][:20]) # good pies :  [0, 118, 122, ...], good gelato :  [0, 29, 129, ...], ...

        if plm == 'BERT':
            filtered_pl_attr_embedding_path = embedding_save_path + 'pl_attr_embedding_BERT.pkl'
        elif plm == 'SBERT':
            filtered_pl_attr_embedding_path = embedding_save_path + 'pl_attr_embedding_SBERT.pkl'
        if os.path.exists(filtered_pl_attr_embedding_path):
            # load attr embeddings and attr-items dictionary
            with open(filtered_pl_attr_embedding_path, 'rb') as fp:
                filtered_pl_attr_embedding = pickle.load(fp)
        else:
            # save attr embeddings and attr-items dictionary
            if plm == 'BERT':
                filtered_pl_attr_embedding = BERT_embed(list(filtered_pl_attr_item_dict.keys()))[0]
            elif plm == 'SBERT':
                filtered_pl_attr_embedding = SBERT_embed(list(filtered_pl_attr_item_dict.keys()))
            with open(filtered_pl_attr_embedding_path, 'wb') as fp:
                pickle.dump(filtered_pl_attr_embedding, fp)
            
        if plm == 'BERT':
            print('filtered_pl_attr_embedding: ', filtered_pl_attr_embedding)
            print('filtered_pl_attr_embedding.shape: ', filtered_pl_attr_embedding.shape) # filtered_pl_attr_embedding[0].shape:  torch.Size([1447, 1024])
        elif plm == 'SBERT':
            print('filtered_pl_attr_embedding: ', filtered_pl_attr_embedding)
            print('filtered_pl_attr_embedding.shape: ', filtered_pl_attr_embedding.shape) # filtered_pl_attr_embedding.shape:  torch.Size([1447, 384])
            
        
        task3_graph_path = embedding_save_path + 'task3_graph_pl_attr.pkl'
        task3_train_edges_path = embedding_save_path + 'task3_train_edges_pl_attr.pkl'
        if os.path.exists(task3_graph_path):
            with open(task3_graph_path, 'rb') as fp:
                task3_graph = pickle.load(fp)
            with open(task3_train_edges_path, 'rb') as fp:
                train_edges = pickle.load(fp)
        else:
            edges = []
            for i, (attr,items) in enumerate(filtered_pl_attr_item_dict.items()):
                edges += [[item, i] for item in items]
            print('\nedges[:20]: ', edges[:20]) # [[0, 0], [118, 0], [122, 0], ...]
            print('\nnum_item_attr_edges: ', len(edges)) # nnum_item_attr_edges:  56332

            edges = torch.tensor(edges, dtype = torch.long).t()
            print('\nedges: ', edges) # tensor([[   0,  118,  122,  ..., 2164, 2235, 2255],
            #                                    [   0,    0,    0,  ..., 1444, 1445, 1446]])

            train_edges = edges.detach().clone()
            
            # generate the task3_graph
            edges[1] += self.num_train_items
            undirected_edges = pyg.utils.to_undirected(edges)
            task3_graph = Data(edge_index = undirected_edges)

            with open(task3_graph_path, 'wb') as fp:
                pickle.dump(task3_graph, fp)
            with open(task3_train_edges_path, 'wb') as fp:
                pickle.dump(train_edges, fp)

        task3_list = train_edges
        task3_list[1] += self.num_train_items
        print('\ntask3_list: ', task3_list) # tensor([[   0,  118,  122,  ..., 2164, 2235, 2255],
        #                                              [2258, 2258, 2258,  ..., 3702, 3703, 3704]])
        

        task3_list = task3_list.t()
        idx = torch.randperm(task3_list.size()[0])
        task3_list = task3_list[idx]
        #print('\ntask3_list: ', task3_list) # task3_list:  tensor([[1004, 2782],
        #                                                           [ 659, 2443],
        #                                                           [ 396, 2416], ...
        
        return filtered_pl_attr_embedding, task3_graph, task3_list


    def task3_phrase_senti_threshold_preprocess(self, threshold = 100):
        with open(self.reviews_path, 'r') as f:
            reviews_dict = json.load(f)

        item_r_count = {}
        writer_1 = open('./English-Jar/lei_' + str(threshold) + '/input/record.per.row.txt', 'w', encoding='utf-8')
        product2text_list = {}
        product2json = {}
        for item in self.train_i_ratings.keys():
            item_r_count[item] = 0

            raw_item_id = self.map_i_n2o[item]
            try:
                out_list = reviews_dict[raw_item_id]
            except KeyError:
                continue
            for review in out_list:
                if isinstance(review, list):
                    for r in review:
                        try:
                            if item_r_count[item] > threshold:
                                break
                            item_r_count[item] += 1

                            assert r['business_id'] == raw_item_id
                            
                            text = re.sub('\n', '', r['text'].strip())
                            writer_1.write('<DOC>\n{}\n</DOC>\n'.format(text))
                            item_id = str(item)
                            json_doc = {'user': r['user_id'],
                                'item': item_id,
                                'rating': int(r['stars']),
                                'text': text}
                            
                            if item_id in product2json:
                                product2json[item_id].append(json_doc)
                            else:
                                product2json[item_id] = [json_doc]
                            
                            if item_id in product2text_list:
                                product2text_list[item_id].append('<DOC>\n{}\n</DOC>\n'.format(text))
                            else:
                                product2text_list[item_id] = ['<DOC>\n{}\n</DOC>\n'.format(text)]
                        except KeyError:
                            continue
                else:
                    try:
                        if item_r_count[item] > threshold:
                            continue
                        item_r_count[item] += 1

                        assert review['business_id'] == raw_item_id
                        
                        text = re.sub('\n', '', review['text'].strip())
                        writer_1.write('<DOC>\n{}\n</DOC>\n'.format(text))
                        item_id = str(item)
                        json_doc = {'user': review['user_id'],
                            'item': item_id,
                            'rating': int(review['stars']),
                            'text': text}
                        
                        if item_id in product2json:
                            product2json[item_id].append(json_doc)
                        else:
                            product2json[item_id] = [json_doc]

                        if item_id in product2text_list:
                            product2text_list[item_id].append('<DOC>\n{}\n</DOC>\n'.format(text))
                        else:
                            product2text_list[item_id] = ['<DOC>\n{}\n</DOC>\n'.format(text)]
                    except KeyError:
                        continue
        
        #print('all_reviews[:10]: ', all_reviews[:10])
        #print('all_item_ids[:10]: ', all_item_ids[:10])

        with open('./English-Jar/lei_' + str(threshold) + '/input/records.per.product.txt', 'w', encoding='utf-8') as f:
            for (product, text_list) in product2text_list.items():
                f.write(product + '\t' + str(len(text_list)) + '\tfake_URL')
                text = '\n\t' + re.sub('\n', '\n\t', ''.join(text_list).strip()) + '\n'
                f.write(text)

        pickle.dump(product2json, open('./English-Jar/lei_' + str(threshold) + '/input/product2json.pickle', 'wb'))

        print('item_r_count: ', item_r_count)
        return

        
    def get_rec_data(self):
        '''
        Get the training, test, validation, and prediction data for the downstream rec task.
        Test set is constructed by taking the last 10 interactions of each training item (items in self.train_i_ratings).
        Training set contains the rest interactions in self.train_i_ratings.
        Validation set contains all the interactions in self.val_i_ratings.
        Prediction set contains all the interactions in self.test_i_ratings.
        Return date of format: [[user_id, item_id, rating], ...]
        '''
        train_triples, test_triples, val_triples, pred_triples = [], [], [], []
        for item, sequence in self.train_i_ratings.items():
            temp = [[user, item, rating] for user, rating in sequence.items()]
            train_triples += temp[:-10]
            test_triples += temp[-10:]

        for item, sequence in self.val_i_ratings.items():
            val_triples += [[user, item, rating] for user, rating in sequence.items()]

        for item, sequence in self.test_i_ratings.items():
            pred_triples += [[user, item, rating] for user, rating in sequence.items()]
        
        return np.array(train_triples), np.array(test_triples), np.array(val_triples), np.array(pred_triples)


if __name__ == "__main__":
    path = "./data/yelp/"
    dataloader = MultiTaskDataLoader(path = path, plm = 'BERT', t1 = True, t2 = True, t3 = True)
    
    
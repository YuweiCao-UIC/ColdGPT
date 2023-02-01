import logging
logging.basicConfig(level=logging.DEBUG)
import pandas as pd
from os.path import exists
import pickle
import math
from bs4 import BeautifulSoup 
import re
from textblob import TextBlob
from collections import Counter
from itertools import chain
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
import numpy as np
import random

def unisrec_embed(order_texts, temp_store_path):
    #print('len(order_texts): ', len(order_texts))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    embeddings = []
    batch_counter, start, batch_size = 0, 0, 4
    while start < len(order_texts):
        if batch_counter % 1000 == 0:
            print('processing batch: ', batch_counter)
        sentences = order_texts[start: start + batch_size]
        encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                      truncation=True, return_tensors='pt')#.to(device)
        outputs = model(**encoded_sentences)
        
        cls_output = outputs.last_hidden_state[:, 0, ].detach().cpu()
        #embeddings.append(cls_output)
        with open(temp_store_path + str(batch_counter), 'wb') as fp:
            pickle.dump(cls_output, fp)
        
        start += batch_size
        batch_counter += 1

    for batch_counter in range(math.ceil(len(order_texts)/batch_size)):
        with open(temp_store_path + str(batch_counter), 'rb') as f:
            batch_embedding = pickle.load(f)
        embeddings.append(batch_embedding)

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)
    return embeddings

class MultiTaskDataLoader():
    def __init__(self, store_path = "/home/anonymous/amazon_sports_data/preprocess_0126/", n = 500, \
        t1 = True, t2 = True, t3 = True, max_len = 50, t2_augment = None):

        self.n = n
        self.store_path = store_path

        # 90% as training. 5% of the training set are used for validation.
        # self.train_i_ratings, self.val_i_ratings, self.test_i_ratings format: 
        # {{mapped_item_id0: {mapped_user_id10: 3.0, mapped_user_id112: 4.0, ...}}, ...}
        # self.map_i_o2n format: {'0006564224': 0, '0560467893': 1, ...}
        # self.map_u_o2n format: {'A3NSN9WOX8470M': 0, 'A2AMX0AJ2BUDNV': 1, ...}
        self.train_i_ratings, self.val_i_ratings, self.test_i_ratings, \
            self.map_i_o2n, self.map_i_o2n_SCS, self.map_u_o2n, statistics = self.load_rating_splits(n = n)
        self.num_train_items = statistics['num_train_items']
        self.num_val_items = statistics['num_val_items']
        self.num_test_items = statistics['num_test_items']
        self.num_train_ratings = statistics['num_train_ratings']
        self.num_val_ratings = statistics['num_val_ratings']
        self.num_test_ratings = statistics['num_test_ratings']
        self.num_item = statistics['num_item'] 
        self.num_user = statistics['num_user']
        self.map_i_n2o = {v:k for k,v in self.map_i_o2n.items()}
        self.map_u_n2o = {v:k for k,v in self.map_u_o2n.items()}
        
        # for t1
        if t1:
            # construct the graphs, initial items and attributes' embeddings
            self.train_graph, self.train_edges, self.val_graph, self.test_graph, \
                self.train_attrs_dict, self.new_val_attrs_dict, self.new_test_attrs_dict, \
                self.train_attrs_embedding, self.new_val_attrs_embedding, self.new_test_attrs_embedding, \
                self.train_items_embedding, self.val_items_embedding, self.test_items_embedding, \
                self.train_item_attrs_dict, self.val_item_attrs_dict, self.test_item_attrs_dict, \
                self.task1_list = self.task1(store_path = store_path, n = n)
            
        
        # for t2
        if t2:
            # get self.u_ratings (only involve training items)
            # format: {{mapped_user_id0: {mapped_item_id10: 3.0, mapped_item_id112: 4.0, ...}}, ...}
            self.u_ratings = self.get_u_ratings(store_path = store_path)
            self.task2_list = self.task2(store_path = store_path, max_len = max_len, augment = t2_augment)

    def load_rating_splits(self, n):
        train_i_ratings_path = self.store_path + 'train_i_ratings.pkl'
        val_i_ratings_path = self.store_path + 'val_i_ratings.pkl'
        test_i_ratings_path = self.store_path + 'test_i_ratings_' +  str(n) + '.pkl'
        map_i_o2n_path = self.store_path + 'map_i_o2n.pkl'
        map_i_o2n_SCS_path = self.store_path + 'map_i_o2n_SCS_' + str(n) + '.pkl'
        map_u_o2n_path = self.store_path + 'map_u_o2n.pkl'
        statistics_path = self.store_path + 'statistics.pkl'

        if exists(train_i_ratings_path):
            print('Loading the train, val, test rating sequences, the item and user id maps, and the statistics...')
            with open(train_i_ratings_path, 'rb') as f:
                train_i_ratings = pickle.load(f)
            with open(val_i_ratings_path, 'rb') as f:
                val_i_ratings = pickle.load(f)
            with open(test_i_ratings_path, 'rb') as f:
                test_i_ratings = pickle.load(f)
            with open(map_i_o2n_path, 'rb') as f:
                map_i_o2n = pickle.load(f)
            with open(map_i_o2n_SCS_path, 'rb') as f:
                map_i_o2n_SCS = pickle.load(f)
            with open(map_u_o2n_path, 'rb') as f:
                map_u_o2n = pickle.load(f)
            with open(statistics_path, 'rb') as f:
                statistics = pickle.load(f)
            print('Loaded.')

            statistics['num_test_items'] = len(map_i_o2n_SCS)
            statistics['num_test_ratings'] = 0
            for item, sequence in test_i_ratings.items():
                statistics['num_test_ratings'] += len(sequence)
            statistics['num_item'] = len(map_i_o2n) + len(map_i_o2n_SCS)
            statistics['num_user'] = len(map_u_o2n)

            for k,v in statistics.items():
                print(k, ' ', v)
            return train_i_ratings, val_i_ratings, test_i_ratings, map_i_o2n, map_i_o2n_SCS, map_u_o2n, statistics
        else:
            print('Files don\'t exist.')
        return

    def task1(self, store_path, n):
        train_graph_path = store_path + 'train_graph.pkl'
        train_edges_path = store_path + 'train_edges.pkl'
        val_graph_path = store_path + 'val_graph.pkl'
        test_graph_path = store_path + 'test_graph_' + str(n) + '.pkl'

        train_attrs_dict_path = store_path + 'train_attrs_dict.pkl'
        new_val_attrs_dict_path = store_path + 'new_val_attrs_dict.pkl'
        new_test_attrs_dict_path = store_path + 'new_test_attrs_dict_' + str(n) + '.pkl'

        train_attrs_embedding_path = store_path + 'train_attrs_embedding.pkl'
        new_val_attrs_embedding_path = store_path + 'new_val_attrs_embedding.pkl'
        new_test_attrs_embedding_path = store_path + 'new_test_attrs_embedding_' + str(n) + '.pkl'

        train_items_embedding_path = store_path + 'train_items_embedding.pkl'
        val_items_embedding_path = store_path + 'val_items_embedding.pkl'
        test_items_embedding_path = store_path + 'test_items_embedding_' + str(n) + '.pkl'

        train_item_attrs_dict_path = store_path + 'train_item_attrs_dict.pkl'
        val_item_attrs_dict_path = store_path + 'val_item_attrs_dict.pkl'
        test_item_attrs_dict_path =  store_path + 'test_item_attrs_dict_' + str(n) + '.pkl'
        
        if exists(train_graph_path):
            print('Loading the graphs and the train edges...')
            with open(train_graph_path, 'rb') as f:
                train_graph = pickle.load(f)
            with open(train_edges_path, 'rb') as f:
                train_edges = pickle.load(f)
            with open(val_graph_path, 'rb') as f:
                val_graph = pickle.load(f)
            with open(test_graph_path, 'rb') as f:
                test_graph = pickle.load(f)
            print('Loaded the graphs and the train edges.')

            print('Loading the attribute dicts...')
            with open(train_attrs_dict_path, 'rb') as f:
                train_attrs_dict = pickle.load(f)
            with open(new_val_attrs_dict_path, 'rb') as f:
                new_val_attrs_dict = pickle.load(f)
            with open(new_test_attrs_dict_path, 'rb') as f:
                new_test_attrs_dict = pickle.load(f)
            print('Loaded the attribute dicts.')

            print('Loading the attribute embeddings...')
            with open(train_attrs_embedding_path, 'rb') as f:
                train_attrs_embedding = pickle.load(f)
            with open(new_val_attrs_embedding_path, 'rb') as f:
                new_val_attrs_embedding = pickle.load(f)
            with open(new_test_attrs_embedding_path, 'rb') as f:
                new_test_attrs_embedding = pickle.load(f)
            print('Loaded the attribute embeddings.')

            print('Loading the item embeddings...')
            with open(train_items_embedding_path, 'rb') as f:
                train_items_embedding = pickle.load(f)
            with open(val_items_embedding_path, 'rb') as f:
                val_items_embedding = pickle.load(f)
            with open(test_items_embedding_path, 'rb') as f:
                test_items_embedding = pickle.load(f)
            print('Loaded the item embeddings.')

            with open(train_item_attrs_dict_path, 'rb') as f:
                train_item_attrs_dict = pickle.load(f)
            with open(val_item_attrs_dict_path, 'rb') as f:
                val_item_attrs_dict = pickle.load(f)
            with open(test_item_attrs_dict_path, 'rb') as f:
                test_item_attrs_dict = pickle.load(f)
            print('Loaded the item attribute dicts')

            task1_list = train_edges
            task1_list[1] += self.num_train_items
            task1_list = task1_list.t()
            idx = torch.randperm(task1_list.size()[0])
            task1_list = task1_list[idx]

            print('num_train_attrs_dict: ', len(train_attrs_dict))
            print('num_new_val_attrs_dict: ', len(new_val_attrs_dict))
            print('num_new_test_attrs_dict: ', len(new_test_attrs_dict))

            return train_graph, train_edges, val_graph, test_graph, \
                train_attrs_dict, new_val_attrs_dict, new_test_attrs_dict, \
                train_attrs_embedding, new_val_attrs_embedding, new_test_attrs_embedding, \
                train_items_embedding, val_items_embedding, test_items_embedding, \
                train_item_attrs_dict, val_item_attrs_dict, test_item_attrs_dict, task1_list
        else:
            print('t1 files don\'t exist.')
    
    def get_u_ratings(self, store_path):
        u_ratings_path = store_path + 'u_ratings.pkl'

        if exists(u_ratings_path):
            print('Loading the u_ratings...')
            with open(u_ratings_path, 'rb') as f:
                u_ratings = pickle.load(f)
            print('Loaded.')
            return u_ratings
        else:
            print('File doesn\'t exist.')

    def augment_one_sequence(self, sequence, df_also_view_minus_also_buy, prob = 0.5):
        prob_indices = [random.random() for i in range(len(sequence))]
        augmented_sequence = [random.choice(df_also_view_minus_also_buy[item]) \
            if item in df_also_view_minus_also_buy.keys() and prob_indices[i] < prob \
            else item for i, item in enumerate(sequence)]
        #print(augmented_sequence)
        return augmented_sequence

    def task2(self, store_path, max_len = 100, augment = 'also_view_minus_also_buy'):
        print('Inside multi_task_dataloader task 2')

        task2_list = []
        #print('len(self.u_ratings): ', len(self.u_ratings)) # 96420

        for user, sequence in self.u_ratings.items():
            if len(sequence) >= 2:
                task2_list.append([item for item, rating in sequence.items()])

        #print('len(task2_list) before augmentation: ', len(task2_list)) # 95967

        if augment:
            if augment == 'corrupt':
                # corrupt the sequences for test purpose
                task2_list_corrupted = [random.sample(list(self.train_i_ratings.keys()), len(s)) for s in task2_list]

                print('\tTotal number of corrupted user purchase sequences: ', len(task2_list_corrupted)) # 95967
                return task2_list_corrupted

            df_also_buy_view_path = store_path + 'df_also_buy_view.pkl'
            if exists(df_also_buy_view_path):
                print('Loading df_also_buy_view...')
                with open(df_also_buy_view_path, 'rb') as f:
                    df_also_buy_view = pickle.load(f)
                print('Loaded.')
            else:
                print('File doesn\'t exist.')

            if augment == 'also_view_minus_also_buy':
                #replaceable_items = df_also_buy_view[df_also_buy_view.also_view_minus_also_buy.astype(bool)]['item_mapped'].tolist()
              
                df_also_view_minus_also_buy = df_also_buy_view[df_also_buy_view.also_view_minus_also_buy.astype(bool)]
                df_also_view_minus_also_buy = dict(zip(df_also_view_minus_also_buy.item_mapped, df_also_view_minus_also_buy.also_view_minus_also_buy))
                
                augment_lists = []
                for _ in range(2):
                    augment_lists += [self.augment_one_sequence(s, df_also_view_minus_also_buy) for s in task2_list]
                task2_list += augment_lists

                task2_list_tpls = [tuple(s) for s in task2_list]
                task2_list_dct = list(dict.fromkeys(task2_list_tpls))
                task2_list = [list(s) for s in task2_list_dct]

                print('len(task2_list) after augmentation: ', len(task2_list)) # 172582

        task2_list_mapped = []
        for seq in task2_list:
            if len(seq) > max_len:
                for i in range(len(seq) - max_len + 1):
                    task2_list_mapped.append(seq[i: i + max_len])
            else:
                task2_list_mapped.append(seq)

        print('\tTotal number of user purchase sequences: ', len(task2_list_mapped))
        return task2_list_mapped

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
    
    def get_IDCF_item_attr_embeddings(self):
        '''
        Get the embeddings (correspond to d_u' in Equation 4) of all the items as needed by the IDCF-HY model,
        where the query items don't have any historical interactions. 
        The embedding of each item is calculated as an average of all its attrbutes' BERT embeddings.
        '''

        '''
        assert len(self.train_attrs_dict) == len(self.train_attrs_embedding)
        assert len(self.new_val_attrs_dict) == len(self.new_val_attrs_embedding)
        assert len(self.new_test_attrs_dict) == len(self.new_test_attrs_embedding)
        '''
        train_attr_embedding_dict = dict(zip(list(self.train_attrs_dict.keys()), self.train_attrs_embedding))
        val_attr_embedding_dict = dict(zip(list(self.new_val_attrs_dict.keys()), self.new_val_attrs_embedding))
        test_attr_embedding_dict = dict(zip(list(self.new_test_attrs_dict.keys()), self.new_test_attrs_embedding))
        attr_embedding_dict = {**train_attr_embedding_dict, **val_attr_embedding_dict, **test_attr_embedding_dict}
        #print(attr_embedding_dict)
        item_embeddings = []
        for item, attrs in self.train_item_attrs_dict.items():
            item_embeddings.append(torch.mean(torch.stack([attr_embedding_dict[attr] for attr in attrs]), 0))
        for item, attrs in self.val_item_attrs_dict.items():
            item_embeddings.append(torch.mean(torch.stack([attr_embedding_dict[attr] for attr in attrs]), 0))
        for item, attrs in self.test_item_attrs_dict.items():
            item_embeddings.append(torch.mean(torch.stack([attr_embedding_dict[attr] for attr in attrs]), 0))
        #print(item_embeddings)
        item_embeddings = torch.stack(item_embeddings)
        #print(item_embeddings)
        #print(item_embeddings.size())
        return item_embeddings

    def get_IDCF_data_filtered(self, item_attr_embeddings = False):
        '''
        Get data needed for running the IDCF method.
        '''
        # supp: support (key) items, vali: validation query items, que: query items
        train_set_supp, test_set_supp, test_set_vali_que, test_set_que = [], [], [], []
        for item, sequence in self.train_i_ratings.items():
            temp = [[item, user, rating] for user, rating in sequence.items()]
            train_set_supp += temp[:-10]
            test_set_supp += temp[-10:]

        for item, sequence in self.val_i_ratings.items():
            test_set_vali_que += [[item, user, rating] for user, rating in sequence.items()]

        for item, sequence in self.test_i_ratings.items():
            test_set_que += [[item, user, rating] for user, rating in sequence.items()]

        # a list of all the support items
        item_supp_list = list(self.train_i_ratings.keys())

        # a dictionary that stores the users (involved in the support training set) that ever bought each item
        item_his_dic = {} 
        for each in train_set_supp:
            item_id = each[0]
            if item_id not in item_his_dic:
                item_his_dic[item_id] = []
            item_his_dic[item_id].append(each[1])
        
        n_rating = 5 # yelp users rate restaurants as 1/2/3/4/5 stars
        edge_IU = np.zeros((self.num_item, self.num_user), dtype=np.int)
        for i in range(1, n_rating+1):
            edge_i = [each[:2] for each in train_set_supp if each[2] == i]
            edge_i = np.array(edge_i, dtype=np.int32)
            edge_IU[edge_i[:,0], edge_i[:,1]] = i
        
        if item_attr_embeddings:
            item_embeddings = self.get_IDCF_item_attr_embeddings()
            return train_set_supp, test_set_supp, test_set_vali_que, test_set_que, item_his_dic, item_supp_list, edge_IU, item_embeddings

        return train_set_supp, test_set_supp, test_set_vali_que, test_set_que, item_his_dic, item_supp_list, edge_IU

    def get_unisrec_data(self):
        store_path = '/home/anonymous/UniSRec/dataset/downstream/amazon_sports_' + str(self.n) + '/'
        _, _, _, data = self.get_rec_data() #user, item, rating
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

        with open(store_path + 'dic_user', 'wb') as fp:
            pickle.dump(dic_user, fp)


        train_user_id, train_item_id_list, train_item_id = [], [], []
        valid_user_id, valid_item_id_list, valid_item_id = [], [], []
        test_user_id, test_item_id_list, test_item_id = [], [], []
        
        for user_id, sequence in self.u_ratings.items():
            sequence = list(sequence.keys())

            test_user_id.append(user_id)
            s = sequence
            if len(s) > 50:
                s = s[-50:]
            test_item_id_list.append(s)
            test_item_id.append(0)
            
            s = sequence[:-1]
            if len(s) > 0:
                valid_user_id.append(user_id)
                if len(s) > 50:
                    s = s[-50:]
                valid_item_id_list.append(s)
                valid_item_id.append(sequence[-1])
            
            for i in range(len(sequence) - 2):
                train_user_id.append(user_id)
                s = sequence[:(i+1)]
                if len(s) > 50:
                    s = s[-50:]
                train_item_id_list.append(s)
                train_item_id.append(sequence[i+1])
        
        valid_item_id_list = [' '.join(str(x) for x in each) for each in valid_item_id_list]
        valid_df = pd.DataFrame(list(zip(valid_user_id, valid_item_id_list, valid_item_id)),
               columns =['user_id:token', 'item_id_list:token_seq', 'item_id:token'])
        #print(valid_df.head(10))
        valid_df.to_csv(store_path + 'amazon_sports.valid.inter', sep='\t', index=False)
        
        train_item_id_list = [' '.join(str(x) for x in each) for each in train_item_id_list]
        train_df = pd.DataFrame(list(zip(train_user_id, train_item_id_list, train_item_id)),
               columns =['user_id:token', 'item_id_list:token_seq', 'item_id:token'])
        #print(train_df.head(10))
        train_df.to_csv(store_path + 'amazon_sports.train.inter', sep='\t', index=False)

        test_item_id_list = [' '.join(str(x) for x in each) for each in test_item_id_list]
        test_df = pd.DataFrame(list(zip(test_user_id, test_item_id_list, test_item_id)),
               columns =['user_id:token', 'item_id_list:token_seq', 'item_id:token'])
        test_df.to_csv(store_path + 'amazon_sports.test.inter', sep='\t', index=False)
        

        item2index_df = pd.DataFrame(list(zip([i for i in range(self.num_train_items + self.num_test_items)], \
            [i for i in range(self.num_train_items + self.num_test_items)])))
        item2index_df.to_csv(store_path + 'amazon_sports.item2index', sep='\t', index=False, header=False)
        
        user2index_df = pd.DataFrame(list(zip(list(self.u_ratings.keys()), \
            list(self.u_ratings.keys()))))
        user2index_df.to_csv(store_path + 'amazon_sports.user2index', sep='\t', index=False, header=False)
        
        #print(self.test_item_attrs_dict)
        item_ids, texts = [], []
        test_item_attrs_dict_mapped = {(k-self.num_val_items):v for k,v in self.test_item_attrs_dict.items()}
        #print(test_item_attrs_dict_mapped)
        for i in range(self.num_train_items + self.num_test_items):
            item_ids.append(i)
            texts.append({**self.train_item_attrs_dict, **test_item_attrs_dict_mapped}[i])
        texts = [' '.join(str(x) for x in each) for each in texts]
        #text_df = pd.DataFrame(list(zip(item_ids, texts)),
        #    columns =['item_id:token', 'text:token_seq'])
        #text_df.to_csv(store_path + 'yelp.text', sep='\t', index=False)
        print('len(texts): ', len(texts))
        embeddings = unisrec_embed(texts, store_path + 'temp/')
        with open(store_path + 'amazon_sports.feat1CLS', 'wb') as fp:
            pickle.dump(embeddings, fp)
        return

if __name__ == "__main__":
    dataloader = MultiTaskDataLoader(n = 520, t1 = True, t2 = True, t3 = False, t2_augment = None)
    #dataloader.get_rec_data()
    #dataloader.get_IDCF_data_filtered()
    dataloader.get_unisrec_data()
    '''
    import random
    import math
    prob = 0.5
    sequence = [10, 12, 3, 6, 9, 7, 8]
    d = {1:[3], 2: [6, 7, 8, 9], 3: [1, 8], 6: [2, 8], 7: [2], 8: [2], 9: [2, 7, 8]}
    df_also_view_minus_also_buy = pd.DataFrame(d.items(), columns=['item_mapped', 'also_view_minus_also_buy'])
    df_also_view_minus_also_buy = dict(zip(df_also_view_minus_also_buy.item_mapped, df_also_view_minus_also_buy.also_view_minus_also_buy))
    print(df_also_view_minus_also_buy)
   
    prob_indices = [random.random() for i in range(len(sequence))]
    print(prob_indices)
    augmented_sequence = [random.choice(df_also_view_minus_also_buy[item]) \
        if item in df_also_view_minus_also_buy.keys() and prob_indices[i] < prob \
        else item for i, item in enumerate(sequence)]
    print(augmented_sequence)
    
    sequences = [[1, 2], [1, 2], [1, 2, 3], [10, 3, 22, 9, 10, 15]]
    sequences_tpls = [tuple(s) for s in sequences]
    print('sequences_tpls: ', sequences_tpls)
    sequences_dct = list(dict.fromkeys(sequences_tpls))
    sequences = [list(s) for s in sequences_dct]
    print('sequences: ', sequences)
    '''

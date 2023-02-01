import logging
logging.basicConfig(level=logging.DEBUG)
import pandas as pd
from os.path import exists
import pickle
import math
#from amazon_movielen_data_process import getselectDF
from bs4 import BeautifulSoup 
import re
from textblob import TextBlob
from collections import Counter
from itertools import chain
from sentence_transformers import SentenceTransformer
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
import gzip
import json

def getselectDF_SCS(path, columns_list):
  i = 0
  df = {}
  for d in parse_SCS(path):
    #print(d)
    temp_d = {}
    for c in columns_list:
      try:
        temp_d[c] = d[c]
      except KeyError:
        continue
    df[i] = temp_d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def parse_SCS(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getselectDF(path, columns_list):
  i = 0
  df = {}
  for d in parse(path):
    #print(d)
    temp_d = {}
    for c in columns_list:
      try:
        temp_d[c] = d[c]
      except KeyError:
        continue
    df[i] = temp_d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        l = l.decode("utf-8") .replace('true', 'True')
        l = l.replace('false', 'False')
        yield eval(l)

def SBERT_embed(s_list):
    '''
    Use Sentence-BERT to embed sentences.
    s_list: a list of sentences/ tokens to be embedded.
    output: the embeddings of the sentences/ tokens.
    '''
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(s_list, convert_to_tensor = True, normalize_embeddings = True)
    return embeddings

def clean_sentence(s):
    '''
    Take s, which is a sentence (string), clean it, and return.
    '''
    # remove newlines and tabs
    s = s.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ')
    # strip HTML tags
    s = BeautifulSoup(s, "html.parser")
    s = s.get_text(separator=" ")
    # replace special chars (*, :, and ^) with space
    s = re.sub(r"[*:^]+", ' ', s)
    return s

def extract_attributes(s_list):
    '''
    Take a list of sentences (strings) and return a list of noun phrases contained in these sentences.
    '''
    #print()
    #print(s_list)
    result = []
    for s in s_list:
        #print(s)
        s = clean_sentence(s)
        blob = TextBlob(s)
        #print(blob.noun_phrases)
        result += blob.noun_phrases
    return result

def extract_attributes_sentence(s):
    '''
    Take a sentence (string) and return a list of noun phrases contained in this sentence.
    '''
    #print(s)
    result = []
    if type(s) == str:
        s = clean_sentence(s)
        blob = TextBlob(s)
        #print(blob.noun_phrases)
        result += blob.noun_phrases
        #print(result)
        #print()
    return result

class MultiTaskDataPreprocessor():
    def __init__(self, filter_threshold = 5, \
            store_path = "/home/anonymous/amazon_sports_data/preprocess_0126/", n = 500):

        self.filter_threshold = filter_threshold
        
        self.new_meta_path = '/home/anonymous/amazon_sports_data/meta_Sports_and_Outdoors.json.gz' # new dataset metadata
        self.meta_path = '/home/anonymous/github/P5/raw_data/meta_Sports_and_Outdoors.json.gz' # old dataset metadata
        self.review_path = '/data1/anonymousanonymous/datasets/amazon/data/amazon_old_data/amazon_review/reviews_Sports_and_Outdoors_5.json.gz'
        self.ratings_path = '/data1/anonymousanonymous/datasets/amazon/data/amazon_old_data/amazon_ratings/ratings_Sports_and_Outdoors.csv'
        
        # read in the ratings, filter them accoring to the sequence length, then
        # sort them according to sequence length
        # self.i_ratings format: {{raw_item_id0: {raw_user_id10: 3.0, raw_user_id112: 4.0, ...}}, ...}
        # e.g: {{'0006564224': {'A3NSN9WOX8470M': 3.0, 'A2AMX0AJ2BUDNV': 4.0, ...}}, ...}
        self.i_ratings = self.read_ratings(load_path = self.ratings_path, n = self.filter_threshold, store_path = store_path)
        
        # get meta data of the items
        # extract and filter item attributes for t1 (too frequent/infrequent attributes are removed)
        # filter items based on their attributes (items that don't have enough attributes are removed)
        # self.df_meta_w_filtered_attr contains: 
        #   1) values from the raw meta data: item, category, brand, description, feature 
        #   2) extracted_attr: a list of attributes extracted from 'description' and 'feature'
        #   3) merged_attr: a list that combines category, brand, and extracted_attr, with redundancies removed.
        #   4) filtered_attr: a list of attributes left in merged_attr after filtering out too frequent/infrequent attributes.
        self.df_meta_w_filtered_attr = self.read_attributes(store_path = store_path)
        
        # filter self.i_ratings, only keep items that have enough attributes.
        # self.i_ratings_filtered is sorted based on the sequence length (longer sequences come first)
        self.i_ratings_filtered = self.filter_ratings(store_path = store_path)

        # split self.i_ratings (already sorted by length in self.read_ratings) into train/val sets 
        # 100% as training, 5% of the training set are used for validation
        # self.train_i_ratings, self.val_i_ratings format: 
        # {{mapped_item_id0: {mapped_user_id10: 3.0, mapped_user_id112: 4.0, ...}}, ...}
        # self.map_i_o2n format: {'0006564224': 0, '0560467893': 1, ...}
        # self.map_u_o2n format: {'A3NSN9WOX8470M': 0, 'A2AMX0AJ2BUDNV': 1, ...}
        self.train_i_ratings, self.val_i_ratings, \
            self.map_i_o2n, self.map_u_o2n, statistics = self.split_ratings(store_path = store_path)
        self.num_train_items = statistics['num_train_items']
        self.num_val_items = statistics['num_val_items']
        self.num_train_ratings = statistics['num_train_ratings']
        self.num_val_ratings = statistics['num_val_ratings']
        self.map_i_n2o = {v:k for k,v in self.map_i_o2n.items()}
        self.map_u_n2o = {v:k for k,v in self.map_u_o2n.items()}
        self.num_item = len(self.map_i_o2n) # 
        self.num_user = len(self.map_u_o2n) # 

        #self.filter_test_items()
        #self.check_test_item_attrs()
        self.df_meta_w_filtered_attr_SCS = self.read_attributes_SCS(store_path, n = n)
        self.map_i_o2n_SCS, self.test_i_ratings = self.get_ratings_SCS(store_path, n = n)
        self.num_test_items = len(self.map_i_o2n_SCS)
        self.map_i_n2o_SCS = {v:k for k,v in self.map_i_o2n_SCS.items()}

        # get self.u_ratings (only involve training items)
        # format: {{mapped_user_id0: {mapped_item_id10: 3.0, mapped_item_id112: 4.0, ...}}, ...}
        self.u_ratings = self.get_u_ratings(store_path = store_path)
        
        # for t1
        # construct the graphs, initial items and attributes' embeddings
        self.train_graph, self.train_edges, self.val_graph, self.test_graph, \
            self.train_attrs_dict, self.new_val_attrs_dict, self.new_test_attrs_dict, \
            self.train_attrs_embedding, self.new_val_attrs_embedding, self.new_test_attrs_embedding, \
            self.train_items_embedding, self.val_items_embedding, self.test_items_embedding, \
            self.train_item_attrs_dict, self.val_item_attrs_dict, self.test_item_attrs_dict = self.task1(store_path = store_path, n = n)
        
        edge_index1, edge_index2 = self.train_graph.edge_index
        edge = torch.stack([edge_index1, edge_index2], dim=1).tolist()
        self.task1_list = []
        for e in edge:
            self.task1_list.append((e[0], e[1]))
        
        '''
        # for t2
        #self.task2(store_path = store_path)
        self.df_also_buy_view = self.get_also_buy_view(store_path = store_path)
        '''

    def get_ratings_SCS(self, store_path, n):
        filtered_hasattr_interaction_path = '/home/anonymous/amazon_sports_data/SCS_interactions_' + str(n) + '_filtered_hasattr.pkl'
        test_i_ratings_path = store_path + 'test_i_ratings_' +  str(n) + '.pkl'
        map_i_o2n_SCS_path = store_path + 'map_i_o2n_SCS_' + str(n) + '.pkl'

        if exists(filtered_hasattr_interaction_path):
            with open(filtered_hasattr_interaction_path, 'rb') as f:
                filtered_hasattr_interaction = pickle.load(f)
        else:
            filtered_interaction_path = '/home/anonymous/amazon_sports_data/SCS_interactions_' + str(n) + '_filtered.pkl'
            with open(filtered_interaction_path, 'rb') as f:
                filtered_interaction = pickle.load(f)
            filtered_hasattr_interaction = [each for each in filtered_interaction if each['item'] in self.df_meta_w_filtered_attr_SCS.item.unique()]
            with open(filtered_hasattr_interaction_path, 'wb') as f:
                pickle.dump(filtered_hasattr_interaction, f)
        print('len(filtered_hasattr_interaction): ', len(filtered_hasattr_interaction))

        if exists(test_i_ratings_path):
            with open(test_i_ratings_path, 'rb') as f:
                test_i_ratings = pickle.load(f)
            with open(map_i_o2n_SCS_path, 'rb') as f:
                map_i_o2n_SCS = pickle.load(f)
        else:
            map_i_o2n_SCS = {item:(len(self.map_i_o2n) + i) for i, item in enumerate(list(self.df_meta_w_filtered_attr_SCS.item.unique()))}
            # i_ratings format: {{mapped_item_id0: {mapped_user_id10: 3.0, mapped_user_id112: 4.0, ...}}, ...}
            test_i_ratings = {}
            for each in filtered_hasattr_interaction:
                item_id = map_i_o2n_SCS[each['item']]
                user_id = self.map_u_o2n[each['user']]
                if item_id not in test_i_ratings:
                    test_i_ratings[item_id] = {user_id: 5.0}
                else:
                    test_i_ratings[item_id][user_id] = 5.0

            with open(test_i_ratings_path, 'wb') as f:
                pickle.dump(test_i_ratings, f)
            with open(map_i_o2n_SCS_path, 'wb') as f:
                pickle.dump(map_i_o2n_SCS, f)

        print('len(map_i_o2n_SCS): ', len(map_i_o2n_SCS))
        return map_i_o2n_SCS, test_i_ratings

    def read_ratings(self, load_path, n = 15, store_path = None):
        print('Inside read_ratings')
        i_ratings_path = store_path + 'i_ratings_n' + str(n) + '.pkl'

        if exists(i_ratings_path):
            print('Loading the filtered data...')
            with open(i_ratings_path, 'rb') as f:
                i_ratings = pickle.load(f)
            print('Loaded.')
            #print('list(i_ratings.keys())[:10]: ', list(i_ratings.keys())[:10])
        else:
            print('Constructing the filtered data...')
            # read in the raw ratings
            #raw_rating_df = pd.read_csv(load_path, sep=",", header=None, names=["item", "user", "rating", "time"])
            raw_rating_df = pd.read_csv(load_path, sep=",", header=None, names=["user", "item", "rating", "time"])
            #print('len(raw_rating_df): ', len(raw_rating_df))
            # sort the records by timestamp
            raw_rating_df = raw_rating_df.sort_values('time')

            # get filtered item rating sequences
            # i_ratings format: {{mapped_item_id0: {mapped_user_id10: 3.0, mapped_user_id112: 4.0, ...}}, ...}
            i_ratings = {}

            # get the raw sequences
            # all_u_ratings format: {{raw_user_id0: {raw_item_id8: 2.0, raw_item_id666: 5.0, ...}}, ...}
            all_u_ratings = {}
            for user, item, rating in zip(raw_rating_df['user'], raw_rating_df['item'], raw_rating_df['rating']):
                if user not in all_u_ratings.keys():
                    all_u_ratings[user] = {}
                all_u_ratings[user][item] = float(rating)

            # filter data based on sequence length
            # filter out users with too short sequences
            u_ratings = {}
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

            # save the filtered data
            with open(i_ratings_path, 'wb') as f:
                pickle.dump(i_ratings, f)
            print('Saved.')

        print('num_items after filtering out short interaction sequences: ', len(i_ratings))
        return i_ratings


    def get_item_meta_data(self, store_path):
        df_meta_path = store_path + 'df_meta_n' + str(self.filter_threshold) + '.pkl'

        if exists(df_meta_path):
            print('Loading item meta data...')
            with open(df_meta_path, 'rb') as f:
                df_meta = pickle.load(f)
            print('Loaded')
            #print('df_meta.head(10): ', df_meta.head(10))
            #for each in df_meta.head(10)['categories'].tolist():
            #   print(each[0])
    
        else:
            print('Extracting item meta data...')
            #columns_list = ['asin', 'category', 'description', 'feature', 'title', 'brand']
            columns_list = ['asin', 'categories', 'description', 'title', 'brand']
            df_meta = getselectDF(self.meta_path, columns_list)
            df_meta = df_meta.rename(columns={'asin':'item'})

            # select the rows for the filtered items
            df_meta = df_meta.loc[df_meta['item'].isin(self.i_ratings.keys())]
            print('len(df_meta) before drop_duplicates: ', len(df_meta)) # len(df_meta) before drop_duplicates:  28809
            df_meta = df_meta.drop_duplicates(subset=['item'])
            print('len(df_meta) after drop_duplicates: ', len(df_meta)) # len(df_meta) after drop_duplicates:  28809

            # save the filtered meta data
            with open(df_meta_path, 'wb') as f:
                pickle.dump(df_meta, f)
            print('Extracted and saved.')
        
        #print(df_meta.head(10))
        return df_meta


    def get_item_meta_data_w_extracted_attributes(self, df_meta, store_path):
        df_meta_w_extracted_attr_path = store_path + 'df_meta_n' + str(self.filter_threshold) + '_extracted_attr.pkl'

        if exists(df_meta_w_extracted_attr_path):
            print('Loading item meta data (with raw attributes) ...')
            with open(df_meta_w_extracted_attr_path, 'rb') as f:
                df_meta_w_extracted_attr = pickle.load(f)
            print('Loaded.')
        else:
            print('Extracting raw attributes from item meta data...')
            # extract attributes from description and feature
            #extracted_attr = [extract_attributes(description_list + feature_list) for description_list, feature_list in zip(df_meta['description'], df_meta['feature'])]
            #print('df_meta[\'description\'].head(10).tolist(): ', df_meta['description'].head(10).tolist())
            extracted_attr = [extract_attributes_sentence(description) for description in df_meta['description']]
            # concatenate to df_meta
            df_meta['extracted_attr'] = extracted_attr
            df_meta_w_extracted_attr = df_meta
            # save the concatenated meta data
            with open(df_meta_w_extracted_attr_path, 'wb') as f:
                pickle.dump(df_meta_w_extracted_attr, f)
            print('Extracted, concatenated to df_meta, and saved.')
            
        return df_meta_w_extracted_attr


    def get_item_meta_data_w_merged_attributes(self, df_meta, store_path):
        df_meta_w_merged_attr_path = store_path + 'df_meta_n' + str(self.filter_threshold) + '_merged_attr.pkl'

        if exists(df_meta_w_merged_attr_path):
            print('Loading item meta data (with merged attributes) ...')
            with open(df_meta_w_merged_attr_path, 'rb') as f:
                df_meta_w_merged_attr = pickle.load(f)
            print('Loaded.')
        else:
            print('Merging attributes for items...')
            # merge the attributes in 'category', 'brand', and 'extracted_attr' (extracted from 'description' and 'feature')
            #merged_attr = [list(set(category + [brand] + extracted_attr)) for category, brand, extracted_attr in zip(df_meta['category'], df_meta['brand'], df_meta['extracted_attr'])]
            merged_attr = [list(set(category[0] + [brand] + extracted_attr)) for category, brand, extracted_attr in zip(df_meta['categories'], df_meta['brand'], df_meta['extracted_attr'])]
            # concatenate to df_meta
            df_meta['merged_attr'] = merged_attr
            df_meta_w_merged_attr = df_meta
            # save the concatenated meta data
            with open(df_meta_w_merged_attr_path, 'wb') as f:
                pickle.dump(df_meta_w_merged_attr, f)
            print('Merged, concatenated to df_meta, and saved.')
            
        return df_meta_w_merged_attr


    def get_item_meta_data_w_filtered_attributes(self, df_meta, store_path):
        df_meta_w_filtered_attr_path = store_path + 'df_meta_n' + str(self.filter_threshold) + '_filtered_attr.pkl'

        if exists(df_meta_w_filtered_attr_path):
            print('Loading item meta data (with filtered attributes, attributes that are too frequent or infrequent are filtered out) ...')
            with open(df_meta_w_filtered_attr_path, 'rb') as f:
                df_meta_w_filtered_attr = pickle.load(f)
            print('Loaded.')
        else:
            print('Flitering attributes for items...')

            merged_attr = df_meta['merged_attr'].tolist()
            print('len(merged_attr): ', len(merged_attr)) #30771
            attr_counts = Counter(list(chain.from_iterable(merged_attr)))
            attr_counts = {k: v for k, v in sorted(attr_counts.items(), key=lambda item: item[1], reverse=True)}
            #print(attr_counts)
            print('Before filtering the attrs')
            #print('attr counts: ', list(attr_counts.values()))
            print('num_attrs: ', len(attr_counts)) # 276,446
            
            #print(list(attr_counts.values())[:1000]) # [28695, 9484, 7190, 4206, 4072, 3873, 3472, 2309, 2020, 1377, 1338, 1296, 1235, 1232, 1182, 1019, ...
            #for k in list(attr_counts.keys())[:200]:
            #    print(k, ' ', attr_counts[k])
            
            # count > 4 and count < 28695: 12,130
            print('After filtering. Kept attrs with count > 4 and count < 2445')
            attr_counts = {k: v for k, v in attr_counts.items() if v > 4 and v < 28695}
            #print('attr counts: ', list(attr_counts.values()))
            print('num_attrs: ', len(attr_counts), '\n')
            
            filtered_attr = []
            count_empty = 0
            count_less_than = 0
            count_less_than_threshold = 3
            average_num_attr = 0
            for item_attrs in merged_attr:
                filtered_item_attrs = [attr for attr in item_attrs if attr in attr_counts.keys()]
                filtered_attr.append(filtered_item_attrs)
                average_num_attr += len(filtered_item_attrs)
                if not filtered_item_attrs:
                    count_empty += 1
                if len(filtered_item_attrs) < count_less_than_threshold:
                    count_less_than += 1
            average_num_attr /= len(merged_attr)
            #print('filtered_attr[:10]: ', filtered_attr[:10])
            print('count_empty: ', count_empty) # count_empty:  101
            print('count_less_than ' + str(count_less_than_threshold) + ': ', count_less_than) # count_less_than 3: 933, count_less_than 5:  6183
            print('average_num_attr: ', average_num_attr) # average_num_attr:  9.527960012496095
            
            # concatenate to df_meta
            df_meta['filtered_attr'] = filtered_attr

            # only keep items with >= count_less_than_threshold attributes
            df_meta_w_filtered_attr = df_meta.loc[df_meta.filtered_attr.map(len) >= count_less_than_threshold]

            # save the concatenated meta data
            with open(df_meta_w_filtered_attr_path, 'wb') as f:
                pickle.dump(df_meta_w_filtered_attr, f)
            print('Merged, concatenated to df_meta, and saved.')

        print('len(df_meta_w_filtered_attr): ', len(df_meta_w_filtered_attr)) # len(df_meta_w_filtered_attr):  27876
        return df_meta_w_filtered_attr

    def read_attributes(self, store_path):
        # get item meta data
        df_meta = self.get_item_meta_data(store_path = store_path)
        #print('len(df_meta): ', len(df_meta)) # len(df_meta):  30771
        #print('len(df_meta.item.unique()): ', len(df_meta.item.unique())) # len(df_meta.item.unique()):  29399
        
        # get item meta data with raw attributes extracted from 'description' and 'feature'
        df_meta_w_extracted_attr = self.get_item_meta_data_w_extracted_attributes(df_meta = df_meta, store_path = store_path)
        #print(df_meta_w_extracted_attr.head(10))

        # get item meta data with merged attributes ('category', 'brand', and 'extracted_attr', which is 
        # extracted from 'description' and 'feature')
        df_meta_w_merged_attr = self.get_item_meta_data_w_merged_attributes(df_meta = df_meta_w_extracted_attr, store_path = store_path)
        #print(df_meta_w_merged_attr.head(10))
        
        # get item meta data with filtered attributes (attributes that are too frequent/infrequent are filtered out)
        df_meta_w_filtered_attr = self.get_item_meta_data_w_filtered_attributes(df_meta = df_meta_w_merged_attr, store_path = store_path)

        return df_meta_w_filtered_attr

    def get_SCS_item_meta_data(self, store_path, SCS_items, n):
        df_meta_path = store_path + 'df_meta_SCS_' + str(n) + '_.pkl'

        if exists(df_meta_path):
            print('Loading item meta data...')
            with open(df_meta_path, 'rb') as f:
                df_meta = pickle.load(f)
            print('Loaded')
        else:
            print('Extracting item meta data...')
            columns_list = ['asin', 'category', 'description', 'feature', 'title', 'brand']
            df_meta = getselectDF_SCS(self.new_meta_path, columns_list)
            df_meta = df_meta.rename(columns={'asin':'item'})

            # select the rows for the filtered items
            df_meta = df_meta.loc[df_meta['item'].isin(SCS_items)]
            print('len(df_meta) before drop_duplicates: ', len(df_meta)) # len(df_meta) before drop_duplicates:  30771
            df_meta = df_meta.drop_duplicates(subset=['item'])
            print('len(df_meta) after drop_duplicates: ', len(df_meta)) # len(df_meta) after drop_duplicates:  29399

            # save the filtered meta data
            with open(df_meta_path, 'wb') as f:
                pickle.dump(df_meta, f)
            print('Extracted and saved.')
        
        #print(df_meta.head(10))
        return df_meta

    def get_item_meta_data_w_extracted_attributes_SCS(self, df_meta, store_path, n):
        df_meta_w_extracted_attr_path = store_path + 'df_meta_SCS_' + str(n) + '_extracted_attr.pkl'

        if exists(df_meta_w_extracted_attr_path):
            print('Loading item meta data (with raw attributes) ...')
            with open(df_meta_w_extracted_attr_path, 'rb') as f:
                df_meta_w_extracted_attr = pickle.load(f)
            print('Loaded.')
        else:
            print('Extracting raw attributes from item meta data...')
            # extract attributes from description and feature
            extracted_attr = [extract_attributes(description_list + feature_list) for description_list, feature_list in zip(df_meta['description'], df_meta['feature'])]
            # concatenate to df_meta
            df_meta['extracted_attr'] = extracted_attr
            df_meta_w_extracted_attr = df_meta
            # save the concatenated meta data
            with open(df_meta_w_extracted_attr_path, 'wb') as f:
                pickle.dump(df_meta_w_extracted_attr, f)
            print('Extracted, concatenated to df_meta, and saved.')
            
        return df_meta_w_extracted_attr

    def get_item_meta_data_w_merged_attributes_SCS(self, df_meta, store_path, n):
        df_meta_w_merged_attr_path = store_path + 'df_meta_SCS_' + str(n) + '_merged_attr.pkl'

        if exists(df_meta_w_merged_attr_path):
            print('Loading item meta data (with merged attributes) ...')
            with open(df_meta_w_merged_attr_path, 'rb') as f:
                df_meta_w_merged_attr = pickle.load(f)
            print('Loaded.')
        else:
            print('Merging attributes for items...')
            # merge the attributes in 'category', 'brand', and 'extracted_attr' (extracted from 'description' and 'feature')
            merged_attr = [list(set(category + [brand] + extracted_attr)) for category, brand, extracted_attr in zip(df_meta['category'], df_meta['brand'], df_meta['extracted_attr'])]
            # concatenate to df_meta
            df_meta['merged_attr'] = merged_attr
            df_meta_w_merged_attr = df_meta
            # save the concatenated meta data
            with open(df_meta_w_merged_attr_path, 'wb') as f:
                pickle.dump(df_meta_w_merged_attr, f)
            print('Merged, concatenated to df_meta, and saved.')
            
        return df_meta_w_merged_attr

    def get_item_meta_data_w_filtered_attributes_SCS(self, df_meta, store_path, n):
        df_meta_w_filtered_attr_path = store_path + 'df_meta_SCS_' + str(n) + '_filtered_attr.pkl'

        if exists(df_meta_w_filtered_attr_path):
            print('Loading item meta data (with filtered attributes, attributes that are too frequent or infrequent are filtered out) ...')
            with open(df_meta_w_filtered_attr_path, 'rb') as f:
                df_meta_w_filtered_attr = pickle.load(f)
            print('Loaded.')
        else:
            print('Flitering attributes for items...')

            merged_attr = df_meta['merged_attr'].tolist()
            #print('len(merged_attr): ', len(merged_attr)) #
            attr_counts = Counter(list(chain.from_iterable(merged_attr)))
            attr_counts = {k: v for k, v in sorted(attr_counts.items(), key=lambda item: item[1], reverse=True)}
            #print(attr_counts)
            print('Before filtering the attrs')
            #print('attr counts: ', list(attr_counts.values()))
            print('num_attrs: ', len(attr_counts)) # 500 num_attrs:  8881

            '''
            train_item_attrs_dict = {self.map_i_o2n[item]: attrs for item, attrs in zip(self.df_meta_w_filtered_attr['item'], self.df_meta_w_filtered_attr['filtered_attr']) \
                if self.map_i_o2n[item] in self.train_i_ratings.keys()}
            train_attrs_set = []
            for item, attrs in train_item_attrs_dict.items():
                train_attrs_set += attrs
            train_attrs_set = set(train_attrs_set)

            seen_attrs = [attr for attr in attr_counts.keys() if attr in train_attrs_set]
            print('num_seen_attrs: ', len(seen_attrs)) # 500 num_seen_attrs:  2413
            '''
            #print(list(attr_counts.values())[:500]) # [438, 255, 170, 136, 71, 67, 61, 60, 53, 48, 39, 36, 35, 34, 33, 32, ...
            #for k in list(attr_counts.keys())[:100]:
            #    print(k, ' ', attr_counts[k])
            
            # count > 4: 91,667
            print('After filtering. Kept attrs with count > 4')
            attr_counts = {k: v for k, v in attr_counts.items() if v > 4 }
            #print('attr counts: ', list(attr_counts.values()))
            print('num_attrs: ', len(attr_counts), '\n')
            
            filtered_attr = []
            count_empty = 0
            count_less_than = 0
            count_less_than_threshold = 3
            average_num_attr = 0
            for item_attrs in merged_attr:
                filtered_item_attrs = [attr for attr in item_attrs if attr in attr_counts.keys()]
                filtered_attr.append(filtered_item_attrs)
                average_num_attr += len(filtered_item_attrs)
                if not filtered_item_attrs:
                    count_empty += 1
                if len(filtered_item_attrs) < count_less_than_threshold:
                    count_less_than += 1
            average_num_attr /= len(merged_attr)
            #print('filtered_attr[:10]: ', filtered_attr[:10])
            print('count_empty: ', count_empty) 
            print('count_less_than ' + str(count_less_than_threshold) + ': ', count_less_than) 
            print('average_num_attr: ', average_num_attr) 

            train_item_attrs_dict = {self.map_i_o2n[item]: attrs for item, attrs in zip(self.df_meta_w_filtered_attr['item'], self.df_meta_w_filtered_attr['filtered_attr']) \
                if self.map_i_o2n[item] in self.train_i_ratings.keys()}
            train_attrs_set = []
            for item, attrs in train_item_attrs_dict.items():
                train_attrs_set += attrs
            train_attrs_set = set(train_attrs_set)

            seen_attrs = [attr for attr in attr_counts.keys() if attr in train_attrs_set]
            print('num_seen_attrs: ', len(seen_attrs)) 

            # concatenate to df_meta
            df_meta['filtered_attr'] = filtered_attr

            # only keep items with >= count_less_than_threshold attributes
            df_meta_w_filtered_attr = df_meta.loc[df_meta.filtered_attr.map(len) >= count_less_than_threshold]

            # save the concatenated meta data
            with open(df_meta_w_filtered_attr_path, 'wb') as f:
                pickle.dump(df_meta_w_filtered_attr, f)
            print('Merged, concatenated to df_meta, and saved.')

        print('len(df_meta_w_filtered_attr): ', len(df_meta_w_filtered_attr))
        return df_meta_w_filtered_attr

    def read_attributes_SCS(self, store_path, n):
        load_path = '/home/anonymous/amazon_sports_data/SCS_interactions_' + str(n) + '_filtered.pkl'
        with open(load_path, 'rb') as f:
            SCS_interactions = pickle.load(f)
        SCS_items = set([each['item'] for each in SCS_interactions])

        # get item meta data
        df_meta_SCS = self.get_SCS_item_meta_data(store_path = store_path, SCS_items = SCS_items, n = n)
        
        # get item meta data with raw attributes extracted from 'description' and 'feature'
        df_meta_w_extracted_attr_SCS = self.get_item_meta_data_w_extracted_attributes_SCS(df_meta = df_meta_SCS, store_path = store_path, n = n)
        #print(df_meta_w_extracted_attr_SCS.head(10)['extracted_attr'])
        
        # get item meta data with merged attributes ('category', 'brand', and 'extracted_attr', which is 
        # extracted from 'description' and 'feature')
        df_meta_w_merged_attr_SCS = self.get_item_meta_data_w_merged_attributes_SCS(df_meta = df_meta_w_extracted_attr_SCS, store_path = store_path, n = n)
        #print(df_meta_w_merged_attr_SCS.head(10)['merged_attr'])
        
        # get item meta data with filtered attributes (attributes that are too frequent/infrequent are filtered out)
        df_meta_w_filtered_attr_SCS = self.get_item_meta_data_w_filtered_attributes_SCS(df_meta = df_meta_w_merged_attr_SCS, store_path = store_path, n = n)

        return df_meta_w_filtered_attr_SCS
        

    def filter_ratings(self, store_path):
        print('Inside filter_ratings')
        i_ratings_filtered_path = store_path + 'i_ratings_filtered_n' + str(self.filter_threshold) + '.pkl'

        if exists(i_ratings_filtered_path):
            print('Loading the filtered ratings data...')
            with open(i_ratings_filtered_path, 'rb') as f:
                i_ratings_filtered = pickle.load(f)
            print('Loaded.')
        else:
            print('Filtering the ratings data, only keep items that have enough attributes ...')
            # only keep the items with enough attributes
            i_ratings_filtered = {item: sequence for item, sequence in self.i_ratings.items() if item in self.df_meta_w_filtered_attr.item.unique()}
            assert len(i_ratings_filtered) == len(self.df_meta_w_filtered_attr)

            # sort i_ratings_filtered based on sequence length
            i_ratings_filtered_sorted = {}
            for k in sorted(i_ratings_filtered, key=lambda k: len(i_ratings_filtered[k]), reverse=True):
                i_ratings_filtered_sorted[k] = i_ratings_filtered[k]
            print("Two longest sequences:")
            for item in list(i_ratings_filtered_sorted.keys())[:2]:
                print(i_ratings_filtered_sorted[item])
            print("Two shortest sequences:")
            for item in list(i_ratings_filtered_sorted.keys())[-2:]:
                print(i_ratings_filtered_sorted[item])

            i_ratings_filtered = i_ratings_filtered_sorted
            # save the filtered data
            with open(i_ratings_filtered_path, 'wb') as f:
                pickle.dump(i_ratings_filtered, f)
            print('Filtered and saved.')

        return i_ratings_filtered


    def split_ratings(self, store_path):
        #train_i_ratings_mapped, val_i_ratings_mapped, test_i_ratings_mapped, map_i_o2n, map_u_o2n, \
        #    num_train_items, num_val_items, num_test_items, num_train_ratings, num_val_ratings, num_test_ratings
        train_i_ratings_path = store_path + 'train_i_ratings.pkl'
        val_i_ratings_path = store_path + 'val_i_ratings.pkl'
        #test_i_ratings_path = store_path + 'test_i_ratings.pkl'
        map_i_o2n_path = store_path + 'map_i_o2n.pkl'
        map_u_o2n_path = store_path + 'map_u_o2n.pkl'
        statistics_path = store_path + 'statistics.pkl'

        if exists(train_i_ratings_path):
            print('Loading the train, val, test rating sequences, the item and user id maps, and the statistics...')
            with open(train_i_ratings_path, 'rb') as f:
                train_i_ratings = pickle.load(f)
            with open(val_i_ratings_path, 'rb') as f:
                val_i_ratings = pickle.load(f)
            #with open(test_i_ratings_path, 'rb') as f:
            #    test_i_ratings = pickle.load(f)
            with open(map_i_o2n_path, 'rb') as f:
                map_i_o2n = pickle.load(f)
            with open(map_u_o2n_path, 'rb') as f:
                map_u_o2n = pickle.load(f)
            with open(statistics_path, 'rb') as f:
                statistics = pickle.load(f)
            print('Loaded.')
            #return train_i_ratings, val_i_ratings, test_i_ratings, map_i_o2n, map_u_o2n, statistics
            return train_i_ratings, val_i_ratings, map_i_o2n, map_u_o2n, statistics
        
        else:
            print('Extracting the train, val, test rating sequences, the item and user id maps, and the statistics...')
            
            # 0~val_start: training items; val_start~train_end: val items;
            train_end = math.ceil(len(self.i_ratings_filtered))
            val_start = math.ceil(0.95*train_end)

            # split the ratings
            train_i_ratings, val_i_ratings = {}, {}
            for item in list(self.i_ratings_filtered.keys())[:val_start]:
                train_i_ratings[item] = self.i_ratings_filtered[item]
            for item in list(self.i_ratings_filtered.keys())[val_start:train_end]:
                val_i_ratings[item] = self.i_ratings_filtered[item]
            num_train_items = len(train_i_ratings)
            num_val_items = len(val_i_ratings)
            
            # get a set of unique users in the training set
            train_users = []
            for item, sequence in train_i_ratings.items():
                train_users += list(sequence.keys())
            train_users = set(train_users)
            print('len(train_users): ', len(train_users))
            
            # filter val_i_ratings to make sure that all the users involved 
            # in them are also in the train_i_ratings (i.e., are old users)
            # count the number of train, val, and test ratings
            num_train_ratings, num_val_ratings = 0, 0
            count_empty_val_items = 0
            for item, sequence in train_i_ratings.items():
                num_train_ratings += len(sequence)
            for item, sequence in val_i_ratings.items():
                val_i_ratings[item] = {user:rating for user, rating in sequence.items() if user in train_users}
                num_val_ratings += len(val_i_ratings[item])
                if len(val_i_ratings[item]) == 0:
                    count_empty_val_items += 1
            # make sure none of the val sequences become empty after filtering out the non-training users
            #print('count_empty_val_items: ', count_empty_val_items)
            #assert count_empty_val_items == 0
            
            # generate id maps
            map_i_o2n = {old_id:new_id for new_id, old_id in enumerate(self.i_ratings_filtered)}
            map_u_o2n = {old_id:new_id for new_id, old_id in enumerate(train_users)}

            # mapping
            train_i_ratings_mapped, val_i_ratings_mapped = {}, {}
            for item, sequence in train_i_ratings.items():
                train_i_ratings_mapped[map_i_o2n[item]] = {map_u_o2n[user]:rating for user, rating in train_i_ratings[item].items()}
            for item, sequence in val_i_ratings.items():
                val_i_ratings_mapped[map_i_o2n[item]] = {map_u_o2n[user]:rating for user, rating in val_i_ratings[item].items()}
            #for item, sequence in test_i_ratings.items():
            #    test_i_ratings_mapped[map_i_o2n[item]] = {map_u_o2n[user]:rating for user, rating in test_i_ratings[item].items()}

            # construct statistics
            statistics = {'num_train_items': num_train_items, 'num_val_items': num_val_items, \
                'num_train_ratings': num_train_ratings, 'num_val_ratings': num_val_ratings}
            print('statistics: ', statistics)
            
            # Save the train, val rating sequences, the item and user id maps, and the statistics

            print('Extracted and saved.')
            with open(train_i_ratings_path, 'wb') as f:
                pickle.dump(train_i_ratings_mapped, f)
            with open(val_i_ratings_path, 'wb') as f:
                pickle.dump(val_i_ratings_mapped, f)
            #with open(test_i_ratings_path, 'wb') as f:
            #    pickle.dump(test_i_ratings_mapped, f)
            with open(map_i_o2n_path, 'wb') as f:
                pickle.dump(map_i_o2n, f)
            with open(map_u_o2n_path, 'wb') as f:
                pickle.dump(map_u_o2n, f)
            with open(statistics_path, 'wb') as f:
                pickle.dump(statistics, f)

            return train_i_ratings_mapped, val_i_ratings_mapped, map_i_o2n, map_u_o2n, statistics


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


    def get_u_ratings(self, store_path):
        u_ratings_path = store_path + 'u_ratings.pkl'

        if exists(u_ratings_path):
            print('Loading the u_ratings...')
            with open(u_ratings_path, 'rb') as f:
                u_ratings = pickle.load(f)
            print('Loaded.')

        else:
            print('Constructing the u_ratings...')
            u_ratings = {}
            for item, sequence in self.train_i_ratings.items():
                for user, rating in sequence.items():
                    if user not in u_ratings:
                        u_ratings[user] = {}
                    u_ratings[user][item] = rating

            with open(u_ratings_path, 'wb') as f:
                pickle.dump(u_ratings, f)
            print('Constructed and saved.')

        # len(u_ratings): 96420
        # len_counts:  {339: 1, 265: 1, 247: 1, 244: 1, 243: 1, ... 5: 3240, 4: 2271, 3: 1373, 2: 847, 1: 453}
        '''
        print('len(u_ratings): ', len(u_ratings)) # len(u_ratings): 96420
        s_lengths = [len(sequence) for user, sequence in u_ratings.items()]
        len_counts = Counter(s_lengths)
        len_counts = {length: count for length, count in sorted(len_counts.items(), key=lambda item: item[0], reverse=True)}
        print('len_counts: ', len_counts) # len_counts:  {339: 1, 265: 1, 247: 1, 244: 1, 243: 1, ... 5: 3240, 4: 2271, 3: 1373, 2: 847, 1: 453}
        '''
        return u_ratings


    def task1(self, store_path, n):
        train_attrs_set_path = store_path + 'train_attrs_set.pkl'

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
        test_item_attrs_dict_path = store_path + 'test_item_attrs_dict_' + str(n) + '.pkl'

        if exists(train_attrs_set_path):
            with open(train_attrs_set_path, 'rb') as f:
                train_attrs_set = pickle.load(f)
                
            print('Loading the graphs and the train edges...')
            with open(train_graph_path, 'rb') as f:
                train_graph = pickle.load(f)
            with open(train_edges_path, 'rb') as f:
                train_edges = pickle.load(f)
            with open(val_graph_path, 'rb') as f:
                val_graph = pickle.load(f)
            print('Loaded the graphs and the train edges.')

            print('Loading the attribute dicts...')
            with open(train_attrs_dict_path, 'rb') as f:
                train_attrs_dict = pickle.load(f)
            with open(new_val_attrs_dict_path, 'rb') as f:
                new_val_attrs_dict = pickle.load(f)
            print('Loaded the attribute dicts.')

            print('Loading the attribute embeddings...')
            with open(train_attrs_embedding_path, 'rb') as f:
                train_attrs_embedding = pickle.load(f)
            with open(new_val_attrs_embedding_path, 'rb') as f:
                new_val_attrs_embedding = pickle.load(f)
            print('Loaded the attribute embeddings.')

            print('Loading the item embeddings...')
            with open(train_items_embedding_path, 'rb') as f:
                train_items_embedding = pickle.load(f)
            with open(val_items_embedding_path, 'rb') as f:
                val_items_embedding = pickle.load(f)
            print('Loaded the item embeddings.')
        
            with open(train_item_attrs_dict_path, 'rb') as f:
                train_item_attrs_dict = pickle.load(f)
            with open(val_item_attrs_dict_path, 'rb') as f:
                val_item_attrs_dict = pickle.load(f)
            print('Loaded the item attribute dicts')
        else:
            print('Constructing the graphs...')
            # get the attributes of training, val, and test items
            train_item_attrs_dict = {self.map_i_o2n[item]: attrs for item, attrs in zip(self.df_meta_w_filtered_attr['item'], self.df_meta_w_filtered_attr['filtered_attr']) \
                if self.map_i_o2n[item] in self.train_i_ratings.keys()}
            val_item_attrs_dict = {self.map_i_o2n[item]: attrs for item, attrs in zip(self.df_meta_w_filtered_attr['item'], self.df_meta_w_filtered_attr['filtered_attr']) \
                if self.map_i_o2n[item] in self.val_i_ratings.keys()}

            # save the item attribute dicts
            with open(train_item_attrs_dict_path, 'wb') as fp:
                pickle.dump(train_item_attrs_dict, fp)
            with open(val_item_attrs_dict_path, 'wb') as fp:
                pickle.dump(val_item_attrs_dict, fp)
            print('Saved the item attribute dicts')

            # get the attr dicts for training, val
            train_attrs_set, val_attrs_set = [], []
            for item, attrs in train_item_attrs_dict.items():
                train_attrs_set += attrs
            train_attrs_set = set(train_attrs_set)
            train_attrs_dict = {attr:id for id, attr in enumerate(list(train_attrs_set))}
            with open(train_attrs_set_path, 'wb') as fp:
                pickle.dump(train_attrs_set, fp)

            for item, attrs in val_item_attrs_dict.items():
                val_attrs_set += attrs
            val_attrs_set = set(val_attrs_set)
            # a set of new val item attributes that are not in the training set
            new_val_attrs_set = val_attrs_set - train_attrs_set
            new_val_attrs_dict = {attr:id+len(train_attrs_dict) for id, attr in enumerate(list(new_val_attrs_set))}
            #print('new_val_attrs_dict: ', new_val_attrs_dict)

            # generate graphs for training, val
            train_graph, train_edges = self.generate_item_attribute_graph(item_attrs_dict = train_item_attrs_dict, \
                attrs_dict = train_attrs_dict, mode = 'train')
            val_graph, _ = self.generate_item_attribute_graph(item_attrs_dict = val_item_attrs_dict, \
                attrs_dict = {**train_attrs_dict, **new_val_attrs_dict}, mode = 'valid', train_edges = train_edges)
            
            # save the graphs and train_edges
            with open(train_graph_path, 'wb') as fp:
                pickle.dump(train_graph, fp)
            print('Saved train_graph')
            with open(train_edges_path, 'wb') as fp:
                pickle.dump(train_edges, fp)
            print('Saved train_edges')
            with open(val_graph_path, 'wb') as fp:
                pickle.dump(val_graph, fp)
            print('Saved val_graph')

            # save the attribute dicts
            with open(train_attrs_dict_path, 'wb') as fp:
                pickle.dump(train_attrs_dict, fp)
            with open(new_val_attrs_dict_path, 'wb') as fp:
                pickle.dump(new_val_attrs_dict, fp)
            print('Saved the attribute dicts')

            # get the initial embeddings for attributes
            print('Extracting attribute embeddings...')
            train_attrs_embedding = SBERT_embed(list(train_attrs_dict.keys()))
            with open(train_attrs_embedding_path, 'wb') as fp:
                pickle.dump(train_attrs_embedding, fp)
            print('Saved train_attrs_embedding')

            if len(new_val_attrs_dict) != 0:
                new_val_attrs_embedding = SBERT_embed(list(new_val_attrs_dict.keys()))
            else:
                new_val_attrs_embedding = torch.empty(0, 1)
            with open(new_val_attrs_embedding_path, 'wb') as fp:
                pickle.dump(new_val_attrs_embedding, fp)
            print('Saved new_val_attrs_embedding')

            # get the initial embeddings for items
            print('Extracting item embeddings...')
            train_items_list = [self.map_i_n2o[item] for item in self.train_i_ratings.keys()]
            train_items_titles = self.df_meta_w_filtered_attr.loc[self.df_meta_w_filtered_attr['item'].isin(train_items_list)]['title'].tolist()
            #print('train_items_titles[:10]: ', train_items_titles[:10])
            train_items_titles = [str(each) for each in train_items_titles]
            assert len(train_items_titles) == len(self.train_i_ratings)
            train_items_embedding = SBERT_embed(train_items_titles)
            with open(train_items_embedding_path, 'wb') as fp:
                pickle.dump(train_items_embedding, fp)
            print('Saved train_items_embedding')

            val_items_list = [self.map_i_n2o[item] for item in self.val_i_ratings.keys()]
            val_items_titles = self.df_meta_w_filtered_attr.loc[self.df_meta_w_filtered_attr['item'].isin(val_items_list)]['title'].tolist()
            val_items_embedding = SBERT_embed(val_items_titles)
            with open(val_items_embedding_path, 'wb') as fp:
                pickle.dump(val_items_embedding, fp)
            print('Saved val_items_embedding')
        
        if exists(test_graph_path):
            with open(test_graph_path, 'rb') as f:
                test_graph = pickle.load(f)
            with open(new_test_attrs_dict_path, 'rb') as f:
                new_test_attrs_dict = pickle.load(f)
            with open(new_test_attrs_embedding_path, 'rb') as f:
                new_test_attrs_embedding = pickle.load(f)
            with open(test_items_embedding_path, 'rb') as f:
                test_items_embedding = pickle.load(f)
            with open(test_item_attrs_dict_path, 'rb') as f:
                test_item_attrs_dict = pickle.load(f)
        else:
            test_item_attrs_dict = {self.map_i_o2n_SCS[item]: attrs for item, attrs in zip(self.df_meta_w_filtered_attr_SCS['item'], self.df_meta_w_filtered_attr_SCS['filtered_attr']) \
                if self.map_i_o2n_SCS[item] in self.test_i_ratings.keys()}

            with open(test_item_attrs_dict_path, 'wb') as fp:
                pickle.dump(test_item_attrs_dict, fp)
            
            test_attrs_set = []
            for item, attrs in test_item_attrs_dict.items():
                test_attrs_set += attrs
            test_attrs_set = set(test_attrs_set)
            # a set of new test item attributes that are not in the training set
            new_test_attrs_set = test_attrs_set - train_attrs_set
            new_test_attrs_dict = {attr:id+len(train_attrs_dict) for id, attr in enumerate(list(new_test_attrs_set))}

            test_graph, _ = self.generate_item_attribute_graph(item_attrs_dict = test_item_attrs_dict, \
                attrs_dict = {**train_attrs_dict, **new_test_attrs_dict}, mode = 'test', train_edges = train_edges)
            
            with open(test_graph_path, 'wb') as fp:
                pickle.dump(test_graph, fp)
            print('Saved test_graph')

            with open(new_test_attrs_dict_path, 'wb') as fp:
                pickle.dump(new_test_attrs_dict, fp)
            
            new_test_attrs_embedding = SBERT_embed(list(new_test_attrs_dict.keys()))
            with open(new_test_attrs_embedding_path, 'wb') as fp:
                pickle.dump(new_test_attrs_embedding, fp)
            print('Saved new_test_attrs_embedding')

            test_items_list = [self.map_i_n2o_SCS[item] for item in self.test_i_ratings.keys()]
            test_items_titles = self.df_meta_w_filtered_attr_SCS.loc[self.df_meta_w_filtered_attr_SCS['item'].isin(test_items_list)]['title'].tolist()
            test_items_embedding = SBERT_embed(test_items_titles)
            with open(test_items_embedding_path, 'wb') as fp:
                pickle.dump(test_items_embedding, fp)
            print('Saved test_items_embedding')

        return train_graph, train_edges, val_graph, test_graph, \
            train_attrs_dict, new_val_attrs_dict, new_test_attrs_dict, \
            train_attrs_embedding, new_val_attrs_embedding, new_test_attrs_embedding, \
            train_items_embedding, val_items_embedding, test_items_embedding, \
            train_item_attrs_dict, val_item_attrs_dict, test_item_attrs_dict

    def get_also_buy_view(self, store_path):
        df_also_buy_view_path = store_path + 'df_also_buy_view.pkl'

        if exists(df_also_buy_view_path):
            print('Loading df_also_buy_view...')
            with open(df_also_buy_view_path, 'rb') as f:
                df_also_buy_view = pickle.load(f)
            print('Loaded.')
        else:
            print('Constructing df_also_buy_view...')

            columns_list = ['asin', 'also_view', 'also_buy']
            df_also_buy_view = getselectDF(self.meta_path, columns_list)
            df_also_buy_view = df_also_buy_view.rename(columns={'asin':'item'})
            #print(' extracted raw meta data')
            train_items = [self.map_i_n2o[item] for item in self.train_i_ratings.keys()]
            #print(' got train items')
            df_also_buy_view = df_also_buy_view.loc[df_also_buy_view['item'].isin(train_items)]
            #print(' extracted train items meta data')
            
            item_mapped = [self.map_i_o2n[item] for item in df_also_buy_view['item']]
            df_also_buy_view['item_mapped'] = item_mapped
            
            also_view_filtered_mapped = [[self.map_i_o2n[item] for item in sequence if item in self.map_i_o2n.keys() \
                and self.map_i_o2n[item] in self.train_i_ratings.keys()] \
                for sequence in df_also_buy_view['also_view']]
            also_buy_filtered_mapped = [[self.map_i_o2n[item] for item in sequence if item in self.map_i_o2n.keys() \
                and self.map_i_o2n[item] in self.train_i_ratings.keys()] \
                for sequence in df_also_buy_view['also_buy']]
            df_also_buy_view['also_view_filtered_mapped'] = also_view_filtered_mapped
            df_also_buy_view['also_buy_filtered_mapped'] = also_buy_filtered_mapped

            also_view_minus_also_buy = [[item for item in also_view if item not in also_buy] \
                for also_view, also_buy in zip(df_also_buy_view['also_view_filtered_mapped'], df_also_buy_view['also_buy_filtered_mapped'])]
            df_also_buy_view['also_view_minus_also_buy'] = also_view_minus_also_buy
            
            with open(df_also_buy_view_path, 'wb') as fp:
                pickle.dump(df_also_buy_view, fp)
            print('Constructed and saved.')

        
            num_item_w_also_view = len(df_also_buy_view[df_also_buy_view.also_view_filtered_mapped.astype(bool)])
            num_item_w_also_buy = len(df_also_buy_view[df_also_buy_view.also_buy_filtered_mapped.astype(bool)])
            num_item_w_also_view_minus_also_buy = len(df_also_buy_view[df_also_buy_view.also_view_minus_also_buy.astype(bool)])
            print('num_item_w_also_view: ', num_item_w_also_view)
            print('num_item_w_also_buy: ', num_item_w_also_buy)
            print('num_item_w_also_view_minus_also_buy: ', num_item_w_also_view_minus_also_buy)
            '''
            num_item_w_also_view:  9161
            num_item_w_also_buy:  13845
            num_item_w_also_view_minus_also_buy:  8738
            '''
        
        return df_also_buy_view


    def task2(self, store_path):
        return
    
    def task3(self, store_path):
        return
    
    '''
    def check_test_item_attrs(self):
        load_path = '/home/anonymous/amazon_sports_data/SCS_interactions_500_filtered.pkl'
        #self.new_meta_path

        return
    '''
    def filter_test_items(self):
        load_path = '/home/anonymous/amazon_sports_data/SCS_interactions_520.pkl'
        store_path = '/home/anonymous/amazon_sports_data/SCS_interactions_520_filtered.pkl'
        with open(load_path, "rb") as f:
            interaction_data = pickle.load(f)
        distinct_item = set([each['item'] for each in interaction_data])
        distinct_item_count = len(distinct_item)
        distinct_user = set([each['user'] for each in interaction_data])
        distinct_user_count = len(distinct_user)
        print('distinct_item_count: ', distinct_item_count)
        print('distinct_user_count: ', distinct_user_count)
        #print('distinct_user: ', distinct_user)
        overlap_item_count = 0 # should be 0
        overlap_items = []
        overlap_user_count = 0 # the higher the better
        #overlap_users = []
        for item in distinct_item:
            if item in self.map_i_o2n.keys():
                overlap_item_count += 1
                overlap_items.append(item)
        for user in distinct_user:
            if user in self.map_u_o2n.keys():
                overlap_user_count += 1
        print('overlap_item_count: ', overlap_item_count)
        print('overlap_items: ', overlap_items)
        print('overlap_user_count: ', overlap_user_count)

        SCS_items = [item for item in distinct_item if item not in overlap_items]
        print('num_SCS_items: ', len(SCS_items))
        SCS_interactions = [each for each in interaction_data if each['item'] in SCS_items]
        print('num_SCS_interactions: ', len(SCS_interactions))

        with open(store_path, 'wb') as f:
            pickle.dump(SCS_interactions, f)
        '''
        500
        #len(SCS_interactions_500):  1754
        distinct_item_count:  497
        distinct_user_count:  1627
        overlap_item_count:  41
        overlap_items:  ['B004UL4FJW', 'B001DCG8GY', 'B00AW8H1JS', 'B000SII2FE', 'B000XEAUDK', 'B0053YQCJG', 'B000YBCNXW', 'B000F3O9F4', 'B0032WSXCA', 'B000BR5MOG', 'B00B4IHXRU', 'B000K7I64K', 'B006JYHI24', 'B00094HAA0', 'B008RI1B14', 'B001C63ECM', 'B000RY7STA', 'B009D69B9I', 'B00131YQO6', 'B0000C521L', 'B007JYUOFG', 'B000F3NYOG', 'B000JIPKO4', 'B0001MQ7A4', 'B0028WKDHS', 'B001PYXKZW', 'B004DM8EOK', 'B003RLLD2I', 'B000ENWACY', 'B00058OCM4', 'B000Y7VRWE', 'B000MVBJMA', 'B004FPZJBG', 'B001QXJ916', 'B003ZZ53NG', 'B000N9FFXK', 'B00BAKUF2M', 'B000YKK7TU', 'B004RNCE5A', 'B0016KGMKK', 'B000BT7HXI']
        overlap_user_count:  1627
        #====
        num_SCS_items:  456
        num_SCS_interactions:  1578
        #====

        510
        # len(SCS_interactions_510):  1847
        distinct_item_count:  500
        distinct_user_count:  1722
        overlap_item_count:  34
        overlap_items:  ['B0045E7L18', 'B001CWZBS6', 'B005P0NPKM', 'B004OWQILK', 'B00E459QEI', 'B00BUH45DU', 'B000IXKCCU', 'B00B1I9TB6', 'B000UNZX36', 'B0016KGMKK', 'B00ASSER44', 'B000EZ0AW4', 'B008HONGRA', 'B0045KDCLA', 'B000ZPJQ7I', 'B00124UKJY', 'B007ZYH4BM', 'B0010EN700', 'B00162RS7O', 'B00BRXK6L2', 'B004Q3PE30', 'B0094AZXYK', 'B001QBKFFC', 'B000FDVVF0', 'B001HN5I2E', 'B000ZFRFDK', 'B005NI433Y', 'B002CL1090', 'B001EZNZKC', 'B0000VVB4S', 'B00FXHSLRE', 'B001G60JA8', 'B001683886', 'B0039IPSUM']
        overlap_user_count:  1722
        #====
        num_SCS_items:  466
        num_SCS_interactions:  1710
        #====

        520
        #len(SCS_interactions_520):  1754
        distinct_item_count:  520
        distinct_user_count:  1670
        overlap_item_count:  38
        overlap_items:  ['B0049TYBL2', 'B003TLS9HS', 'B00545U97U', 'B00DQFV2NK', 'B000QOM6P2', 'B004YPQEUW', 'B000OF91EW', 'B006CUDR5C', 'B0009O8ZJ0', 'B0058YRVZ0', 'B002V32HP0', 'B00165P4UE', 'B0041X1BAK', 'B005NHKRLM', 'B001IA4K34', 'B003ZVK5RO', 'B003JSY1BO', 'B00AU6G64S', 'B0055MBGRE', 'B00E80WC86', 'B005TY5PXY', 'B001UYWTV8', 'B003DQTISI', 'B001KGA10M', 'B0012SDUR4', 'B000N8OTME', 'B004X166UU', 'B004VQ5VDA', 'B0073H2O48', 'B001CJX4WO', 'B00A9KDW3S', 'B001PS8K4Y', 'B007VSXF4M', 'B0024QV3F4', 'B001RCEXYO', 'B000WFDD2K', 'B000N8JX20', 'B000BO63LK']
        overlap_user_count:  1670
        #====
        num_SCS_items:  482
        num_SCS_interactions:  1621
        #====
        '''
        return

def debug_meta():
    store_path = "/home/anonymous/amazon_sports_data/preprocess_0126/"
    n = 1
    i_ratings_path = store_path + 'i_ratings_n' + str(n) + '.pkl'
    with open(i_ratings_path, 'rb') as f:
        i_ratings = pickle.load(f)

    items = ['0000032069', '0000031909', '0000032034', '0000031852', '0000032050', '0000031895', '0188477284', '0531904822', '059445039X', '060791548X']
    for item in items:
        print(item in i_ratings.keys())
    return

def extract_new_items():
    load_path = '/home/anonymous/github/P5/raw_data/zeroshot_test_data_item=520.pkl'
    save_path = '/home/anonymous/amazon_sports_data/SCS_interactions_520.pkl'

    with open(load_path, "rb") as f:
        raw_data = pickle.load(f)
   
    SCS_interactions_520 = [{'user': each['reviewerID'], 'item': each['asin']} for each in raw_data if each['ground_truth'] == True]
    #print('len(SCS_interactions_500): ', len(SCS_interactions_500)) #len(SCS_interactions_500):  1754
    #print('len(SCS_interactions_510): ', len(SCS_interactions_510)) #len(SCS_interactions_510):  1847
    print('len(SCS_interactions_520): ', len(SCS_interactions_520)) #len(SCS_interactions_520):  1754
    with open(save_path, 'wb') as f:
        pickle.dump(SCS_interactions_520, f)
    return

def check_data_overlap():
    # overlapped 500 items
    overlap_items = ['B004UL4FJW', 'B001DCG8GY', 'B00AW8H1JS', 'B000SII2FE', 'B000XEAUDK', 'B0053YQCJG', 'B000YBCNXW', 'B000F3O9F4', 'B0032WSXCA', 'B000BR5MOG', 'B00B4IHXRU', 'B000K7I64K', 'B006JYHI24', 'B00094HAA0', 'B008RI1B14', 'B001C63ECM', 'B000RY7STA', 'B009D69B9I', 'B00131YQO6', 'B0000C521L', 'B007JYUOFG', 'B000F3NYOG', 'B000JIPKO4', 'B0001MQ7A4', 'B0028WKDHS', 'B001PYXKZW', 'B004DM8EOK', 'B003RLLD2I', 'B000ENWACY', 'B00058OCM4', 'B000Y7VRWE', 'B000MVBJMA', 'B004FPZJBG', 'B001QXJ916', 'B003ZZ53NG', 'B000N9FFXK', 'B00BAKUF2M', 'B000YKK7TU', 'B004RNCE5A', 'B0016KGMKK', 'B000BT7HXI']
    
    ratings_path = '/data1/anonymousanonymous/datasets/amazon/data/amazon_old_data/amazon_ratings/ratings_Sports_and_Outdoors.csv'
    raw_rating_df = pd.read_csv(ratings_path, sep=",", header=None, names=["user", "item", "rating", "time"])
    old_items = set(raw_rating_df['item'].tolist())
    old_users = set(raw_rating_df['user'].tolist())

    #seen_item_count = 0
    #for item in overlap_items:
    #    print(item in old_items)

    #print('seen_item_count: ', seen_item_count) # seen_user_count: 
    return

if __name__ == "__main__":
    dataloader = MultiTaskDataPreprocessor(n = 520)
    #dataloader.task1(store_path = "/home/anonymous/amazon_home_data/preprocess_0920/")
    #debug_meta()
    #extract_new_items()
    #check_data_overlap()
    '''
    train_attrs_set = ['attr1', 'attr2', 'attr3']
    train_attrs_dict = {attr:id for id, attr in enumerate(list(train_attrs_set))}
    new_test_attrs_set = ['attr4', 'attr5']
    new_test_attrs_dict = {attr:id+len(train_attrs_dict) for id, attr in enumerate(list(new_test_attrs_set))}
    print(train_attrs_dict)
    print(new_test_attrs_dict)
    attrs_dict = {**train_attrs_dict, **new_test_attrs_dict}
    print(attrs_dict)
    

    #Our sentences we like to encode
    sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.',
    'token one',
    'cheese cake']
    embeddings = SBERT_embed(sentences)
    print(embeddings)
    print(embeddings.size()) # torch.Size([5, 384])
    '''
    
    '''
    import numpy as np
    num_train_items = 10
    items = torch.LongTensor([3, 6, 7, 9])
    #items_neg = torch.randint(num_train_items, (len(items), ))
    #print(items_neg)
    choices = list(set([i for i in range(num_train_items)]) - set(items.tolist()))
    #print(choices) # [0, 1, 2, 4, 5, 8]
    items_neg = np.random.choice(choices, len(items))
    items_neg = torch.from_numpy(items_neg).type(torch.LongTensor)
    print(items_neg)
    

    test_set_his_i = [[1, 2], [2, 1, 3, 5]]
    test_set_his_i = [torch.tensor(each, dtype = torch.long) for each in test_set_his_i]
    print(test_set_his_i)
    test_set_his_i = torch.nn.utils.rnn.pad_sequence(test_set_his_i, batch_first = True, padding_value = 0.)
    print(test_set_his_i)
    '''
    '''
    also_view = []
    also_buy = [3, 4]
    #print(also_view - also_buy)
    print([item for item in also_view if item not in also_buy])
    '''

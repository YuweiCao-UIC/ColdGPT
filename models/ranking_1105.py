import sys
#sys.path.append('/home/anonymous/Attribute_cold_start_amazon_home/')
sys.path.append('/home/anonymous/Attribute_cold_start_amazon_sports/')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils.EarlyStop import EarlyStoppingCriterion
from sklearn.metrics import ndcg_score
import numpy as np
from tqdm import tqdm
import pickle
import os
#from utils.multi_task_dataloader_amazon_home_1105 import MultiTaskDataLoader
from utils.multi_task_dataloader_amazon_sports_0127 import MultiTaskDataLoader
from utils.parser_1105 import parse_args
from model_hete_1105 import HeteModel

def load_new_embeddings():
    if os.path.exists(new_store_path):
        with open(new_store_path, 'rb') as fp:
            new_embeddings = pickle.load(fp)
    else:
        train_graph = mt_dataloader.train_graph
        attr_embedding = mt_dataloader.train_attrs_embedding
        item_embedding = mt_dataloader.train_items_embedding

        model = HeteModel(args = args, num_items = num_train_items, num_attrs = num_train_attrs, \
                graph = train_graph, device = device, item_embedding = item_embedding, attr_embedding = attr_embedding, \
                t1 = True, t2 = args.t2, t3 = args.t3)

        model.load_model(model_path)
        model.to(device)

        model.num_items += mt_dataloader.num_test_items
        model.num_attrs += len(mt_dataloader.new_test_attrs_dict)
        model.task1.encoder.item_init_embeddings = torch.cat((model.task1.encoder.item_init_embeddings, mt_dataloader.test_items_embedding), 0)
        model.task1.encoder.attr_init_embeddings = torch.cat((model.task1.encoder.attr_init_embeddings, mt_dataloader.new_test_attrs_embedding), 0)
        model.task1.graph = mt_dataloader.test_graph.to(device)
        new_embeddings = model.task1.extract_embeddings().clone().detach()

        with open(new_store_path, 'wb') as fp:
            pickle.dump(new_embeddings, fp)

    #print('new_embeddings.size(): ', new_embeddings.size())

    item_embeddings = new_embeddings[:(mt_dataloader.num_train_items + mt_dataloader.num_test_items)]
    attr_embeddings = new_embeddings[(mt_dataloader.num_train_items + mt_dataloader.num_test_items):]

    # np.set_printoptions(threshold=sys.maxsize)
    # print('item_embeddings: ', item_embeddings.numpy())
    # print('attr_embeddings: ', attr_embeddings.numpy())
    # print('len(attr_embeddings): ', len(attr_embeddings))
    # print('len(mt_dataloader.train_attrs_dict) + len(mt_dataloader.new_test_attrs_dict): ', len(mt_dataloader.train_attrs_dict) + len(mt_dataloader.new_test_attrs_dict))
    assert len(attr_embeddings) == len(mt_dataloader.train_attrs_dict) + len(mt_dataloader.new_test_attrs_dict)
    return item_embeddings, attr_embeddings

class RecModel(nn.Module):
   
    def __init__(self, args, num_users, num_items, user_embedding, item_embedding, activation = None):
        super(RecModel, self).__init__()

        self.activation = activation

        self.item_linear = nn.Linear(args.embed_dim, args.embed_dim)
        self.user_linear = nn.Linear(args.embed_dim, args.embed_dim)
        torch.nn.init.xavier_uniform(self.item_linear.weight)
        self.item_linear.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform(self.user_linear.weight)
        self.user_linear.bias.data.fill_(0.01)
        
        if user_embedding is None:
            self.user_embedding = nn.Parameter(torch.Tensor(num_users, args.embed_dim))
            nn.init.xavier_uniform_(self.user_embedding)
        else:
            self.user_embedding = nn.Parameter(user_embedding)
            self.user_embedding.requires_grad = False

        if item_embedding is None:
            self.item_embedding = nn.Parameter(torch.Tensor(num_items, args.embed_dim))
            nn.init.xavier_uniform_(self.item_embedding)
        else:
            self.item_embedding = nn.Parameter(item_embedding)
            self.item_embedding.requires_grad = False

    def forward(self, users, items):
        
        items = self.item_linear(self.item_embedding[items])
        users = self.user_linear(self.user_embedding[users])
        
        ratings = torch.mul(users, items).sum(1)
        return ratings

    def load_model(self, path):
        model_dict = torch.load(path)
        self.load_state_dict(model_dict)
    
    # load the pre-trained linear layers
    def load_linear(self, path):
        model_dict = self.state_dict()
        linear_dict = torch.load(path)
        linear_dict = {k:v for k, v in linear_dict.items() if k in ['l1.weight', 'l1.bias', 'l2.weight', 'l2.bias', 'l3.weight', 'l3.bias']}
        model_dict.update(linear_dict)
        self.load_state_dict(model_dict)

class RecTask:
    def __init__(self, args, path_store, train_data, test_data, val_data, prediction_data, \
            num_users, num_items, train_items_end_at, pred_item_starts_at, \
            device, user_embedding = None, item_embedding = None, load_path = None):

        self.num_items = num_items
        self.num_users = num_users
        self.train_items_end_at = train_items_end_at # training item idices: [0, train_items_end_at), i.e., include 0 and exclude train_items_end_at
        self.pred_item_starts_at = pred_item_starts_at # prediction (test) item idices: [pred_item_starts_at, self.num_items), i.e., include pred_item_starts_at and exclude self.num_items
        self.args = args
        self.device = device

        self.val_dic = None
        if train_data is not None:
            self.train_data = train_data
            self.dataloader_train, self.train_dic = self.build_dataloader(args, train_data)
        if test_data is not None:
            self.test_data = test_data
            self.dataloader_test, self.test_dic = self.build_dataloader(args, test_data)
        if val_data is not None:
            self.dataloader_val, self.val_dic = self.build_dataloader(args, val_data)
        if prediction_data is not None:
            self.dataloader_prediction, self.pre_dic = self.build_dataloader(args, prediction_data)

        '''
        if item_embedding is not None:
            item_embedding = F.normalize(item_embedding, p=2.0, dim = 0)
        if user_embedding == None:
            user_embedding = self.get_user_embeddings(item_embedding)

        self.model = RecModel(args, num_users = num_users, num_items = num_items, \
             user_embedding = user_embedding, item_embedding = item_embedding, activation = args.activation).to(device)
        if load_path:
            self.model.load_model(load_path)

        self.opt = torch.optim.Adam(self.model.parameters(), args.lr_rec, weight_decay = args.weight_decay)
        self.early_stop = EarlyStoppingCriterion(patience = args.early_stop, save_path = path_store, delta = args.early_stop_delta)
        if self.args.loss == 'mse':
            self.loss_mse = nn.MSELoss(reduction = 'sum')
        '''

    def build_dataloader(self, args, data):
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

        users = torch.LongTensor(data[:, 0]).to(self.device)
        items = torch.LongTensor(data[:, 1]).to(self.device)
        ratings = torch.Tensor(data[:, 2]).to(self.device)
        data = TensorDataset(users, items, ratings)
        data_loader = DataLoader(data, batch_size = args.batch_size, shuffle = True)

        return data_loader, dic_user

    def get_user_embeddings(self, item_embedding):
        
        user_embeddings = torch.mean(item_embedding, 0)
        user_embeddings = user_embeddings.repeat(self.num_users, 1)
        train_user_neighbor_dict = {int(user): torch.tensor([int(item) for [item, rating] in ratings]).to(self.device) for user, ratings in self.train_dic.items()}
        train_user_neighbor_dict = {k: v for k, v in sorted(train_user_neighbor_dict.items(), key=lambda item: item[0])}
        #print(len(train_user_neighbor_dict))

        #print(train_user_neighbor_dict[0])
        #test = torch.mean(torch.index_select(item_embedding, 0, torch.tensor(train_user_neighbor_dict[0])), 0)
        #print(test)
        train_user_embedding_dict = {user: torch.mean(torch.index_select(item_embedding, 0, neighbors), 0) for user, neighbors in train_user_neighbor_dict.items()}
        #print('train_user_embedding_dict[0]: ', train_user_embedding_dict[0])
        #print('train_user_embedding_dict[1]: ', train_user_embedding_dict[1])

        users = torch.tensor(list(train_user_neighbor_dict.keys())).to(self.device)
        user_embeddings[users] = torch.stack(list(train_user_embedding_dict.values()), 0)
        #print('user_embeddings[:5]: ', user_embeddings[:5])
        #print('user_embeddings.size(): ', user_embeddings.size())
        return user_embeddings

    def negative_sample_train(self, users):
        train_item_set = set([i for i in range(self.train_items_end_at)])
        negative_items = []
        for user in users:
            user = int(user)
            # corner case when user only exists in test triples
            if user in self.train_dic:
                historical_items = set(self.train_dic[user][:, 0].astype(int))
                sampled_item = random.sample(train_item_set - historical_items, k = 1)[0]
            else:
                sampled_item = random.sample(train_item_set, k = 1)[0]
            negative_items.append(sampled_item)
        return negative_items

    def train(self, args):
        logging.info("Fine-tuning to get the users' embeddings for the downstream recommendation task.")

        for epoch in range(args.epochs):
            self.model.train()
            batch_count = 0

            epoch_train_loss = 0
            epoch_num_samples = 0
            for users, items, ratings in self.dataloader_train:
                if batch_count % 10000 == 0:
                    logging.info("training epoch:%d batch:%d/%d" % (epoch, batch_count, len(self.dataloader_train)))
                #print('possitive pairs:')
                ratings_pred = self.model(users, items)
                #print()

                if self.args.loss == 'mse':
                    loss = self.loss_mse(ratings_pred, ratings)
                elif self.args.loss == 'bpr':
                    items_neg = self.negative_sample_train(users)
                    items_neg = torch.tensor(items_neg, dtype = torch.long).to(self.device)
                    
                    #print('users: ', users)
                    #print('items: ', items)
                    #print('items_neg: ', items_neg)
                    #print('negative pairs:')
                    ratings_pred_neg = self.model(users, items_neg)
                    
                    loss = -(ratings_pred - ratings_pred_neg).sigmoid().log().sum()
                    print('Epoch ', str(epoch), ' batch ', str(batch_count))
                    print('ratings_pred: ', ratings_pred)
                    print('ratings_pred_neg: ', ratings_pred_neg)
                    print('\n\n')

                print('Training epoch ', str(epoch), ' batch ', str(batch_count))
                print('\t', self.args.loss, ' loss: ', loss.item()/len(users))
                epoch_train_loss += loss.item()
                epoch_num_samples += len(users)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                batch_count += 1

            loss_valid = self.test()
            print('*** Training epoch ', str(epoch), ' average loss ', epoch_train_loss/epoch_num_samples)
            print('*** Test', self.args.loss, ' loss: ', loss_valid)

            logging.info("---------- Validate after epoch %d ----------" % (epoch + 1))
            if self.args.loss == 'mse':
                logging.info("mse loss_valid: %.6f" % (loss_valid))
            elif self.args.loss == 'bpr':
                logging.info("bpr loss_valid: %.6f" % (loss_valid))

            self.early_stop(loss_valid, self.model)
            if self.early_stop.early_stop:
                logging.info("Early stopping. Epochs:%d early_stop_loss:%.6f" % (epoch + 1, self.early_stop.best_loss))
                break

    def test(self):
        loss_valid = 0
        user_count = 0
        self.model.eval()
        for users, items, ratings in self.dataloader_test:
            user_count += len(users)
            ratings_pred = self.model(users, items)

            if self.args.loss == 'mse':
                loss = self.loss_mse(ratings_pred, ratings)
            elif self.args.loss == 'bpr':

                items_neg = self.negative_sample_train(users)
                items_neg = torch.tensor(items_neg, dtype = torch.long).to(self.device)

                ratings_pred_neg = self.model(users, items_neg)
                loss = -(ratings_pred - ratings_pred_neg).sigmoid().log().sum()

            loss_valid += loss.item()
        return loss_valid / user_count

    '''
    Calculate the prediction mae, mse, rmse, ndcg scores.
    '''
    def predict(self):
        
        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        print('Inside rec_task predict')
        # # item_embedding = self.model.item_embedding.cpu().detach().numpy()
        # # print('item_embedding[:100]: ', item_embedding[:100])
        # # print('item_embedding[-100:]: ', item_embedding[-100:])
        # # print()
        # user_embedding = self.model.user_embedding.cpu().detach().numpy()
        # print('user_embedding[:100]: ', user_embedding[:100])
        # print('user_embedding[-100:]: ', user_embedding[-100:])
        # #exit()
        
        print('self.model.item_linear.weight: ', self.model.item_linear.weight.detach().clone())
        print('self.model.user_linear.weight: ', self.model.user_linear.weight.detach().clone())


        if len(self.args.k_list) > 0:
            ndcg, recall = self.calculate_ndcg_recall()
            for key in ndcg:
                logging.info("ndcg@%i: %.6f" % (key, ndcg[key]))
            for key in recall:
                logging.info("recall@%i: %.6f" % (key, recall[key]))

        if self.args.loss == 'bpr':
            return ndcg, recall

        elif self.args.loss == 'mse':
            all_prediction = []
            all_truth = []

            self.model.eval()
            for users, items, ratings in self.dataloader_prediction:
                ratings_pred = self.model(users, items)
                all_prediction += ratings_pred.tolist()
                all_truth += ratings.tolist()
            
            count = len(all_prediction)
            all_prediction = torch.tensor(all_prediction)
            all_truth = torch.tensor(all_truth)
            mse = self.loss_mse(all_prediction, all_truth).item() / count
            rmse = np.sqrt(mse)
            mae_fcn = nn.L1Loss()
            mae = mae_fcn(all_prediction, all_truth).item()

            logging.info("Prediction results:\n\tmae:%.6f, \n\tmse:%.6f, \n\trmse:%.6f \n" % (mae, mse, rmse))

            if len(self.args.k_list) > 0:
                return mae, mse, rmse, ndcg, recall
            else:
                return mae, mse, rmse
            

    '''
    Calculate NDCG, Recall
    '''
    def calculate_ndcg_recall(self):
        result_dic_ndcg = {}
        result_dic_recall = {}

        for k in self.args.k_list:
            result_dic_ndcg[k] = 0
            result_dic_recall[k] = 0

        user_count = 0

        for user in tqdm(self.pre_dic):
            if self.args.rank_new:
                # calculate items only in prediction set
                users = torch.tensor([user] * (self.num_items - self.pred_item_starts_at), dtype = torch.long).to(self.device)
                items = torch.tensor(list(range(self.pred_item_starts_at, self.num_items))).to(self.device)
                gt = torch.zeros(self.num_items - self.pred_item_starts_at)
                gt[self.pre_dic[user][:, 0] - self.pred_item_starts_at] = torch.tensor(self.pre_dic[user][:, 1], dtype = gt.dtype)

            else:
                users = torch.tensor([user] * self.num_items, dtype = torch.long).to(self.device)
                items = torch.tensor(list(range(self.num_items))).to(self.device)
                gt = torch.zeros(self.num_items)
                gt[self.pre_dic[user][:, 0]] = torch.tensor(self.pre_dic[user][:, 1], dtype = gt.dtype)

            ratings_pred = self.model(users, items)

            if not self.args.rank_new:
                if user in self.train_dic:
                    ratings_pred[self.train_dic[user][:, 0].astype(int)] = -999.9
                if user in self.test_dic:
                    ratings_pred[self.test_dic[user][:, 0].astype(int)] = -999.9
                if self.val_dic and user in self.val_dic:
                    ratings_pred[self.val_dic[user][:, 0].astype(int)] = -999.9

            values, indices = torch.topk(ratings_pred, k = max(self.args.k_list))
            gt_values, gt_indices = torch.topk(gt, k = max(self.args.k_list))
            '''
            print('user: ', user)
            print('\tpd_values: ', ratings_pred.tolist())
            print('\tpd_values_sorted_k: ', values)
            print('\tgt_values_sorted_k: ', gt_values)
            '''

            for k in self.args.k_list:

                result = gt[indices]

                idcg = self.dcg(gt_values[:k])
                dcg = self.dcg(result[:k])
                ndcg = dcg / idcg
                recall = (result[:k] != 0).sum() / (gt != 0).sum()

                result_dic_ndcg[k] += ndcg
                result_dic_recall[k] += recall

            user_count += 1
        for key in result_dic_ndcg:
            result_dic_ndcg[key] = result_dic_ndcg[key] / user_count
            result_dic_recall[key] = result_dic_recall[key] / user_count

        return result_dic_ndcg, result_dic_recall

    '''
    calculate dcg
    '''
    def dcg(self, result):
        discount = torch.tensor(range(len(result))).to(result.device) + 2
        discount = discount.log2()
        return (result / discount).sum()

    def train_alignment(self, new_item_embedding):
        train_user_neighbor_dict = {int(user): [int(item) for [item, rating] in ratings] for user, ratings in self.train_dic.items()}
        train_user_neighbor_dict = {k: v for k, v in sorted(train_user_neighbor_dict.items(), key=lambda item: item[0])}

        train_user_test_neighbor_dict = {int(user): [int(item) for [item, rating] in ratings] for user, ratings in self.test_dic.items()}
        train_user_test_neighbor_dict = {k: v for k, v in sorted(train_user_test_neighbor_dict.items(), key=lambda item: item[0])}

        cos = torch.nn.CosineSimilarity(dim=1)
        count, diff, ave_pos, ave_neg = 0, 0, 0, 0
        for user in train_user_neighbor_dict.keys():
        #for user in [0, 1]:
            pos_items = train_user_neighbor_dict[user]
            neg_items = list(set([i for i in range(self.num_items)]) - set(pos_items))
            if user in train_user_test_neighbor_dict.keys():
                neg_items = list(set(neg_items) - set(train_user_test_neighbor_dict[user]))
            #print('pos_items: ', pos_items)
            #print('neg_items: ', neg_items)
            if new_item_embedding is None:
                pos_sims = cos(self.model.user_embedding[[user] * len(pos_items)], self.model.item_embedding[pos_items])
            else:
                pos_sims = cos(self.model.user_embedding[[user] * len(pos_items)], new_item_embedding[pos_items])
            #print('pos_sims: ', pos_sims)
            if new_item_embedding is None:
                neg_sims = cos(self.model.user_embedding[[user] * len(neg_items)], self.model.item_embedding[neg_items])
            else:
                neg_sims = cos(self.model.user_embedding[[user] * len(neg_items)], new_item_embedding[neg_items])

            ave_pos_sims = torch.mean(pos_sims)
            ave_neg_sims = torch.mean(neg_sims)
            if ave_pos_sims > ave_neg_sims:
                count += 1
            diff += (ave_pos_sims - ave_neg_sims)
            ave_pos += ave_pos_sims
            ave_neg += ave_neg_sims

        percent_pos_greater_than_neg = count/len(train_user_neighbor_dict)
        ave_pos_minus_neg = (diff/len(train_user_neighbor_dict)).item()
        ave_pos /= len(train_user_neighbor_dict)
        ave_pos = ave_pos.item()
        ave_neg /= len(train_user_neighbor_dict)
        ave_neg = ave_neg.item()

        return percent_pos_greater_than_neg, ave_pos_minus_neg, ave_pos, ave_neg

    def test_alignment(self):
        train_user_neighbor_dict = {int(user): [int(item) for [item, rating] in ratings] for user, ratings in self.train_dic.items()}
        train_user_neighbor_dict = {k: v for k, v in sorted(train_user_neighbor_dict.items(), key=lambda item: item[0])}

        train_user_test_neighbor_dict = {int(user): [int(item) for [item, rating] in ratings] for user, ratings in self.test_dic.items()}
        train_user_test_neighbor_dict = {k: v for k, v in sorted(train_user_test_neighbor_dict.items(), key=lambda item: item[0])}

        cos = torch.nn.CosineSimilarity(dim=1)
        count, diff, ave_pos, ave_neg = 0, 0, 0, 0
        for user in train_user_test_neighbor_dict.keys():
        #for user in [0, 1]:
            pos_items = train_user_test_neighbor_dict[user]
            neg_items = list(set([i for i in range(self.num_items)]) - set(pos_items))
            if user in train_user_neighbor_dict.keys():
                neg_items = list(set(neg_items) - set(train_user_neighbor_dict[user]))
            #print('pos_items: ', pos_items)
            #print('neg_items: ', neg_items)
            pos_sims = cos(self.model.user_embedding[[user] * len(pos_items)], self.model.item_embedding[pos_items])
            #print('pos_sims: ', pos_sims)
            neg_sims = cos(self.model.user_embedding[[user] * len(neg_items)], self.model.item_embedding[neg_items])

            ave_pos_sims = torch.mean(pos_sims)
            ave_neg_sims = torch.mean(neg_sims)
            if ave_pos_sims > ave_neg_sims:
                count += 1
            diff += (ave_pos_sims - ave_neg_sims)
            ave_pos += ave_pos_sims
            ave_neg += ave_neg_sims

        percent_pos_greater_than_neg = count/len(train_user_test_neighbor_dict)
        ave_pos_minus_neg = (diff/len(train_user_test_neighbor_dict)).item()
        ave_pos /= len(train_user_test_neighbor_dict)
        ave_pos = ave_pos.item()
        ave_neg /= len(train_user_test_neighbor_dict)
        ave_neg = ave_neg.item()

        return percent_pos_greater_than_neg, ave_pos_minus_neg, ave_pos, ave_neg

    def pred_alignment(self, new_item_embedding):
        pred_user_neighbor_dict = {int(user): [int(item) for [item, rating] in ratings] for user, ratings in self.pre_dic.items()}
        pred_user_neighbor_dict = {k: v for k, v in sorted(pred_user_neighbor_dict.items(), key=lambda item: item[0])}
        # for user in list(pred_user_neighbor_dict.keys())[:5]:
        #     print(user, pred_user_neighbor_dict[user])
        #     print()

        cos = torch.nn.CosineSimilarity(dim=1)
        count, diff, ave_pos, ave_neg = 0, 0, 0, 0
        for user in pred_user_neighbor_dict.keys():
        #for user in [0, 1]:
            pos_items = pred_user_neighbor_dict[user]
            neg_items = list(set([i+self.num_items for i in range(len(new_item_embedding) - self.num_items)]) - set(pos_items))
            #print('pos_items: ', pos_items)
            #print('neg_items: ', neg_items)
            pos_sims = cos(self.model.user_embedding[[user] * len(pos_items)], new_item_embedding[pos_items])
            #print('pos_sims: ', pos_sims)
            neg_sims = cos(self.model.user_embedding[[user] * len(neg_items)], new_item_embedding[neg_items])

            ave_pos_sims = torch.mean(pos_sims)
            ave_neg_sims = torch.mean(neg_sims)
            if ave_pos_sims > ave_neg_sims:
                count += 1
            diff += (ave_pos_sims - ave_neg_sims)
            ave_pos += ave_pos_sims
            ave_neg += ave_neg_sims

        percent_pos_greater_than_neg = count/len(pred_user_neighbor_dict)
        ave_pos_minus_neg = (diff/len(pred_user_neighbor_dict)).item()
        ave_pos /= len(pred_user_neighbor_dict)
        ave_pos = ave_pos.item()
        ave_neg /= len(pred_user_neighbor_dict)
        ave_neg = ave_neg.item()

        return percent_pos_greater_than_neg, ave_pos_minus_neg, ave_pos, ave_neg

    def pred_ranking(self, new_item_embedding, rank_new = True):

        result_dic_ndcg = {}
        result_dic_recall = {}
        for k in self.args.k_list:
            result_dic_ndcg[k] = 0
            result_dic_recall[k] = 0
        
        for user in self.pre_dic.keys():
        #for user in [0, 1]:
            if rank_new:
                gt = torch.zeros(len(new_item_embedding) - self.num_items)
                gt[self.pre_dic[user][:, 0] - self.num_items] = torch.tensor(self.pre_dic[user][:, 1], dtype = gt.dtype)
                #print('self.pre_dic[user]: ', self.pre_dic[user])
                #print('gt: ', gt)

                items = [i+self.num_items for i in range(len(new_item_embedding) - self.num_items)]
                #print('items: ', items)

            else:
                gt = torch.zeros(len(new_item_embedding))
                gt[self.pre_dic[user][:, 0]] = torch.tensor(self.pre_dic[user][:, 1], dtype = gt.dtype)

                items = [i for i in range(len(new_item_embedding))]

            users = [user] * len(items)
            users = self.model.user_embedding[users]
            #print('users: ', users)
            items = new_item_embedding[items]
            #print('items: ', items)

            ratings_pred = torch.mul(users, items).sum(1)
            #print('ratings_pred: ', ratings_pred)
            #print()

            if not rank_new:
                if user in self.train_dic:
                    ratings_pred[self.train_dic[user][:, 0].astype(int)] = -999.9
                if user in self.test_dic:
                    ratings_pred[self.test_dic[user][:, 0].astype(int)] = -999.9

            values, indices = torch.topk(ratings_pred, k = max(self.args.k_list))
            gt_values, gt_indices = torch.topk(gt, k = max(self.args.k_list))

            for k in self.args.k_list:

                result = gt[indices]

                idcg = self.dcg(gt_values[:k])
                dcg = self.dcg(result[:k])
                ndcg = dcg / idcg
                recall = (result[:k] != 0).sum() / (gt != 0).sum()

                result_dic_ndcg[k] += ndcg
                result_dic_recall[k] += recall


        for key in result_dic_ndcg:
            result_dic_ndcg[key] = result_dic_ndcg[key] / len(self.pre_dic)
            result_dic_recall[key] = result_dic_recall[key] / len(self.pre_dic)

        return result_dic_ndcg, result_dic_recall

    def pred_ranking_user_embeddings(self, new_item_embedding):
        with open(user_embeddings_store_path, 'rb') as fp:
            user_embeddings = pickle.load(fp)
        user_embeddings = user_embeddings.to(self.device)

        result_dic_ndcg = {}
        result_dic_recall = {}
        for k in self.args.k_list:
            result_dic_ndcg[k] = 0
            result_dic_recall[k] = 0
        
        for user in self.pre_dic.keys():
            gt = torch.zeros(len(new_item_embedding) - self.num_items)
            gt[self.pre_dic[user][:, 0] - self.num_items] = torch.tensor(self.pre_dic[user][:, 1], dtype = gt.dtype)

            items = [i+self.num_items for i in range(len(new_item_embedding) - self.num_items)]
            
            #print('items: ', items)
            users = [user] * len(items)
            users = user_embeddings[users]
            #print('users: ', users)
            items = new_item_embedding[items]
            #print('items: ', items)

            ratings_pred = torch.mul(users, items).sum(1)
            #print('ratings_pred: ', ratings_pred)
            #print()

            values, indices = torch.topk(ratings_pred, k = max(self.args.k_list))
            gt_values, gt_indices = torch.topk(gt, k = max(self.args.k_list))

            for k in self.args.k_list:

                result = gt[indices]

                idcg = self.dcg(gt_values[:k])
                dcg = self.dcg(result[:k])
                ndcg = dcg / idcg
                recall = (result[:k] != 0).sum() / (gt != 0).sum()

                result_dic_ndcg[k] += ndcg
                result_dic_recall[k] += recall


        for key in result_dic_ndcg:
            result_dic_ndcg[key] = result_dic_ndcg[key] / len(self.pre_dic)
            result_dic_recall[key] = result_dic_recall[key] / len(self.pre_dic)

        return result_dic_ndcg, result_dic_recall

    def pred_ranking_new(self, new_item_embedding, rank_new = True):
        
        user_embeddings = torch.mean(new_item_embedding[:self.num_items], 0)
        user_embeddings = user_embeddings.repeat(self.num_users, 1)
        #train_test_user_neighbor_dict = {int(user): torch.tensor([int(item) for [item, rating] in ratings]).to(self.device) for user, ratings in {**self.train_dic, **self.test_dic}.items()}
        #train_test_user_neighbor_dict = {int(user): torch.tensor([int(item) for [item, rating] in ratings]).to(self.device) for user, ratings in self.train_dic.items()}
        _, train_test_dic = self.build_dataloader(args, np.concatenate((self.train_data, self.test_data), axis=0))
        train_test_user_neighbor_dict = {int(user): torch.tensor([int(item) for [item, rating] in ratings]).to(self.device) for user, ratings in train_test_dic.items()}
        train_test_user_neighbor_dict = {k: v for k, v in sorted(train_test_user_neighbor_dict.items(), key=lambda item: item[0])}
        #print(len(train_user_neighbor_dict))

        #print(train_user_neighbor_dict[0])
        #test = torch.mean(torch.index_select(item_embedding, 0, torch.tensor(train_user_neighbor_dict[0])), 0)
        #print(test)
        train_test_user_embedding_dict = {user: torch.mean(torch.index_select(new_item_embedding, 0, neighbors), 0) for user, neighbors in train_test_user_neighbor_dict.items()}
        #print('train_user_embedding_dict[0]: ', train_user_embedding_dict[0])
        #print('train_user_embedding_dict[1]: ', train_user_embedding_dict[1])

        users = torch.tensor(list(train_test_user_embedding_dict.keys())).to(self.device)
        user_embeddings[users] = torch.stack(list(train_test_user_embedding_dict.values()), 0)

        result_dic_ndcg = {}
        result_dic_recall = {}
        for k in self.args.k_list:
            result_dic_ndcg[k] = 0
            result_dic_recall[k] = 0
        
        for user in self.pre_dic.keys():
        #for user in [0, 1]:
            if rank_new:
                gt = torch.zeros(len(new_item_embedding) - self.num_items)
                gt[self.pre_dic[user][:, 0] - self.num_items] = torch.tensor(self.pre_dic[user][:, 1], dtype = gt.dtype)
                #print('self.pre_dic[user]: ', self.pre_dic[user])
                #print('gt: ', gt)

                items = [i+self.num_items for i in range(len(new_item_embedding) - self.num_items)]
            else:
                gt = torch.zeros(len(new_item_embedding))
                gt[self.pre_dic[user][:, 0]] = torch.tensor(self.pre_dic[user][:, 1], dtype = gt.dtype)

                items = [i for i in range(len(new_item_embedding))]

            #print('items: ', items)
            users = [user] * len(items)
            users = user_embeddings[users]
            #print('users: ', users)
            items = new_item_embedding[items]
            #print('items: ', items)

            ratings_pred = torch.mul(users, items).sum(1)
            #print('ratings_pred: ', ratings_pred)
            #print()

            if not rank_new:
                if user in self.train_dic:
                    ratings_pred[self.train_dic[user][:, 0].astype(int)] = -999.9
                if user in self.test_dic:
                    ratings_pred[self.test_dic[user][:, 0].astype(int)] = -999.9

            values, indices = torch.topk(ratings_pred, k = max(self.args.k_list))
            gt_values, gt_indices = torch.topk(gt, k = max(self.args.k_list))

            for k in self.args.k_list:

                result = gt[indices]

                idcg = self.dcg(gt_values[:k])
                dcg = self.dcg(result[:k])
                ndcg = dcg / idcg
                recall = (result[:k] != 0).sum() / (gt != 0).sum()

                result_dic_ndcg[k] += ndcg
                result_dic_recall[k] += recall


        for key in result_dic_ndcg:
            result_dic_ndcg[key] = result_dic_ndcg[key] / len(self.pre_dic)
            result_dic_recall[key] = result_dic_recall[key] / len(self.pre_dic)

        return result_dic_ndcg, result_dic_recall

if __name__ == "__main__":
    
    args = parse_args()
    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + str(args.gpu))

    mt_dataloader = MultiTaskDataLoader(t1 = True, t2 = False, t3 = False, t2_augment = None, n = 520)
    train_triples, test_triples, val_triples, pred_triples = mt_dataloader.get_rec_data()
    pred_triples[:, 1] -= mt_dataloader.num_val_items # map the triples to bypass the validation items
    num_train_items = mt_dataloader.num_train_items
    num_train_attrs = len(mt_dataloader.train_attrs_dict)

    Recommendation_task = RecTask(args = args, path_store = args.rec_path_store, \
        train_data = train_triples, test_data = test_triples, val_data = None, prediction_data = pred_triples,\
        num_users = mt_dataloader.num_user, num_items = mt_dataloader.num_train_items, \
        train_items_end_at = mt_dataloader.num_train_items, pred_item_starts_at = None, \
        device = device, user_embedding= None, item_embedding = None, load_path = None)

    '''
    bert_item_embedding = torch.cat((mt_dataloader.train_items_embedding, mt_dataloader.test_items_embedding), 0).to(device)
    result_dic_ndcg, result_dic_recall = Recommendation_task.pred_ranking_new(bert_item_embedding, rank_new = True)
    print('result_dic_ndcg: ', result_dic_ndcg)
    print('result_dic_recall: ', result_dic_recall)
    '''

    #model_path = '/home/anonymous/Attribute_cold_start_amazon_home/models/model/hete_model_t1.pkl'
    #new_store_path = '/home/anonymous/Attribute_cold_start_amazon_home/models/model/embeddings_t1.pkl'
    #model_path = '/home/anonymous/Attribute_cold_start_amazon_home/models/model/hete_model_t1_t2.pkl'
    #new_store_path = '/home/anonymous/Attribute_cold_start_amazon_home/models/model/embeddings_t1_t2.pkl'

    #model_path = '/home/anonymous/Attribute_cold_start_amazon_sports/models/model/hete_model_t1.pkl'
    #new_store_path = '/home/anonymous/Attribute_cold_start_amazon_sports/models/model/embeddings_t1.pkl'
    #new_store_path = '/home/anonymous/Attribute_cold_start_amazon_sports/models/model/embeddings_t1_510.pkl'
    #new_store_path = '/home/anonymous/Attribute_cold_start_amazon_sports/models/model/embeddings_t1_520.pkl'

    model_path = '/home/anonymous/Attribute_cold_start_amazon_sports/models/model/hete_model_t1_t2.pkl'
    #new_store_path = '/home/anonymous/Attribute_cold_start_amazon_sports/models/model/embeddings_t1_t2.pkl'
    #new_store_path = '/home/anonymous/Attribute_cold_start_amazon_sports/models/model/embeddings_t1_t2_510.pkl'
    new_store_path = '/home/anonymous/Attribute_cold_start_amazon_sports/models/model/embeddings_t1_t2_520.pkl'

    load_new_embeddings()
    with open(new_store_path, 'rb') as fp:
        new_embeddings = pickle.load(fp)
    new_item_embedding = new_embeddings[:(mt_dataloader.num_train_items + mt_dataloader.num_test_items)]
    new_item_embedding = F.normalize(new_item_embedding, p=2.0, dim = 0)
    new_item_embedding = new_item_embedding.to(device)

    result_dic_ndcg, result_dic_recall = Recommendation_task.pred_ranking_new(new_item_embedding, rank_new = True)
    print('result_dic_ndcg: ', result_dic_ndcg)
    print('result_dic_recall: ', result_dic_recall)
    

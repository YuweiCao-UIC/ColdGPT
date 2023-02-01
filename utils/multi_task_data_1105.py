from torch.utils.data import TensorDataset, Dataset, BatchSampler
import random
import torch

class NodePairDataset(Dataset):
    '''
    Dataset for task 1: self-supervised learning on attribute-item graph
    node_pair_list: [(item_id, attr_id), ...] e.g.: [(10, 19), ...]
    '''
    def __init__(self, node_pair_list):
        self.node_pair_list = node_pair_list

    def __len__(self):
        return len(self.node_pair_list)
    
    def __getitem__(self, index):
        #return self.node_pair_list[index]
        return torch.LongTensor(self.node_pair_list[index])

#task3_dataset = ReviewDataset(mt_dataloader.task3_review_sentences, mt_dataloader.task3_review_item_ids, mt_dataloader.task3_review_pretrain_indices)

class ReviewDataset(Dataset):
    '''
    Dataset for task 3: predict items using reviews
    reviews: a list of the raw text of the review sentences, or a tensor of the BERT embeddings of the review sentences
    items: a list of the item_ids that correspond to the review sentences
    '''
    def __init__(self, reviews, items):
        self.reviews = reviews
        self.items = items
        #print('Inside ReviewDataset __init__')
        #print(self.reviews)
        #print(self.items)
        #print(set(self.items))
        #print(len(list(set(self.items))))
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, index):
        return (self.reviews[index], self.items[index])
        #print(index)
        #print(self.reviews[index])
        #print(self.items[index])
        #print(torch.LongTensor(self.items[index]))
        #return (self.reviews[index], torch.LongTensor(self.items[index]))

'''
class ReviewDataset(Dataset):
    
    #Dataset for task 3: predict items using reviews
    #review_item_list: [[review sentence, item_id], ...] e.g.: [['The burger is delicious', 16], ...]
    
    def __init__(self, review_item_list):
        self.reviews = [i[0] for i in review_item_list]
        self.items = [i[1] for i in review_item_list]
        #print('Inside ReviewDataset __init__')
        #print(self.reviews)
        #print(self.items)
        #print(set(self.items))
        #print(len(list(set(self.items))))
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, index):
        return (self.reviews[index], self.items[index])
        #print(index)
        #print(self.reviews[index])
        #print(self.items[index])
        #print(torch.LongTensor(self.items[index]))
        #return (self.reviews[index], torch.LongTensor(self.items[index]))
'''

class BERT4RecDataset(Dataset):
    '''
    Dataset for task 2: predict masked items in users' purchase history
    '''
    def __init__(self, args, u2seq, num_items):
        self.u2seq = u2seq
        #self.users = sorted(self.u2seq.keys())
        self.max_len = args.BERT4Rec_max_len
        self.mask_prob = args.BERT4Rec_mask_prob
        self.mask_token = num_items + 1
        self.num_items = num_items
    
    def __len__(self):
        #return len(self.users)
        return len(self.u2seq)
    
    def __getitem__(self, index):
        #user = self.users[index]
        #seq = self._getseq(user)
        seq = self._getseq(index)

        tokens = []
        labels = []

        prob_mask_last = torch.rand(1)
        if prob_mask_last < 0.1:
            tokens = seq[:-1]
            tokens.append(self.mask_token)
            labels = [0] * len(seq[:-1])
            labels.append(seq[-1])
        else:
            for s in seq:
                prob = torch.rand(1)
                if prob < self.mask_prob:
                    #prob /= self.mask_prob
                    #if prob < 0.8:
                    #    tokens.append(self.mask_token)
                    #elif prob < 0.9:
                    #    tokens.append(torch.randint(1, self.num_items + 1, (1,)).item())
                    #else:
                    #    tokens.append(s)
                    tokens.append(self.mask_token)
                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)

        #tokens = tokens[-self.max_len:]
        #labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return (torch.LongTensor(tokens), torch.LongTensor(labels))

    def _getseq(self, user):
        #return self.u2seq[user]
        return [each + 1 for each in self.u2seq[user]] # add one to each item in the original sequence (make item encoding starts from 1)

class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        task_2_dataset_dic = {}
        task_types = [1, 2, 3]
        for each in zip(task_types, datasets):
            task_type = each[0]
            task_2_dataset_dic[task_type] = each[1]
        self._task_2_dataset_dic = task_2_dataset_dic
    
    def __len__(self):
        return sum(len(dataset) if dataset is not None else 0 for dataset in self._datasets)
    
    def __getitem__(self, idx):
        #print('in MultiTaskDataset __getitem__')
        #print(idx)
        task_type, sample_id = idx
        #if task_type == 3:
        #    print(self._task_2_dataset_dic[task_type][sample_id])
        return task_type, self._task_2_dataset_dic[task_type][sample_id]

class MultiTaskBatchSampler(BatchSampler):
    def __init__(self, datasets, args):
        self._datasets = datasets
        self._batch_size = args.pretrain_batch_size
        train_data_list = []
        for dataset in datasets:
            if dataset is not None:
                train_data_list.append(self._get_shuffled_index_batches(len(dataset), self._batch_size))
            else:
                train_data_list.append([])
        self._train_data_list = train_data_list
    
    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [list(range(i, min(i+batch_size, dataset_len))) for i in range(0, dataset_len, batch_size)]
        random.shuffle(index_batches)
        return index_batches
    
    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)
    
    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(self._train_data_list)
        for local_task_idx in all_indices:
            #task_type = self._datasets[local_task_idx][0][3].item()
            task_type = local_task_idx + 1
            batch = next(all_iters[local_task_idx])
            yield [(task_type, sample_id) for sample_id in batch]
    
    @staticmethod
    def _gen_task_indices(train_data_list):
        all_indices = []
        for i in range(0, len(train_data_list)):
            all_indices += [i] * len(train_data_list[i])
        random.shuffle(all_indices)
        return all_indices

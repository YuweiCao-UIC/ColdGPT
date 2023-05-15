from torch.utils.data import TensorDataset, Dataset, BatchSampler
import random
import torch

class NodePairDataset(Dataset):
    def __init__(self, node_pair_list):
        self.node_pair_list = node_pair_list

    def __len__(self):
        return len(self.node_pair_list)
    
    def __getitem__(self, index):
        return torch.LongTensor(self.node_pair_list[index])


class BERT4RecDataset(Dataset):
    def __init__(self, args, u2seq, num_items):
        self.u2seq = u2seq
        self.max_len = args.BERT4Rec_max_len
        self.mask_prob = args.BERT4Rec_mask_prob
        self.mask_token = num_items + 1
        self.num_items = num_items
    
    def __len__(self):
        return len(self.u2seq)
    
    def get_pred_seq(self, task2_pred_dic):
        pred_seq, indices = [], []
        for user, seq in task2_pred_dic.items():
            mapped_seq = [each + 1 for each in seq]
            mapped_seq.append(self.mask_token)
            mask_len = self.max_len - len(mapped_seq)
            mapped_seq = [0] * mask_len + mapped_seq
            pred_seq.append(mapped_seq)
            indices.append([0] * (self.max_len - 1) + [1])
        return torch.LongTensor(pred_seq), torch.LongTensor(indices)
        
    def __getitem__(self, index):
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
                    tokens.append(self.mask_token)
                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)

        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return (torch.LongTensor(tokens), torch.LongTensor(labels))

    def _getseq(self, user):
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
        task_type, sample_id = idx
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


if __name__ == "__main__":
    from parser import parse_args
    args = parse_args()
    u2seq = [[261, 6, 783, 1187, 87, 444, 806, 1397, 2145, 148, 152, 1170, 1469, 398, 820, 1136, 181, 177, 3, 726, 1527, 282, 5], \
        [1150, 13, 295, 772, 37, 296, 30, 2081, 638, 124, 414, 797, 806, 28, 367, 117, 443, 590, 226, 58], \
        [760, 943, 6, 654, 1657, 1316, 1135, 62, 733, 35, 910, 97, 477, 1375, 995, 796, 1666, 1315, 924, 606], \
        [321, 12, 991, 2125, 1673, 286, 36, 699, 95, 401, 404]]
    num_items = 2258
    t2_dataset = BERT4RecDataset(args, u2seq, num_items)
    x, y = t2_dataset.__getitem__(3)
    print('x: ', x)
    print('y: ', y)

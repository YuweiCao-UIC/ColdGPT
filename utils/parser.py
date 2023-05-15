import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_batch_size', type=int, default=128, help='batch size of the pretraining task')
    parser.add_argument('--embed_dim', type = int, default = 64, help='node embedding dim')
    parser.add_argument('--lr', type = float, default = 0.005, help = 'learning rate for pretraining')
    parser.add_argument('--weight_decay', type = float, default = 0.01, help = 'weight_decay for pretraining')
    parser.add_argument('--epochs', type = int, default = 999, help = 'max epochs')
    parser.add_argument('--early_stop', type = int, default = 5, help = 'early stop patience')
    parser.add_argument('--early_stop_delta', type = float, default = 1e-3, help = 'early stop delta')
    parser.add_argument('--gpu', type = int, default = 0, help = 'gpu index. -1 stands for cpu')
    parser.add_argument('--seed', type = int, default = 4, help = 'random seed')
    parser.add_argument('--pretrain_loss', type = str, default = 'directau', help = 'infonce or directau')
    parser.add_argument('--plm', type = str, default = 'SBERT', help = 'BERT or SBERT')

    # Hete graph (HeteModel task 1) args
    parser.add_argument('--edge_dropout', type = float, default = 0.3, help = 'edge dropout for cl')
    
    # BERT4Rec (HeteModel task 2) args
    parser.add_argument('--BERT4Rec_max_len', type=int, default=100, help='the max length of the users\' purchase sequences')
    parser.add_argument('--BERT4Rec_n_layers', type=int, default=1, help='the number of transformer layers in the BERT4Rec module')
    parser.add_argument('--BERT4Rec_n_heads', type=int, default=1, help='the number of heads in each transformer layer of the BERT4Rec module')
    parser.add_argument('--BERT4Rec_mask_prob', type = float, default = 0.2, help = 'the probability for each token (item) to be masked when pretraining using the BERT4Rec module')
    parser.add_argument('--BERT4Rec_dropout', type = float, default = 0.3, help = 'dropout rate of the BERT4Rec module')

    # hete pretrain args
    parser.add_argument('--data_path', type = str, default = './data/yelp/', help = 'The path to load yelp dataset')
    parser.add_argument('--log_path', type = str, default = './log/log', help = 'The path to save the training log')
    parser.add_argument('--model_load_path', type = str, default = '', help = 'The path to load pre-trained model')
    parser.add_argument('--model_save_path', type = str, default = './model/checkpoint.pkl', help = 'The path to save pre-trained hete model')
    parser.add_argument('--lambda_reg', type = float, default = 1, help = 'The weight of the regularization term')
    parser.add_argument('--t1', default = False, action="store_true", help = 'If true, pretrain on task 1 (self-supervised learning on attribute-item graph)')
    parser.add_argument('--t2', default = False, action="store_true", help = 'If true, pretrain on task 2 (predict masked items in users\' purchase history)')
    parser.add_argument('--t3', default = False, action="store_true", help = 'If true, pretrain on task 3 (predict items using reviews)')
    parser.add_argument('--w_t1', type = float, default = 1, help = 'The weight of task 1')
    parser.add_argument('--w_t2', type = float, default = 1, help = 'The weight of task 2')
    parser.add_argument('--w_t3', type = float, default = 1, help = 'The weight of task 3')

    # downstream recommendation args
    parser.add_argument('--lr_rec', type = float, default = 0.001, help = 'Learning rate for recommendation task')
    parser.add_argument('--loss', type = str, default = 'mse', help = 'mse or bpr')
    parser.add_argument('--activation', type = str, default = 'sigmoid', help = "'clamp', 'sigmoid', or 'None'")
    parser.add_argument('--rank_new', action = 'store_true')
    parser.add_argument('--k_list', type = list, default = [5, 20, 40], help = 'k list for computing ndcg')

    return parser.parse_args()

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=500, help='test set')
    parser.add_argument('--pretrain_batch_size', type=int, default=128, help='batch size of the pretraining task')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of downstream rec task')
    parser.add_argument('--embed_dim', type = int, default = 64, help='node embedding dim')
    parser.add_argument('--lr', type = float, default = 0.005, help = 'learning rate')
    parser.add_argument('--epochs', type = int, default = 999, help = 'max epochs')
    parser.add_argument('--early_stop', type = int, default = 50, help = 'early stop patience')
    parser.add_argument('--early_stop_delta', type = float, default = 1e-3, help = 'early stop delta')
    parser.add_argument('--gpu', type = int, default = 0, help = 'gpu index. -1 stands for cpu')
    parser.add_argument('--seed', type = int, default = 4, help = 'random seed')
    parser.add_argument('--pretrain_loss', type = str, default = 'directau', help = 'infonce or directau')
    parser.add_argument('--encoder_layers', type = int, default = 3, help = 'num of encoder layers')

    # Hete graph (HeteModel task 1) args
    parser.add_argument('--edge_dropout', type = float, default = 0.3, help = 'edge dropout for cl')
    parser.add_argument('--temperature', type = float, default = 0.5, help = 'temperature of contrastive learning')
    parser.add_argument('--gnn', type = str, default = 'gcn', help = 'gcn or gat')
    parser.add_argument('--heads', type = int, default = 1, help = 'attention heads for gat')
    parser.add_argument('--conv_layers', type = int, default = 1, help = '1, 2, or 3, number of GCNConv layers in the heterogeneous graph')
    parser.add_argument('--perturb', default = False, action="store_true", help = 'If true, perturb the embbedings')
    parser.add_argument('--eps', type = float, default = 0.01, help = 'Magnitude of the noise for perturbation')

    # BERT4Rec (HeteModel task 2) args
    parser.add_argument('--t2_augment', type = str, default = None, help = "None or 'also_view_minus_also_buy' or 'corrupt'")
    parser.add_argument('--BERT4Rec_max_len', type=int, default=100, help='the max length of the users\' purchase sequences')
    parser.add_argument('--BERT4Rec_n_layers', type=int, default=2, help='the number of transformer layers in the BERT4Rec module')
    parser.add_argument('--BERT4Rec_n_heads', type=int, default=2, help='the number of heads in each transformer layer of the BERT4Rec module')
    parser.add_argument('--BERT4Rec_mask_prob', type = float, default = 0.2, help = 'the probability for each token (item) to be masked when pretraining using the BERT4Rec module')
    parser.add_argument('--BERT4Rec_dropout', type = float, default = 0.3, help = 'dropout rate of the BERT4Rec module')
   
    # BERT (HeteModel task 3) args
    parser.add_argument('--num_sample', type=int, default=0, help='In the case that there are too many reviews, randomly sample num_sample from them')
    parser.add_argument('--review_len_threshold', type=int, default=0, help='reviews with lengths less than this number will be filtered out')
    parser.add_argument('--bert_linear_layers', type = int, default = 3, help = 'num of bert_linear layers')

    # hete pretrain args
    parser.add_argument('--log_path', type = str, default = './hete_log_t1_t2_woiteme_newdata_mlp2_reg_10', help = 'The path to save the training log')
    parser.add_argument('--train', default = False, action="store_true", help = 'If true, pre-train hete model, else load pre-trained hete model from args.hete_path_store')
    parser.add_argument('--fine_tune', default = False, action="store_true", help = 'If true, fine-tune recommendation model to get users\' embeddings, else load fine-tuned recommendation model')
    parser.add_argument('--validate', default = False, action="store_true", help = 'If true, evaluate on the validation set')
    parser.add_argument('--predict', default = False, action="store_true", help = 'If true, evaluate on the prediction set')
    parser.add_argument('--hete_path_store', type = str, default = './hete_model_t1_t2_woiteme_newdata_reg.pkl', help = 'The path to save/load pre-trained hete model')
    parser.add_argument('--rec_path_store', type = str, default = './rec_model_t1_t2_woiteme_newdata_mlp2_reg_10.pkl', help = 'The path to save/load fine-tuned recommendation model')
    parser.add_argument('--reg', default = False, action="store_true", help = 'If true, add a regularization term to the loss to regularize items and attrs embeddings')
    parser.add_argument('--lambda_reg', type = float, default = 1, help = 'The weight of the regularization term')    
    parser.add_argument('--t1', default = False, action="store_true", help = 'If true, pretrain on task 1 (self-supervised learning on attribute-item graph)')
    parser.add_argument('--t2', default = False, action="store_true", help = 'If true, pretrain on task 2 (predict masked items in users\' purchase history)')
    parser.add_argument('--t3', default = False, action="store_true", help = 'If true, pretrain on task 3 (predict items using reviews)')
    parser.add_argument('--w_t1', type = float, default = 1, help = 'The weight of task 1')
    parser.add_argument('--w_t2', type = float, default = 1, help = 'The weight of task 2')
    parser.add_argument('--w_t3', type = float, default = 1, help = 'The weight of task 3')

    # downstream recommendation args
    parser.add_argument('--k_list', type = list, default = [5, 20, 40], help = 'k list for computing ndcg')

    return parser.parse_args()

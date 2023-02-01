import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()
        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class BERT4RecEmbedding(nn.Module):
    """
    BERT4RecEmbedding consists the following:
        1. TokenEmbedding : embeddings of the tokens (items) in the sequences
        2. PositionalEmbedding : positional information using sin, cos
    BERTRecEmbedding outputs the sum of 1 and 2
    """
    def __init__(self, embed_size, max_len, num_items, task1, dropout=0.1, perturb=False):
        """
        :param embed_size: embedding size of the token embeddings
        :param dropout: dropout rate
        """
        super().__init__()
        self.token_0 = nn.Parameter(torch.zeros(1, embed_size), requires_grad = False)
        self.task1 = task1
        self.token_mask = nn.Parameter(torch.Tensor(1, embed_size))
        nn.init.xavier_normal_(self.token_mask)

        self.num_items = num_items
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.perturb = perturb
       
    def forward(self, sequence):
        #print('In BERT4RecEmbedding forward')
        #print(sequence)
        embeddings = self.task1.extract_embeddings(perturb = self.perturb)
        x = torch.cat((self.token_0, embeddings[:self.num_items], self.token_mask), 0)[sequence]
        p = self.position(sequence)
        x += p
        return self.dropout(x), embeddings

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class LayerNorm(nn.Module):
    """
    Construct a layernorm module
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
        
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class BERT4Rec(nn.Module):
    def __init__(self, args, task1, num_items):
        super().__init__()
        max_len = args.BERT4Rec_max_len
        n_layers = args.BERT4Rec_n_layers
        heads = args.BERT4Rec_n_heads
        self.hidden = args.embed_dim
        dropout = args.BERT4Rec_dropout
        perturb = args.perturb
        self.task1 = task1
        # multi-layers of transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, heads, self.hidden * 4, dropout) for _ in range(n_layers)])
        
        self.embedding = BERT4RecEmbedding(embed_size = self.hidden, max_len = max_len, num_items = num_items, task1 = self.task1, dropout = dropout, perturb = perturb)

    def forward(self, x):
        #print('---- Inside BERT4Rec forward ----')
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1) # mask out the pads

        # embedding the indexed sequence to sequence of vectors
        x, embeddings = self.embedding(x)
        #print(x.size())
       
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask) # x: B x T x E (B: batch size, T: sequence length, E: embeddings dimension, V: total number of items in the training set)
        x = x.view(-1, x.size(-1))  #  convert the size of x from B x T x E to (B*T) x E

        #print(x)
        #print(x.size())

        #return torch.mm(x, torch.cat((self.pad_embedding, self.encoder.get_item_embeddings()), 0).transpose(0,1))
        return x, embeddings

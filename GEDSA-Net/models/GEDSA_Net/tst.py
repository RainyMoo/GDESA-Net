from ops.fc import FC, MLP
from ops.layer_norm import LayerNorm
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------
    
class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        
        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y
        
    
class Merge_SGA(nn.Module):
    def __init__(self, __C):
        super(Merge_SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
                
        self.ffn = FFN(__C)
        self.grid_topk = False
        self.region_topk = False


        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)
        
        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)
        
        self.dropout4 = nn.Dropout(__C.DROPOUT_R)
        self.norm4 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout5 = nn.Dropout(__C.DROPOUT_R)
        self.norm5 = LayerNorm(__C.HIDDEN_SIZE)
        
        self.dropout6 = nn.Dropout(__C.DROPOUT_R)
        self.norm6 = LayerNorm(__C.HIDDEN_SIZE)
        
    def forward(self, x, y, z, x_mask, y_mask, z_mask):

        y_topk = y
        x_topk = x
       
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=y_topk, k=y_topk, q=x, mask=y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.mhatt1(v=x_topk, k=x_topk, q=y, mask=x_mask)
        ))
        
        x = self.norm3(x + self.dropout3(
            self.mhatt2(v=z, k=z, q=x, mask=z_mask)
        ))
        
        y = self.norm4(y + self.dropout4(
            self.mhatt2(v=z, k=z, q=y, mask=z_mask)
        ))
        
        x = self.norm5(x+ self.dropout5(
            self.ffn(x)
        ))

        y = self.norm6(y+ self.dropout6(
            self.ffn(y)
        ))
        

        return x, y


class TST_ED(nn.Module):
    def __init__(self, __C):
        super(TST_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.union_enc_list = nn.ModuleList([Merge_SGA(__C) for _ in range(int(__C.LAYER))])
        

    def forward(self, lang, region, grid, lang_mask, region_mask, grid_mask):
        # Get encoder last hidden vector
        
        for enc in self.enc_list:
            lang = enc(lang, lang_mask)

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        
        for union_enc in self.union_enc_list:
            region, grid = union_enc(region, grid, lang, region_mask, grid_mask, lang_mask)
        
        return lang, region, grid
    
    
from utils.make_mask import make_mask
from ops.fc import FC, MLP
from ops.layer_norm import LayerNorm
from models.GEDSA_Net.tst import TST_ED
from models.GEDSA_Net.adapter import Adapter
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers import BertTokenizerFast, BertModel

import os
# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)
        #att = softmax_with_temperature(att, T=0.5)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

class Gate_Fusion(nn.Module):
    def __init__(self, __C):
        super(Gate_Fusion, self).__init__()
        self.__C = __C
        self.gate = nn.Linear(__C.FLAT_OUT_SIZE * 2, 1)
    
    def forward(self, lang_feat, img_feat):
        sum_feat = torch.cat((lang_feat, img_feat), dim=1)
        VT = F.sigmoid(self.gate(sum_feat))
        img_feat = torch.mul(img_feat, VT)
        lang_feat_fanshu = torch.norm(lang_feat, p=2)
        img_feat_fanshu = torch.norm(img_feat, p=2)
        gama = min(lang_feat_fanshu / img_feat_fanshu, 1)
        proj_feat = lang_feat + gama * img_feat
        
        return proj_feat    
    
# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class BiGeometricProjection(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.grid_proj = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.region_proj = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.gate = nn.Sequential(nn.Linear(__C.HIDDEN_SIZE * 2, __C.HIDDEN_SIZE), nn.Sigmoid())

    def forward(self, region_feat, grid_feat, rg_iou):
        """
        region_feat: [B, R, D]
        grid_feat:   [B, G, D]
        rg_iou:      [B, R, G] (值域 0~1)
        """
        rg_iou = rg_iou.to(dtype=grid_feat.dtype)

        iou_r2g = F.softmax(rg_iou * 10.0, dim=-1) # 
        
        iou_g2r = F.softmax(rg_iou.transpose(1, 2) * 10.0, dim=-1)

        grid_context = torch.matmul(iou_r2g, grid_feat)
        
        region_semantic = torch.matmul(iou_g2r, region_feat)
        
        region_out = region_feat + self.gate(torch.cat([region_feat, grid_context], -1)) * grid_context
        
        grid_out = grid_feat + self.gate(torch.cat([grid_feat, region_semantic], -1)) * region_semantic
        
        return region_out, grid_out

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        # -----------------------
        # 1. GloVe + LSTM encoder
        # -----------------------
        if __C.USE_GloVe:
            self.embedding = nn.Embedding(token_size, __C.WORD_EMBED_SIZE)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
            self.lstm = nn.LSTM(
                __C.WORD_EMBED_SIZE, __C.HIDDEN_SIZE,
                num_layers=1, batch_first=True
            )

        # -----------------------
        # 2. BERT encoder
        # -----------------------
        if __C.USE_BERT:
            self.bert_model = BertModel.from_pretrained("bert-base-uncased")
            #self.bert_proj = nn.Linear(768, __C.HIDDEN_SIZE) 
            self.lstm = nn.LSTM(
                768, __C.HIDDEN_SIZE,
                num_layers=1, batch_first=True
            )

            # 冻结 BERT 所有参数
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.adapter = Adapter(__C)
        

        self.bigeo_proj = BiGeometricProjection(__C)

   
        self.backbone = TST_ED(__C)

        
        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        self.attflat_grid = AttFlat(__C)
    
        self.gate_fusion1 = Gate_Fusion(__C)
        self.gate_fusion2 = Gate_Fusion(__C)

        self.proj_norm2 = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj2 = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)
    
    def encode_text(self, ques_ix):
        """
        ques_ix:
            glove 模式：LongTensor(B,14)
            bert 模式：LongTensor(B,14)
            clip 模式：LongTensor(B,14) (clip.tokenize 生成)
        """
        if self.__C.USE_GloVe:
            emb = self.embedding(ques_ix)
            lang_feat, _ = self.lstm(emb)
            return lang_feat                       # (B,14,H)
    
        elif self.__C.USE_BERT:
            # ques_ix 是 tokenizer 输出的 input_ids
            attention_mask = (ques_ix != 0).long()
            out = self.bert_model(
                input_ids=ques_ix,
                attention_mask=attention_mask
            ).last_hidden_state                   # (B,14,768)
            #lang_feat = self.bert_proj(out)
            lang_feat, _ = self.lstm(out)
            return lang_feat          #  (B,14,H)

    
        else:
            raise ValueError("Unknown TEXT_ENCODER")



    def compute_local_alignment_loss(self, region_feat, grid_feat, rg_iou, rg_align):
        """
        Args:
            region_feat: [B, R, D]
            grid_feat:   [B, G, D]
            rg_iou:      [B, R, G] (数值 0.0 ~ 1.0)
            rg_align:    [B, R, G] (数值 0 或 1)
        """
        rg_iou = rg_iou.detach()
        rg_align = rg_align.detach().float()

        weights = rg_iou * rg_align 
        
        total_weight = weights.sum()
        if total_weight < 1e-5:
            return torch.tensor(0.0, device=region_feat.device, requires_grad=True)
    
        region_norm = F.normalize(region_feat, dim=-1)
        grid_norm = F.normalize(grid_feat, dim=-1)
    
        sim_matrix = torch.matmul(region_norm, grid_norm.transpose(1, 2))
    
        dist_matrix = 1.0 - sim_matrix
        
        weighted_dist = weights * dist_matrix
        
        loss = weighted_dist.sum() / total_weight
        
        return loss

    def forward(self, frcn_feat, grid_feat, bbox_feat, grid_bbox_feat, w_feat, h_feat, region_align, region_iou, ques_ix, ques_tensor):

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2)) 
        lang_feat = self.encode_text(ques_ix)
        
                   
        frcn_feat, frcn_feat_mask, grid_feat, grid_feat_mask, rg_align, rg_iou = self.adapter(frcn_feat, grid_feat, bbox_feat, grid_bbox_feat,w_feat, h_feat, region_align, region_iou) 


        local_align_loss = self.compute_local_alignment_loss(frcn_feat, grid_feat, rg_iou, rg_align)
            
        frcn_feat, grid_feat = self.bigeo_proj(frcn_feat,grid_feat,rg_iou)
        
        # Backbone Framework
        
        lang_feat, frcn_feat, grid_feat = self.backbone(
            lang_feat,
            frcn_feat,
            grid_feat,
            lang_feat_mask,
            frcn_feat_mask,
            grid_feat_mask
        )


        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )
    
        grid_feat = self.attflat_grid(
            grid_feat,
            grid_feat_mask
        )
        
        frcn_feat = self.attflat_img(
            frcn_feat,
            frcn_feat_mask
        )
        
        # Classification layers
        proj_feat1 = self.gate_fusion1(lang_feat, frcn_feat)

        proj_feat2 = self.gate_fusion2(lang_feat, grid_feat)

        
        proj_feat = self.proj_norm2(proj_feat1 + proj_feat2)
        proj_feat = self.proj2(proj_feat)
        
        
        if self.__C.ITC_LOSS == 'True':
            
            return proj_feat, local_align_loss

        else:
            return proj_feat



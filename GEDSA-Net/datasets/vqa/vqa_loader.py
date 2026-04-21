import torch
import numpy as np
import glob, json, re, en_vectors_web_lg
from core.base_dataset import BaseDataSet
from utils.ans_punct import prep_ans
from transformers import BertTokenizerFast
import time
import clip
import math

import os

class DataSet(BaseDataSet):
    def __init__(self, __C):
        super(DataSet, self).__init__()
        self.__C = __C

        if getattr(self.__C, "USE_BERT", False):
            from transformers import BertTokenizerFast
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(
                "bert-base-uncased",
                do_lower_case=True
            )
        
        # ============================
        # 初始化 CLIP tokenizer（无 special tokens）
        # ============================
        if getattr(self.__C, "USE_CLIP", False):
            self.clip_tokenizer = clip.tokenize   # 直接引用 clip.tokenize

        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        # Loading all region image paths
        
        frcn_feat_path_list = \
            glob.glob(__C.FEATS_PATH[__C.DATASET]['train'] + '/*.npz') + \
            glob.glob(__C.FEATS_PATH[__C.DATASET]['val'] + '/*.npz') + \
            glob.glob(__C.FEATS_PATH[__C.DATASET]['test'] + '/*.npz')
        
        #frcn_feat_path_list = \
        #    glob.glob(__C.FEATS_PATH[__C.DATASET]['train'] + '/*.npy') + \
        #    glob.glob(__C.FEATS_PATH[__C.DATASET]['val'] + '/*.npy') + \
        #   glob.glob(__C.FEATS_PATH[__C.DATASET]['test'] + '/*.npy')
        

        # Loading all grid image paths
        grid_feat_path_list = \
            glob.glob(__C.FEATS_PATH['vqa_grid']['train'] + '/*.npy') + \
            glob.glob(__C.FEATS_PATH['vqa_grid']['val'] + '/*.npy') + \
            glob.glob(__C.FEATS_PATH['vqa_grid']['test'] + '/*.npy')
       
        # Loading all vit image paths

        # vit_feat_path_list = \
        #     glob.glob(__C.FEATS_PATH['vqa_vit']['train'] + '/*.npy') + \
        #     glob.glob(__C.FEATS_PATH['vqa_vit']['val'] + '/*.npy') + \
        #     glob.glob(__C.FEATS_PATH['vqa_vit']['test'] + '/*.npy')

        
        # Loading region-grid align paths
        region_grid_align_path_list = \
            glob.glob(__C.ALIGNS_PATH['vqa_8']['train'] + '/*.npz') + \
            glob.glob(__C.ALIGNS_PATH['vqa_8']['val'] + '/*.npz') + \
            glob.glob(__C.ALIGNS_PATH['vqa_8']['test'] + '/*.npz')
        
        # Loading all graph_relations image paths
        # '''
        # spatial_graph_path_list= \
        #     glob.glob(__C.SPATIALS_PATH[__C.DATASET]['train'] + '/*.npz') + \
        #     glob.glob(__C.SPATIALS_PATH[__C.DATASET]['val'] + '/*.npz') + \
        #     glob.glob(__C.SPATIALS_PATH[__C.DATASET]['test'] + '/*.npz')    
        # '''   
            
        # Loading question word list
        stat_ques_list = \
            json.load(open(__C.RAW_PATH[__C.DATASET]['train'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['val'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['test'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['vg'], 'r'))['questions']

        # Loading answer word list
        # stat_ans_list = \
        #     json.load(open(__C.RAW_PATH[__C.DATASET]['train-anno'], 'r'))['annotations'] + \
        #     json.load(open(__C.RAW_PATH[__C.DATASET]['val-anno'], 'r'))['annotations']

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(__C.RAW_PATH[__C.DATASET][split], 'r'))['questions']
            if __C.RUN_MODE in ['train']:
                self.ans_list += json.load(open(__C.RAW_PATH[__C.DATASET][split + '-anno'], 'r'))['annotations']

        # Define run data size
        if __C.RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        print(' ========== Dataset size:', self.data_size)


        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}
        self.iid_to_frcn_feat_path = self.img_feat_path_load(frcn_feat_path_list)
        
        self.iid_to_grid_feat_path = self.img_feat_path_load(grid_feat_path_list)
        
        #self.iid_to_vit_feat_path = self.img_feat_path_load(vit_feat_path_list)
        
        # {image id} -> {image region-grid align information}
        self.iid_to_align_path = self.img_feat_path_load(region_grid_align_path_list)
        
        # {image id} -> {image spatial graph relation information}
        #self.iid_to_spatial_graph_path = self.img_feat_path_load(spatial_graph_path_list)

        # {question id} -> {question}
        self.qid_to_ques = self.ques_load(self.ques_list)

        # Tokenize
        self.token_to_ix, self.pretrained_emb = self.tokenize(stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)

        # Answers statistic
        self.ans_to_ix, self.ix_to_ans = self.ans_stat('datasets/vqa/answer_dict.json')
        # self.ans_to_ix, self.ix_to_ans = self.ans_stat(stat_ans_list, ans_freq=8)
        self.ans_size = self.ans_to_ix.__len__()
        print(' ========== Answer token vocab size (occur more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')



    def img_feat_path_load(self, path_list):
        iid_to_path = {}

        for ix, path in enumerate(path_list):
            iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
            # print(iid)
            iid_to_path[iid] = path

        return iid_to_path


    def ques_load(self, ques_list):
        qid_to_ques = {}

        for ques in ques_list:
            qid = str(ques['question_id'])
            qid_to_ques[qid] = ques

        return qid_to_ques

    def tokenize(self, stat_ques_list, use_glove):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }
        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        for ques in stat_ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques['question'].lower()
            ).replace('-', ' ').replace('/', ' ').split()

            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb


    # def ans_stat(self, stat_ans_list, ans_freq):
    #     ans_to_ix = {}
    #     ix_to_ans = {}
    #     ans_freq_dict = {}
    #
    #     for ans in stat_ans_list:
    #         ans_proc = prep_ans(ans['multiple_choice_answer'])
    #         if ans_proc not in ans_freq_dict:
    #             ans_freq_dict[ans_proc] = 1
    #         else:
    #             ans_freq_dict[ans_proc] += 1
    #
    #     ans_freq_filter = ans_freq_dict.copy()
    #     for ans in ans_freq_dict:
    #         if ans_freq_dict[ans] <= ans_freq:
    #             ans_freq_filter.pop(ans)
    #
    #     for ans in ans_freq_filter:
    #         ix_to_ans[ans_to_ix.__len__()] = ans
    #         ans_to_ix[ans] = ans_to_ix.__len__()
    #
    #     return ans_to_ix, ix_to_ans

    def ans_stat(self, json_file):
        ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))

        return ans_to_ix, ix_to_ans



    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_ques_ans(self, idx):
        if self.__C.RUN_MODE in ['train']:
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]
            iid = str(ans['image_id'])

            # Process question
            
            # download ques_feature through bert pretrained
            #ques_tensor_iter = np.zeros(1)
           
            if self.__C.USE_BERT == 'True':
                ques_ix_iter = self.proc_ques_bert(ques, max_token=14)
            else:
                ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)
            

            # Process answer
            ans_iter = self.proc_ans(ans, self.ans_to_ix)
            #print(ques_tensor_iter.shape)

            return ques_ix_iter, ans_iter, iid

        else:
            ques = self.ques_list[idx]
            iid = str(ques['image_id'])
            
            #ques_tensor_iter = np.zeros(1)

            ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)

            return ques_ix_iter, np.zeros(1), iid


    # def load_ques_ans(self, idx):
    #     """
    #     返回:
    #         ques_ix_iter: (14,) int64 —— 文本token索引（无special token，padding=0）
    #         ans_iter:     answer soft label / vector
    #         iid:          image id
    #     """
    
    #     # ----------------------------
    #     # Train 模式
    #     # ----------------------------
    #     if self.__C.RUN_MODE in ['train']:
    #         ans = self.ans_list[idx]
    #         ques = self.qid_to_ques[str(ans['question_id'])]
    #         iid = str(ans['image_id'])
    
    #         # ---- Question Encoding ----
    #         if getattr(self.__C, "USE_CLIP", False):
    #             ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)
    
    #         elif getattr(self.__C, "USE_BERT", False):
    #             ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)
    
    #         else:
    #             # 默认：GloVe+LSTM
    #             ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)
    
    #         # ---- Answer ----
    #         ans_iter = self.proc_ans(ans, self.ans_to_ix)
    
    #         return ques_ix_iter, np.zeros(1), ans_iter, iid
    
    #     # ----------------------------
    #     # Eval / Test 模式
    #     # ----------------------------
    #     else:
    #         ques = self.ques_list[idx]
    #         iid = str(ques['image_id'])
    
    #         if getattr(self.__C, "USE_CLIP", False):
    #             ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)
    
    #         elif getattr(self.__C, "USE_BERT", False):
    #             ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)
    
    #         else:
    #             ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)
    
    #         return ques_ix_iter, np.zeros(1), np.zeros(1), iid

    '''
    def load_img_feats(self, idx, iid):
        frcn_feat = np.load(self.iid_to_frcn_feat_path[iid])
        grid_feat = np.load(self.iid_to_grid_feat_path[iid]) # 64 2048
        vit_feat = np.load(self.iid_to_vit_feat_path[iid])  # 197 768
        align = np.load(self.iid_to_align_path[iid])
        grid_feat = grid_feat.astype(np.float32)   
        vit_feat = vit_feat.astype(np.float32)
        
        spatial_graph = np.load(self.iid_to_spatial_graph_path[iid])
        frcn_feat_spa = spatial_graph['graph']
        frcn_feat_x = frcn_feat['x'].transpose((1, 0))
        frcn_feat_bbox = frcn_feat['bbox']
        frcn_feat_h = frcn_feat['image_h']
        frcn_feat_w = frcn_feat['image_w']
        frcn_feat_iter = self.proc_img_feat(frcn_feat_x, img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])

        
        bbox_feat_iter = self.proc_img_feat(
            self.proc_bbox_feat(
                frcn_feat['bbox'],
                (frcn_feat['image_h'], frcn_feat['image_w'])
            ),
            img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['BBOX_FEAT_SIZE'][0]
        )
        
        bbox_feat_iter = self.proc_img_feat(
            frcn_feat['bbox'],
            img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['BBOX_FEAT_SIZE'][0]
        )

        #grid_feat_iter = np.zeros(1)
        grid_feat_iter = self.proc_img_feat(grid_feat, img_feat_pad_size=self.__C.FEAT_SIZE['vqa_grid']['FRCN_FEAT_SIZE'][0])
        spa_graph_iter = self.proc_spa_graph(frcn_feat_spa, img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])
        vit_feat_iter = vit_feat
        w_feat_iter = frcn_feat_w
        h_feat_iter = frcn_feat_h
        
        # process region-grid-align
        region_grid_align = align['mask']
        region_grid_aligns = self.proc_img_feat(region_grid_align, img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])
        region_num = region_grid_align.shape[0]
        grid_num = region_grid_align.shape[1]
        
        r2g_mask = np.eye(self.__C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][0])
        r2g_mask[region_num:,:] = 0
        r2g_mask[:,region_num:] = 0
        
        region_aligns = (np.concatenate([r2g_mask, region_grid_aligns],axis=1) == 0)
        
        g2r_mask = np.eye(grid_num)
        grid_aligns = (np.concatenate([region_grid_aligns.transpose(1, 0),g2r_mask],axis=1) == 0)
        
        region_aligns_iter = region_aligns  # (100, 164)
        grid_aligns_iter = grid_aligns  #(64, 164)        
        
        return frcn_feat_iter, grid_feat_iter, vit_feat_iter, bbox_feat_iter, w_feat_iter, h_feat_iter, spa_graph_iter, region_aligns_iter, grid_aligns_iter
    '''
    
    def load_img_feats(self, idx, iid):
        # '''
        # vqa_feat:['bbox', 'image_h', 'image_w', 'num_bbox', 'x']
        # vqa_x152_feat:['bbox','cls_prob','features','image_height','image_width','num_boxes','objects']
        # #np.load(self.iid_to_frcn_feat_path[iid],allow_pickle=True).item()
        # '''
        # for x_152_8 feats
        # frcn_feat = np.load(self.iid_to_frcn_feat_path[iid],allow_pickle=True).item()
        # frcn_feat_x = frcn_feat['features']
        # frcn_feat_bbox = frcn_feat['bbox']
        # frcn_feat_h = np.array(frcn_feat['image_height'])
        # frcn_feat_w = np.array(frcn_feat['image_width'])
        
        #for vqa_8_feat_base
        frcn_feat = np.load(self.iid_to_frcn_feat_path[iid],allow_pickle=True)
        #frcn_feat_x = frcn_feat['x'].transpose((1, 0))
        frcn_feat_x = frcn_feat['x']
        frcn_feat_bbox = frcn_feat['bbox']
        frcn_feat_h = np.array(frcn_feat['image_h'])
        frcn_feat_w = np.array(frcn_feat['image_w'])
        
        frcn_feat_iter = self.proc_img_feat(frcn_feat_x, img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])
        
        grid_feat = np.load(self.iid_to_grid_feat_path[iid]) # 64 2048
        grid_feat_iter = self.proc_img_feat(grid_feat, img_feat_pad_size=self.__C.FEAT_SIZE['vqa_grid']['FRCN_FEAT_SIZE'][0])

        grid_bbox_feat_iter = self.make_grid_bboxes(
            image_w=int(frcn_feat['image_w']),
            image_h=int(frcn_feat['image_h']),
            S=int(math.sqrt(self.__C.FEAT_SIZE['vqa_grid']['FRCN_FEAT_SIZE'][0])),
            inclusive=True,   # 与你的像素+1约定一致
        ) 
        
        bbox_feat_iter = self.proc_img_feat(
            frcn_feat['bbox'],
            img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['BBOX_FEAT_SIZE'][0]
        )       

        w_feat_iter = frcn_feat_w
        h_feat_iter = frcn_feat_h
        
        #process region-grid-align
        align = np.load(self.iid_to_align_path[iid])
        maxcov = align['maxcov']
        iou = align['iou']
        
        # align_mask
        
        rg_align_mask = self.build_rg_mask(maxcov=maxcov, tau_cov=1e-8, iou = iou, tau_iou=None)

        
        
        region_num = rg_align_mask.shape[0]
        grid_num = rg_align_mask.shape[1]
        
        r2g_mask = self.proc_img_feat(rg_align_mask, img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])
        
#         r2r_mask = np.eye(self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])
#         r2r_mask[region_num:,:] = 0
#         r2r_mask[:,region_num:] = 0
        
#         region_aligns_iter = (np.concatenate([r2r_mask, r2g_mask],axis=1)==0)  # 100, 164
        
        # g2g_mask = np.eye(grid_num)
        # grid_aligns_iter = (np.concatenate([r2g_mask.transpose(1, 0),g2g_mask],axis=1)==0)    # 64, 164     
        
        
        # iou
        r2g_iou = self.proc_img_feat(iou, img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])
        
#         r2r_iou = np.eye(self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])
#         r2r_iou[region_num:,:] = 0
#         r2r_iou[:,region_num:] = 0
        
#         region_iou_iter = np.concatenate([r2r_iou, r2g_iou],axis=1)  # 100, 164
        
#         g2g_iou = np.eye(grid_num)
#         grid_iou_iter = np.concatenate([r2g_iou.transpose(1, 0),g2g_iou],axis=1)    # 64, 164   

        region_aligns_iter = r2g_mask
        #grid_aligns_iter = r2g_mask.transpose(1, 0)
        region_iou_iter = r2g_iou
        #grid_iou_iter = r2g_iou.transpose(1, 0)

        #print(frcn_feat_iter.shape,grid_feat_iter.shape,bbox_feat_iter.shape,region_aligns_iter.shape,grid_aligns_iter.shape)
        return frcn_feat_iter, grid_feat_iter, bbox_feat_iter, grid_bbox_feat_iter, w_feat_iter, h_feat_iter, region_aligns_iter, region_iou_iter,np.zeros(1)
        # return frcn_feat_iter, grid_feat_iter, bbox_feat_iter, w_feat_iter, h_feat_iter, region_aligns_iter, grid_aligns_iter, region_iou_iter, grid_iou_iter      
    # ------------------------------------
    # ---- Real-Time Processing Utils ----
    # ------------------------------------

    def proc_spa_graph(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, img_feat_pad_size - img_feat.shape[1])),
            mode='constant',
            constant_values=19
        )

        return img_feat
    
    def proc_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return img_feat
    
    def pad_list(self, input_list, target_length, pad_value=""):
        """
        将 Python 列表填充或截断到固定长度
        """
        current_len = len(input_list)
        
        if current_len >= target_length:
            return input_list[:target_length]
        
        else:
            pad_num = target_length - current_len
            # 列表拼接
            return input_list + [pad_value] * pad_num
        

    def proc_bbox_feat(self, bbox, img_shape):
        if self.__C.BBOX_NORMALIZE:
            bbox_nm = np.zeros((bbox.shape[0], 4), dtype=np.float32)

            bbox_nm[:, 0] = bbox[:, 0] / float(img_shape[1])
            bbox_nm[:, 1] = bbox[:, 1] / float(img_shape[0])
            bbox_nm[:, 2] = bbox[:, 2] / float(img_shape[1])
            bbox_nm[:, 3] = bbox[:, 3] / float(img_shape[0])
            return bbox_nm
        # bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox


    # def proc_ques_bert(self, ques, max_token):
    #     bert_model = './bert-base-uncased'
        
    #     tokenizer = BertTokenizer.from_pretrained(bert_model)

    #     words = re.sub(
    #         r"([.,'!?\"()*#:;])",
    #         '',
    #         ques['question'].lower()
    #     )
    #     tokens = tokenizer(words, add_special_tokens=False, padding='max_length', truncation=True, max_length=max_token)

    #     return np.array(tokens['input_ids'])

    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break
        return ques_ix

    
    def proc_attr_obj(self, attr_obj, word_length, max_region):
        # 1. 确定序列长度
        # 如果你现在的句子是 "this is a red cat"，长度是 5
        # 为了保险，你可以设大一点，比如 10，或者设为你的句子固定长度
        seq_len = word_length  
        
        # 2. 初始化矩阵
        # [Region数量, 序列长度]
        # 原来的代码这里可能写的是 (max_region, 2)，一定要改掉！
        attr_obj_feat = np.zeros((max_region, seq_len), dtype=np.int64)
    
        # 3. 修改循环逻辑
        # 不要写 (attr, obj)，改成 words (接收整个元组)
        for r, words in enumerate(attr_obj[:max_region]):
            
            # words 现在是 ('this', 'is', 'a', 'red', 'cat')
            
            # 4. 内部循环遍历单词
            for i, word in enumerate(words):
                if i >= seq_len:
                    break
                
                # 5. 转 Token ID (根据你原有的词表逻辑)
                # 假设你有一个 self.token_to_ix 字典
                if word in self.token_to_ix:
                    attr_obj_feat[r, i] = self.token_to_ix[word]
                else:
                    # 处理未知词，如果有 UNK token 的话
                    attr_obj_feat[r, i] = self.token_to_ix['UNK']
                    pass 
    
        return attr_obj_feat
    
    # def proc_ques(self, ques, token_to_ix, max_token=14):
    #     q_str = ques["question"].strip()
    
    #     # ============================================
    #     # 1) CLIP（无 special tokens + pad 到 14）
    #     # ============================================
    #     if getattr(self.__C, "USE_CLIP", False):
    #         # 使用你自定义的 CLIP tokenizer（不添加 special tokens）
    #         # self.clip_tokenizer_tokenize() 返回 list[int]
    #         # step 1: CLIP tokenize (77 tokens)
    #         toks = clip.tokenize([q_str], truncate=True)[0].numpy()   # (77,)
        
    #         # step 2: remove CLIP special tokens（49406=SOT, 49407=EOT）
    #         toks = toks[(toks != 49406) & (toks != 49407)]
        
    #         # step 3: remove pad (0)
    #         toks = toks[toks != 0]
        
    #         # step 4: truncate to max_token
    #         toks = toks[:max_token]
        
    #         # step 5: pad to max_token
    #         if len(toks) < max_token:
    #             toks = np.pad(toks, (0, max_token - len(toks)), constant_values=0)
        
    #         return toks.astype(np.int64)
    
    
    #     # ============================================
    #     # 2) BERT（无 special tokens + pad 到 14）
    #     # ============================================
    #     if getattr(self.__C, "USE_BERT", False):
    #         ids = self.bert_tokenizer(
    #             q_str,
    #             add_special_tokens=False,      # <<< 关键：不要 CLS、SEP
    #             truncation=True,
    #             max_length=max_token
    #         )["input_ids"]
    
    #         # padding
    #         pad_len = max_token - len(ids)
    #         if pad_len > 0:
    #             ids = ids + [0] * pad_len
    
    #         return np.array(ids, np.int64)
    
    
    #     # ============================================
    #     # 3) GloVe（原始 + pad 到 14）
    #     # ============================================
    #     ques_ix = np.zeros(max_token, np.int64)
    
    #     words = re.sub(
    #         r"([.,'!?\"()*#:;])",
    #         "",
    #         q_str.lower()
    #     ).replace("-", " ").replace("/", " ").split()
    
    #     for ix, w in enumerate(words[:max_token]):
    #         ques_ix[ix] = token_to_ix.get(w, token_to_ix["UNK"])
    
    #     return ques_ix

    def get_score(self, occur):
        if occur == 0:
            return .0
        elif occur == 1:
            return .3
        elif occur == 2:
            return .6
        elif occur == 3:
            return .9
        else:
            return 1.


    def proc_ans(self, ans, ans_to_ix):
        ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
        ans_prob_dict = {}

        for ans_ in ans['answers']:
            ans_proc = prep_ans(ans_['answer'])
            if ans_proc not in ans_prob_dict:
                ans_prob_dict[ans_proc] = 1
            else:
                ans_prob_dict[ans_proc] += 1

        if self.__C.LOSS_FUNC in ['kld']:
            for ans_ in ans_prob_dict:
                if ans_ in ans_to_ix:
                    ans_score[ans_to_ix[ans_]] = ans_prob_dict[ans_] / 10.
        else:
            for ans_ in ans_prob_dict:
                if ans_ in ans_to_ix:
                    ans_score[ans_to_ix[ans_]] = self.get_score(ans_prob_dict[ans_])

        return ans_score
    
    
    def build_rg_mask(self,
        maxcov: np.ndarray,
        tau_cov: float = 1e-9,
        iou: np.ndarray = None,
        tau_iou: float = None,
    ) -> np.ndarray:
        """
        基于覆盖率与可选 IoU 的在线构造：
          rg_mask[i,j] = 1  若 max(R2G,G2R) >= tau_cov
                      或 (启用 IoU 兜底时) IoU >= tau_iou
          否则 = 0

        参数
        ----
        r2g, g2r, maxcov : [Nr, Ng]，可为 float16/float32
        tau_cov          : 覆盖率阈值（建议 0.4~0.6）
        iou              : [Nr, Ng]，可选
        tau_iou          : IoU 兜底阈值（如 0.1~0.2）；仅当 iou 与 tau_iou 均不为 None 时生效

        返回
        ----
        rg_mask : [Nr, Ng]，float32 的 0/1 矩阵
        """
        # 统一到 float32 做比较（支持你存的 float16）

        cond = (maxcov >= float(tau_cov))

        if (iou is not None) and (tau_iou is not None):
            cond = np.logical_or(cond, iou >= float(tau_iou))

        return cond

    def make_grid_bboxes(self, image_w, image_h, S=8, inclusive=True):
        """
        返回 [S*S, 4] 的网格 bbox，xyxy。
        inclusive=True 时，x2/y2 会减去 1.0，以匹配你 bbox_transform 里宽高 +1 的像素约定。
        """
        x_edges = np.linspace(0.0, float(image_w), S + 1, dtype=np.float32)
        y_edges = np.linspace(0.0, float(image_h), S + 1, dtype=np.float32)
        x1 = x_edges[:-1]
        x2 = x_edges[1:] - (1.0 if inclusive else 0.0)
        y1 = y_edges[:-1]
        y2 = y_edges[1:] - (1.0 if inclusive else 0.0)
        X1, Y1 = np.meshgrid(x1, y1)   # [S,S]
        X2, Y2 = np.meshgrid(x2, y2)   # [S,S]
        cells = np.stack([X1, Y1, X2, Y2], axis=-1).reshape(-1, 4)  # [S*S,4]
        return cells.astype(np.float32)

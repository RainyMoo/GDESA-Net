import numpy as np
import glob, json, re, en_vectors_web_lg
from base_dataset import BaseDataSet
from utils.ans_punct import prep_ans
import math
import os

class DataSet(BaseDataSet):
    def __init__(self, __C):
        super(DataSet, self).__init__()
        self.__C = __C

        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        ques_dict_preread = {
            'train': json.load(open(__C.RAW_PATH[__C.DATASET]['train'], 'r')),
            'val': json.load(open(__C.RAW_PATH[__C.DATASET]['val'], 'r')),
            'testdev': json.load(open(__C.RAW_PATH[__C.DATASET]['testdev'], 'r')),
            'test': json.load(open(__C.RAW_PATH[__C.DATASET]['test'], 'r')),
        }

        # Loading all image paths
        frcn_feat_path_list = glob.glob(__C.FEATS_PATH[__C.DATASET]['default-frcn'] + '/*.npz')
        grid_feat_path_list = glob.glob(__C.FEATS_PATH[__C.DATASET]['default-grid'] + '/*.npy')
        align_feat_path_list = glob.glob(__C.FEATS_PATH[__C.DATASET]['default-align'] + '/*.npz')

        # Loading question word list
        # stat_ques_dict = {
        #     **ques_dict_preread['train'],
        #     **ques_dict_preread['val'],
        #     **ques_dict_preread['testdev'],
        #     **ques_dict_preread['test'],
        # }

        # Loading answer word list
        # stat_ans_dict = {
        #     **ques_dict_preread['train'],
        #     **ques_dict_preread['val'],
        #     **ques_dict_preread['testdev'],
        # }

        # Loading question and answer list
        self.ques_dict = {}
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            if split in ques_dict_preread:
                self.ques_dict = {
                    **self.ques_dict,
                    **ques_dict_preread[split],
                }
            else:
                self.ques_dict = {
                    **self.ques_dict,
                    **json.load(open(__C.RAW_PATH[__C.DATASET][split], 'r')),
                }

        # Define run data size
        self.data_size = self.ques_dict.__len__()
        print(' ========== Dataset size:', self.data_size)


        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}
        self.iid_to_frcn_feat_path = self.img_feat_path_load(frcn_feat_path_list)
        self.iid_to_grid_feat_path = self.img_feat_path_load(grid_feat_path_list)
        self.iid_to_align_feat_path = self.img_feat_path_load(align_feat_path_list)

        # Loading dict: question dict -> question list
        self.qid_list = list(self.ques_dict.keys())

        #data_path = '../'
        self.classes = []  # 1600
        with open(os.path.join('objects_vocab.txt')) as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())
        
        # Load attributes
        self.attributes = []  # 400
        with open(os.path.join('attributes_vocab.txt')) as f:
            for att in f.readlines():
                self.attributes.append(att.split(',')[0].lower().strip())

        # Tokenize
        self.token_to_ix, self.pretrained_emb, max_token = self.tokenize('datasets/gqa/dicts.json', __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)

        self.max_token = -1
        if self.max_token == -1:
            self.max_token = max_token
        print('Max token length:', max_token, 'Trimmed to:', self.max_token)

        # Answers statistic
        self.ans_to_ix, self.ix_to_ans = self.ans_stat('datasets/gqa/dicts.json')
        self.ans_size = self.ans_to_ix.__len__()
        print(' ========== Answer token vocab size:', self.ans_size)
        print('Finished!')
        print('')



    def img_feat_path_load(self, path_list):
        iid_to_path = {}

        for ix, path in enumerate(path_list):
            iid = path.split('/')[-1].split('.')[0]
            iid_to_path[iid] = path

        return iid_to_path


    # def tokenize(self, stat_ques_dict, use_glove):
    #     token_to_ix = {
    #         'PAD': 0,
    #         'UNK': 1,
    #         'CLS': 2,
    #     }
    #
    #     spacy_tool = None
    #     pretrained_emb = []
    #     if use_glove:
    #         spacy_tool = en_vectors_web_lg.load()
    #         pretrained_emb.append(spacy_tool('PAD').vector)
    #         pretrained_emb.append(spacy_tool('UNK').vector)
    #         pretrained_emb.append(spacy_tool('CLS').vector)
    #
    #     max_token = 0
    #     for qid in stat_ques_dict:
    #         ques = stat_ques_dict[qid]['question']
    #         words = re.sub(
    #             r"([.,'!?\"()*#:;])",
    #             '',
    #             ques.lower()
    #         ).replace('-', ' ').replace('/', ' ').split()
    #
    #         if len(words) > max_token:
    #             max_token = len(words)
    #
    #         for word in words:
    #             if word not in token_to_ix:
    #                 token_to_ix[word] = len(token_to_ix)
    #                 if use_glove:
    #                     pretrained_emb.append(spacy_tool(word).vector)
    #
    #     pretrained_emb = np.array(pretrained_emb)
    #
    #     return token_to_ix, pretrained_emb, max_token
    #
    #
    # def ans_stat(self, stat_ans_dict):
    #     ans_to_ix = {}
    #     ix_to_ans = {}
    #
    #     for qid in stat_ans_dict:
    #         ans = stat_ans_dict[qid]['answer']
    #         ans = prep_ans(ans)
    #
    #         if ans not in ans_to_ix:
    #             ix_to_ans[ans_to_ix.__len__()] = ans
    #             ans_to_ix[ans] = ans_to_ix.__len__()
    #
    #     return ans_to_ix, ix_to_ans


    def tokenize(self, json_file, use_glove):
        token_to_ix, max_token = json.load(open(json_file, 'r'))[2:]
        spacy_tool = None
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()

        pretrained_emb = []
        for word in token_to_ix:
            if use_glove:
                pretrained_emb.append(spacy_tool(word).vector)
        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb, max_token


    def ans_stat(self, json_file):
        ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))[:2]

        return ans_to_ix, ix_to_ans


    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_ques_ans(self, idx):

        qid = self.qid_list[idx]
        iid = self.ques_dict[qid]['imageId']

        ques = self.ques_dict[qid]['question']
        ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=self.max_token)
        ans_iter = np.zeros(1)

        if self.__C.RUN_MODE in ['train']:
            # process answers
            ans = self.ques_dict[qid]['answer']
            ans_iter = self.proc_ans(ans, self.ans_to_ix)
        

        return ques_ix_iter, ans_iter, iid


    def load_img_feats(self, idx, iid):
        frcn_feat = np.load(self.iid_to_frcn_feat_path[iid],allow_pickle=True)
        frcn_feat_iter = self.proc_img_feat(frcn_feat['x'], img_feat_pad_size=self.__C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][0])

        grid_feat = np.load(self.iid_to_grid_feat_path[iid])
        grid_feat = self.proc_img_feat(grid_feat, img_feat_pad_size=self.__C.FEAT_SIZE['gqa']['GRID_FEAT_SIZE'][0])
        grid_feat_iter = grid_feat
        #grid_feat_iter = grid_feat.astype(np.float32)
        
        #frcn_feat_bbox = frcn_feat['bbox']
        # frcn_feat_h = frcn_feat['height']
        # frcn_feat_w = frcn_feat['width']
        
        frcn_feat_h = frcn_feat['image_h']
        frcn_feat_w = frcn_feat['image_w']
        '''
        bbox_feat_iter = self.proc_img_feat(
            self.proc_bbox_feat(
                frcn_feat['bbox'],
                (frcn_feat['height'], frcn_feat['width'])
            ),
            img_feat_pad_size=self.__C.FEAT_SIZE['gqa']['BBOX_FEAT_SIZE'][0]
        )
        '''
         # for attr + object
        objects = frcn_feat['info'].item()
        objects_id = objects["objects"]        # e.g. ['car','dog',...]
        attrs = objects["attrs_id"]  # e.g. ['red','small',...]
        labels = [self.classes[i]  for i in objects_id.tolist()]
        attri = [self.attributes[i] for i in attrs[0].tolist()]
        attr_obj = list(zip(attri, labels))
        #attr_obj = [('the',o,'is', a) for a, o in zip(attri,labels)]
        # attr_obj  = [f"{a} {o}" for a, o in zip(attri,labels)]
        # attr_obj_pad = self.pad_list(attr_obj,self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0],pad_value='')
        # attr_obj_iter = self.clip_tokenizer(attr_obj_pad, truncate=True)
        
        attr_obj_iter = self.proc_attr_obj(
                attr_obj,
                word_length=2,
                max_region=self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])  # usually 100

        
        grid_bbox_feat_iter = self.make_grid_bboxes(
            image_w=int(frcn_feat['image_w']),
            image_h=int(frcn_feat['image_h']),
            S=int(math.sqrt(self.__C.FEAT_SIZE['gqa']['GRID_FEAT_SIZE'][0])),
            inclusive=True,   # 与你的像素+1约定一致
        ) 
        
        bbox_feat_iter = self.proc_img_feat(
            frcn_feat['bbox'],
            img_feat_pad_size=self.__C.FEAT_SIZE['gqa']['BBOX_FEAT_SIZE'][0]
        )
            
        w_feat_iter = frcn_feat_w
        h_feat_iter = frcn_feat_h
        
        #process region-grid-align
        align = np.load(self.iid_to_align_feat_path[iid])
        maxcov = align['maxcov']
        iou = align['iou']
        
        rg_align_mask = self.build_rg_mask(maxcov=maxcov, tau_cov=1e-8, iou = iou, tau_iou=None)
        
        region_num = rg_align_mask.shape[0]
        grid_num = rg_align_mask.shape[1]
        
        r2g_mask = self.proc_img_feat(rg_align_mask, img_feat_pad_size=self.__C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][0])
        r2g_iou = self.proc_img_feat(iou, img_feat_pad_size=self.__C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][0])
        
        region_aligns_iter = r2g_mask
        region_iou_iter = r2g_iou

        return frcn_feat_iter, grid_feat_iter, bbox_feat_iter, grid_bbox_feat_iter, w_feat_iter, h_feat_iter, region_aligns_iter, region_iou_iter, attr_obj_iter

    # ------------------------------------
    # ---- Real-Time Processing Utils ----
    # ------------------------------------

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


    def proc_bbox_feat(self, bbox, img_shape):
        bbox_feat = np.zeros((bbox.shape[0], 4), dtype=np.float32)

        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])

        return bbox_feat
        
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
    

    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix
    
    def proc_ans(self, ans, ans_to_ix):
        ans_ix = np.zeros(1, np.int64)
        ans = prep_ans(ans)
        ans_ix[0] = ans_to_ix[ans]

        return ans_ix

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


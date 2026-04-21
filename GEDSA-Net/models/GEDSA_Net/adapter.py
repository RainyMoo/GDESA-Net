from ops.fc import FC, MLP
import torch.nn as nn
import torch
from core.base_dataset import BaseAdapter
from utils.make_mask import make_mask
from ops.fc import FC, MLP
from ops.layer_norm import LayerNorm
import torch.nn.functional as F
import math

class Adapter(BaseAdapter):
    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C
          

    def bbox_proc(self,boxes, imgs_wh):
        boxes[:, :, 0] = boxes[:, :, 0] / imgs_wh[:, 0:1]
        boxes[:, :, 1] = boxes[:, :, 1] / imgs_wh[:, 1:2]
        boxes[:, :, 2] = boxes[:, :, 2] / imgs_wh[:, 0:1]
        boxes[:, :, 3] = boxes[:, :, 3] / imgs_wh[:, 1:2]                
        area = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])
        return torch.cat((boxes, area.unsqueeze(2)), -1)
    
    def vqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][1]

        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
            self.grid_bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)
        self.grid_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)


    def gqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][1]

        
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
            self.grid_bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
        self.grid_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)


    def vqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        grid_feat = feat_dict['GRID_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        grid_bbox_feat = feat_dict['GRID_BBOX_FEAT']
        region_align = feat_dict['REGION_ALIGN']  # 64 100, 64   
        region_iou = feat_dict['REGION_IOU']

        
        
        w_feat = feat_dict['W_FEAT']
        h_feat = feat_dict['H_FEAT']
        w_feat = w_feat.unsqueeze(1)
        h_feat = h_feat.unsqueeze(1)
        wh_feat = torch.cat((w_feat, h_feat), dim=-1)
        
        img_feat_mask = make_mask(frcn_feat)
        grid_feat_mask = make_mask(grid_feat)

        
        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_proc(bbox_feat, wh_feat)
            bbox_feat = self.bbox_linear(bbox_feat)
            #bbox_feat = self.bbox_linear(region_abs)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)

            grid_bbox_feat = self.bbox_proc(grid_bbox_feat, wh_feat)
            grid_bbox_feat = self.grid_bbox_linear(grid_bbox_feat)
            grid_feat = torch.cat((grid_feat, grid_bbox_feat), dim=-1)

        img_feat = self.frcn_linear(frcn_feat)
        
        grid_feat = self.grid_linear(grid_feat)


        return img_feat, img_feat_mask, grid_feat, grid_feat_mask, region_align, region_iou


    def gqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        grid_feat = feat_dict['GRID_FEAT']
        grid_bbox_feat = feat_dict['GRID_BBOX_FEAT']        
        region_align = feat_dict['REGION_ALIGN']        
        region_iou = feat_dict['REGION_IOU']
    
        w_feat = feat_dict['W_FEAT']
        h_feat = feat_dict['H_FEAT']
        w_feat = w_feat.unsqueeze(1)
        h_feat = h_feat.unsqueeze(1)        
        wh_feat = torch.cat((w_feat, h_feat), dim=-1)
        
        img_feat_mask = make_mask(frcn_feat)
        grid_feat_mask = make_mask(grid_feat)
        
        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_proc(bbox_feat, wh_feat)
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
            grid_bbox_feat = self.bbox_proc(grid_bbox_feat, wh_feat)
            grid_bbox_feat = self.grid_bbox_linear(grid_bbox_feat)
            grid_feat = torch.cat((grid_feat, grid_bbox_feat), dim=-1)

        
        img_feat = self.frcn_linear(frcn_feat)        
        grid_feat = self.grid_linear(grid_feat)

        return img_feat, img_feat_mask, grid_feat, grid_feat_mask, region_align, region_iou





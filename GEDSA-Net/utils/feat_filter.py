def feat_filter(dataset, frcn_feat, grid_feat, bbox_feat, grid_bbox_feat, w_feat, h_feat, region_align,region_iou):
    feat_dict = {}

    if dataset in ['vqa']:
        feat_dict['FRCN_FEAT'] = frcn_feat
        feat_dict['GRID_FEAT'] = grid_feat
        feat_dict['BBOX_FEAT'] = bbox_feat
        feat_dict['GRID_BBOX_FEAT'] = grid_bbox_feat
        feat_dict['W_FEAT'] = w_feat
        feat_dict['H_FEAT'] = h_feat
        feat_dict['REGION_ALIGN'] = region_align
        feat_dict['REGION_IOU'] = region_iou

    elif dataset in ['gqa']:
        feat_dict['FRCN_FEAT'] = frcn_feat
        feat_dict['GRID_FEAT'] = grid_feat
        feat_dict['BBOX_FEAT'] = bbox_feat
        feat_dict['GRID_BBOX_FEAT'] = grid_bbox_feat
        feat_dict['W_FEAT'] = w_feat
        feat_dict['H_FEAT'] = h_feat
        feat_dict['REGION_ALIGN'] = region_align
        feat_dict['REGION_IOU'] = region_iou

    elif dataset in ['clevr']:
        feat_dict['FRCN_FEAT'] = frcn_feat
        feat_dict['GRID_FEAT'] = grid_feat
        feat_dict['BBOX_FEAT'] = bbox_feat
        feat_dict['GRID_BBOX_FEAT'] = grid_bbox_feat
        feat_dict['W_FEAT'] = w_feat
        feat_dict['H_FEAT'] = h_feat
        feat_dict['REGION_ALIGN'] = region_align
        feat_dict['REGION_IOU'] = region_iou
        
    elif dataset in ['vqa_grid']:
        feat_dict['FRCN_FEAT'] = frcn_feat
        feat_dict['BBOX_FEAT'] = bbox_feat

    else:
        exit(-1)

    return feat_dict



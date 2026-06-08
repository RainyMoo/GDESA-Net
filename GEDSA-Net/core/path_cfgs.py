import os

class PATH:
    def __init__(self):
        self.init_path()
        # self.check_path()


    def init_path(self):

        self.DATA_ROOT = '../ccq_dataset/'

        self.DATA_PATH = {
            'vqa_region': self.DATA_ROOT + '/vqa_v2/feats/vqa_region/d2-152',
            'vqa':  self.DATA_ROOT + '/vqa_v2',
            'gqa': self.DATA_ROOT+ '/gqa',
            'vqa_grid': self.DATA_ROOT + '/vqa_v2/feats/vqa_grid/grid8',
        }


        self.FEATS_PATH = {
            

            'vqa': {
                'train': self.DATA_PATH['vqa_region'] + '/train2014',
                'val': self.DATA_PATH['vqa_region']  + '/val2014',
                'test': self.DATA_PATH['vqa_region'] + '/test2015',
                
            },

            'gqa': {
                'default-frcn': self.DATA_PATH['gqa'] + '/feats' + '/d2-gqa-frcn',
                'default-grid': self.DATA_PATH['gqa'] + '/feats' + '/d2-gqa-grid-144',
                'default-align': self.DATA_PATH['gqa'] + '/feats' + '/align_8',  # add align for gqa
            },
            
            
            'vqa_grid': {
                'train': self.DATA_PATH['vqa_grid'] + '/train2014',
                'val': self.DATA_PATH['vqa_grid']  + '/val2014',
                'test': self.DATA_PATH['vqa_grid']  + '/test2015',
            }

        }
        
        

        self.ALIGNS_PATH = {
            
            'vqa_8': {
                'train': self.DATA_PATH['vqa_region'] + '/align_8' + '/train2014',
                'val': self.DATA_PATH['vqa_region']+ '/align_8' + '/val2014',
                'test': self.DATA_PATH['vqa_region'] + '/align_8' + '/test2015',
            },
        }
        self.RAW_PATH = {
            'vqa': {
                'train': self.DATA_PATH['vqa'] + '/raw' + '/v2_OpenEnded_mscoco_train2014_questions.json',
                'train-anno': self.DATA_PATH['vqa'] + '/raw' + '/v2_mscoco_train2014_annotations.json',
                'val': self.DATA_PATH['vqa'] + '/raw' + '/v2_OpenEnded_mscoco_val2014_questions.json',
                'val-anno': self.DATA_PATH['vqa'] + '/raw' + '/v2_mscoco_val2014_annotations.json',
                'vg': self.DATA_PATH['vqa'] + '/raw' + '/VG_questions.json',
                'vg-anno': self.DATA_PATH['vqa'] + '/raw' + '/VG_annotations.json',
                'test': self.DATA_PATH['vqa'] + '/raw' + '/v2_OpenEnded_mscoco_test2015_questions.json',
            },
            'gqa': {
                'train': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/train_balanced_questions.json',
                'val': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/val_balanced_questions.json',
                'testdev': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/testdev_balanced_questions.json',
                'test': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/submission_all_questions.json',
                'val_all': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/val_all_questions.json',
                'testdev_all': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/testdev_all_questions.json',
                'train_choices': self.DATA_PATH['gqa'] + '/raw' + '/eval/train_choices',
                'val_choices': self.DATA_PATH['gqa'] + '/raw' + '/eval/val_choices.json',
            }
        }


        self.SPLITS = {
            'vqa': {
                'train': '',
                'val': 'val',
                'test': 'test',
            },
            'gqa': {
                'train': '',
                'val': 'testdev',
                'test': 'test',
            }
        }


        self.RESULT_PATH = './results/result_test'
        self.PRED_PATH = './results/pred'
        self.CACHE_PATH = './results/cache'
        self.LOG_PATH = './results/log'
        self.CKPTS_PATH = './ckpts'

        if 'result_test' not in os.listdir('./results'):
            os.mkdir('./results/result_test')

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')

        if 'cache' not in os.listdir('./results'):
            os.mkdir('./results/cache')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')


    def check_path(self, dataset=None):
        print('Checking dataset ........')


        if dataset:
            for item in self.FEATS_PATH[dataset]:
                if not os.path.exists(self.FEATS_PATH[dataset][item]):
                    print(self.FEATS_PATH[dataset][item], 'NOT EXIST')
                    exit(-1)
                    
            # for item in self.SPATIALS_PATH['vqa']:
            #     if not os.path.exists(self.SPATIALS_PATH['vqa'][item]):
            #         print(self.SPATIALS_PATH['vqa'][item], 'NOT EXIST')
            #         exit(-1)

            for item in self.RAW_PATH[dataset]:
                if not os.path.exists(self.RAW_PATH[dataset][item]):
                    print(self.RAW_PATH[dataset][item], 'NOT EXIST')
                    exit(-1)

        else:
            for dataset in self.FEATS_PATH:
                for item in self.FEATS_PATH[dataset]:
                    if not os.path.exists(self.FEATS_PATH[dataset][item]):
                        print(self.FEATS_PATH[dataset][item], 'NOT EXIST')
                        exit(-1)
                        
            # for dataset in self.SPATIALS_PATH:
            #     for item in self.SPATIALS_PATH[dataset]:
            #         if not os.path.exists(self.SPATIALS_PATH[dataset][item]):
            #             print(self.SPATIALS_PATH[dataset][item], 'NOT EXIST')
            #             exit(-1)

            for dataset in self.RAW_PATH:
                for item in self.RAW_PATH[dataset]:
                    if not os.path.exists(self.RAW_PATH[dataset][item]):
                        print(self.RAW_PATH[dataset][item], 'NOT EXIST')
                        exit(-1)

        print('Finished!')
        print('')


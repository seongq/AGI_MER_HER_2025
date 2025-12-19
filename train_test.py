import torch
import numpy as np
import argparse
from data_utils import *
from torch.utils.data import DataLoader, TensorDataset, Subset

import train_tva_1
import random
import os
from dataset import *
# print("test중")


if __name__ == '__main__':
    # get arguments
    p = argparse.ArgumentParser()
    # p.add_argument("--test_second", type=str, default="second")
    # p.add_argument("--test_argument", type=str, default='test')
    p.add_argument("--aux_classifier", type=str, default='tva', choices=('t', 'v', 'a', 'tv', 'ta', 'va', 'tva', "NOOO"))
    p.add_argument('--modals', type=str, default='tva', choices=('t', 'v', 'a', 'tv', 'ta', 'va', 'tva'))
    p.add_argument('--fusion', type=str, default="mean_std", help = "mean_std, summation, concatenation, FiLM (bimodal), Gated (bimodal)")
    p.add_argument('--aux_fusion_tv',type=str, help = "mean_std, summation, concatenation, FiLM (bimodal), Gated (bimodal)")
    p.add_argument('--aux_fusion_ta',type=str, help = "mean_std, summation, concatenation, FiLM (bimodal), Gated (bimodal)")
    p.add_argument('--aux_fusion_va',type=str, help = "mean_std, summation, concatenation, FiLM (bimodal), Gated (bimodal)")

    p.add_argument("--balancing", type=str, help="OGM(bimodal), OGM-GE(bimodal), PMR(bimodal), MMCOSINE")
    p.add_argument("--balancing_aux_tv", type=str, help="OGM(bimodal), OGM-GE(bimodal), PMR(bimodal), MMCOSINE")
    p.add_argument("--balancing_aux_ta", type=str, help="OGM(bimodal), OGM-GE(bimodal), PMR(bimodal), MMCOSINE")
    p.add_argument("--balancing_aux_va", type=str, help="OGM(bimodal), OGM-GE(bimodal), PMR(bimodal), MMCOSINE")

    ##############################MKD관련##################################
    
    p.add_argument("--MKD", type=str2bool, required=True)
    p.add_argument("--MKD_modals",default="tva", type=str, required=True, choices=('tv', 'ta', 'va', 'tva', 't', 'v', 'a'))
    p.add_argument("--MKD_student_modals", type=str, required=True, choices=('tv', 'ta', 'va', 'tva', 't', 'v', 'a'))    
    p.add_argument("--MULTITASK_MKD", type=str, required=True, choices=('tv', 'ta', 'va', 'tva'))
    
    
    p.add_argument('--devmode', type=str,required=True, choices=('debugmode','siljunmode'))
    p.add_argument('--normalization', type=str2bool, default=False)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument("--folder",required=True, type=str, choices=('01', '02', '03', '04', '05'))
    p.add_argument('--data_path', type=str, default='/workspace/datasets/emotion/iscross/file_per_sample')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--rnntype', type=str, default='gru')
    p.add_argument('--rnndir', type=str, default=True,
                   help='Uni (False) or Bi (True) directional')
    p.add_argument('--rnnsize', type=int, default=60)#30)#200
    # video params
    p.add_argument('--vid_rnnnum', type=int, default=1)#1)#3
    p.add_argument('--vid_rnndp', type=int, default=0.3)#0.3
    p.add_argument('--vid_rnnsize', type=int, default=60)
    p.add_argument('--vid_nh', type=int, default=6,
                        help='number of attention heads for mha')#4
    p.add_argument('--vid_dp', type=int, default=0.1,
                        help='dropout rate for mha')#0.1
    # text params
    p.add_argument('--txt_rnnnum', type=int, default=1)
    p.add_argument('--txt_rnndp', type=int, default=0.3)#0.3
    p.add_argument('--txt_rnnsize', type=int, default=60)
    p.add_argument('--txt_nh', type=int, default=6,
                   help='number of attention heads for mha')#4
    p.add_argument('--txt_dp', type=int, default=0.1,
                   help='dropout rate for mha')#0.1
    # audio params
    p.add_argument('--aud_rnnnum', type=int, default=1)
    p.add_argument('--aud_rnndp', type=int, default=0.3)  # 0.3
    p.add_argument('--aud_rnnsize', type=int, default=60)
    p.add_argument('--aud_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--aud_dp', type=int, default=0.1,
                   help='dropout rate for mha')  # 0.1
    # tv params
    p.add_argument('--tv_nh', type=int, default=6,
                   help='number of attention heads for mha')#4
    p.add_argument('--tv_dp', type=int, default=0.1,
                   help='dropout rate for mha')#0.1
    # ta params
    p.add_argument('--ta_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--ta_dp', type=int, default=0.1,
                   help='dropout rate for mha')  # 0.1
    # vt params
    p.add_argument('--vt_nh', type=int, default=6,
                   help='number of attention heads for mha')#4
    p.add_argument('--vt_dp', type=int, default=0.1,
                   help='dropout rate for mha')
    # va params
    p.add_argument('--va_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--va_dp', type=int, default=0.1,
                   help='dropout rate for mha')
    # at params
    p.add_argument('--at_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--at_dp', type=int, default=0.1,
                   help='dropout rate for mha')
    # av params
    p.add_argument('--av_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--av_dp', type=int, default=0.1,
                   help='dropout rate for mha')

    # tf params
    p.add_argument('--tf_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--tf_dp', type=int, default=0.1,
                   help='dropout rate for mha')

    # vf params
    p.add_argument('--vf_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--vf_dp', type=int, default=0.1,
                   help='dropout rate for mha')

    # af params
    p.add_argument('--af_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--af_dp', type=int, default=0.1,
                   help='dropout rate for mha')

    p.add_argument('--output_dim', type=int, default=7,
                        help='number of classes')
    p.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    
    p.add_argument("--text_feature", type=str, default='glove_from_iscross')
    p.add_argument("--video_feature", type=str, default = "resnet_from_iscross")
    p.add_argument("--audio_feature", type=str, default="mfcc_from_iscross")
    
    params = p.parse_args()
    #seed = 123
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(params.seed)
    
    assert (set(params.aux_classifier).issubset(set(params.modals)))  

    
    p.add_argument("--MKD", type=str2bool, required=True)
    p.add_argument("--MKD_modals", type=str, required=True, choices=('tv', 'ta', 'va', 'tva'))
    p.add_argument("--MKD_student_modals", type=str, required=True, choices=('tv', 'ta', 'va', 'tva'))    
    p.add_argument("--MULTITASK_MKD", type=str, required=True, choices=('tv', 'ta', 'va', 'tva'))

    # get train data
    print("실행중임")
    
    
    # 
    
    
    
    id_to_numberid_path = "/workspace/datasets/data-processed-icassp-20/IEMOCAP/seven_category_120/seven_C_id_label.txt"

    ids_number_in_folder = f"/workspace/datasets/data-processed-icassp-20/IEMOCAP/seven_category_120/folds/fold{params.folder}/fold{params.folder}_id.txt"

    with open(id_to_numberid_path,'r') as f:
        lines = f.readlines()

    str_ids = [line.strip().split('\t')[0] for line in lines]

 
    
    
    with open(ids_number_in_folder, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    train_ids = lines[0].split(" ")
    train_ids = [int(id) for id in train_ids]
    dev_ids = lines[1].split(" ")
    dev_ids = [int(id) for id in dev_ids]
    test_ids = lines[2].split(" ")
    test_ids = [int(id) for id in test_ids]

    ids = {}

    ids['train'] = train_ids
    ids['dev'] = dev_ids
    ids['test'] = test_ids

    modes = ['train', 'dev', 'test']
    
    train_folders = [str_ids[i] for i in ids['train']]
    dev_folders =  [str_ids[i] for i in ids['dev']]
    test_folders =  [str_ids[i] for i in ids['test']]
    if params.devmode == "debugmode":
        train_folders = train_folders[0:10]
        dev_folders = dev_folders[0:10]
        test_folders = test_folders[0:10]
        

    if params.normalization:

        from sklearn.preprocessing import StandardScaler
        scaler_audio = StandardScaler()


        print("train set 불러오기")
        TRAIN_AUDIOFEATURES = []
        TRAIN_VIDEOFEATURES = []
        TRAIN_TEXTFEATURES = []
        TRAIN_LABELS = []
        for i,name in enumerate(train_folders):
            if i % 100 == 0:
                print(f'{i} 번째 불러오기')
            audiofeature = np.load(os.path.join(params.data_path, name, 'audio',params.audio_feature, 'sample.npy'))
            videofeature = np.load(os.path.join(params.data_path, name, 'video',params.video_feature, 'sample.npy'))
            textfeature = np.load(os.path.join(params.data_path, name, 'text',params.text_feature, 'sample.npy'))
            with open(os.path.join(params.data_path, name, "label", 'sample.txt')) as f:
                label = f.read()
                label = int(label)
                # print(label)
            
            TRAIN_LABELS.append(label)
                
            TRAIN_AUDIOFEATURES.append(audiofeature)
            TRAIN_VIDEOFEATURES.append(videofeature)
            TRAIN_TEXTFEATURES.append(textfeature)
        TRAIN_AUDIOFEATURES = np.stack(TRAIN_AUDIOFEATURES, axis=0)
        TRAIN_VIDEOFEATURES = np.stack(TRAIN_VIDEOFEATURES, axis=0)
        TRAIN_TEXTFEATURES = np.stack(TRAIN_TEXTFEATURES, axis=0)
        TRAIN_LABELS = np.stack(TRAIN_LABELS,axis=0)
    
    
        print("dev set 불러오는중")
        dev_AUDIOFEATURES = []
        dev_VIDEOFEATURES = []
        dev_TEXTFEATURES = []
        dev_LABELS = []
        for name in dev_folders:
            audiofeature = np.load(os.path.join(params.data_path, name, 'audio',params.audio_feature, 'sample.npy'))
            videofeature = np.load(os.path.join(params.data_path, name, 'video',params.video_feature, 'sample.npy'))
            textfeature = np.load(os.path.join(params.data_path, name, 'text',params.text_feature, 'sample.npy'))
            with open(os.path.join(params.data_path, name, "label", 'sample.txt')) as f:
                label = f.read()
                label = int(label)
                # print(label)
            
            dev_LABELS.append(label)
                
            dev_AUDIOFEATURES.append(audiofeature)
            dev_VIDEOFEATURES.append(videofeature)
            dev_TEXTFEATURES.append(textfeature)
        dev_AUDIOFEATURES = np.stack(dev_AUDIOFEATURES, axis=0)
        dev_VIDEOFEATURES = np.stack(dev_VIDEOFEATURES, axis=0)
        dev_TEXTFEATURES = np.stack(dev_TEXTFEATURES, axis=0)
        dev_LABELS = np.stack(dev_LABELS,axis=0)
        print("dev set 불러옴")


        print('test set 불러오는중')
    
        
        test_AUDIOFEATURES = []
        test_VIDEOFEATURES = []
        test_TEXTFEATURES = []
        test_LABELS = []
        for name in test_folders:
            # print('name 불렁오기')
            audiofeature = np.load(os.path.join(params.data_path, name, 'audio',params.audio_feature, 'sample.npy'))
            videofeature = np.load(os.path.join(params.data_path, name, 'video',params.video_feature, 'sample.npy'))
            textfeature = np.load(os.path.join(params.data_path, name, 'text',params.text_feature, 'sample.npy'))
            with open(os.path.join(params.data_path, name, "label", 'sample.txt')) as f:
                label = f.read()
                label = int(label)
                # print(label)
            
            test_LABELS.append(label)
                
            test_AUDIOFEATURES.append(audiofeature)
            test_VIDEOFEATURES.append(videofeature)
            test_TEXTFEATURES.append(textfeature)
        test_AUDIOFEATURES = np.stack(test_AUDIOFEATURES, axis=0)
        test_VIDEOFEATURES = np.stack(test_VIDEOFEATURES, axis=0)
        test_TEXTFEATURES = np.stack(test_TEXTFEATURES, axis=0)
        test_LABELS = np.stack(test_LABELS,axis=0)
        
        print('test set 불러옴')
        
        s1 = TRAIN_AUDIOFEATURES.shape[1]
        s2 = TRAIN_AUDIOFEATURES.shape[2]
        TRAIN_AUDIOFEATURES = np.reshape(TRAIN_AUDIOFEATURES, [TRAIN_AUDIOFEATURES.shape[0], -1])
        
        print('scaler 작업중')
        scaler_audio.fit(TRAIN_AUDIOFEATURES)
        
        
        scaler_audio.transform(TRAIN_AUDIOFEATURES)
        TRAIN_AUDIOFEATURES = np.reshape(TRAIN_AUDIOFEATURES, [TRAIN_AUDIOFEATURES.shape[0], s1,s2])
        
        print('trainset scaler로 작업완료')
        train_dataset = TensorDataset(torch.Tensor(TRAIN_TEXTFEATURES).float().to('cuda'),
                                    torch.Tensor(TRAIN_VIDEOFEATURES).float().to('cuda'),
                                    torch.Tensor(TRAIN_AUDIOFEATURES).float().to('cuda'),
                                    torch.Tensor(TRAIN_LABELS).long().to('cuda'))
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
        
        
        
        
        dev_AUDIOFEATURES = np.reshape(dev_AUDIOFEATURES, [dev_AUDIOFEATURES.shape[0], -1])
        scaler_audio.transform(dev_AUDIOFEATURES)
        dev_AUDIOFEATURES = np.reshape(dev_AUDIOFEATURES, [dev_AUDIOFEATURES.shape[0], s1,s2])
        print('dev set scaler로 작업완료')
        dev_dataset = TensorDataset(torch.Tensor(dev_TEXTFEATURES).float().to('cuda'),
                                    torch.Tensor(dev_VIDEOFEATURES).float().to('cuda'),
                                    torch.Tensor(dev_AUDIOFEATURES).float().to('cuda'),
                                    torch.Tensor(dev_LABELS).long().to('cuda'))
        dev_loader = DataLoader(dev_dataset, batch_size=params.batch_size, shuffle=False)
        
        
        test_AUDIOFEATURES = np.reshape(test_AUDIOFEATURES, [test_AUDIOFEATURES.shape[0], -1])
        scaler_audio.transform(test_AUDIOFEATURES)
        test_AUDIOFEATURES = np.reshape(test_AUDIOFEATURES, [test_AUDIOFEATURES.shape[0], s1,s2])
        print('test set scaler로 작업완료')
        test_dataset = TensorDataset(torch.Tensor(test_TEXTFEATURES).float().to('cuda'),
                                    torch.Tensor(test_VIDEOFEATURES).float().to('cuda'),
                                    torch.Tensor(test_AUDIOFEATURES).float().to('cuda'),
                                    torch.Tensor(test_LABELS).long().to('cuda'))
        test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)
        
        
        
        
        
        
        
        
        params.n_train = len(TRAIN_TEXTFEATURES)

    
        params.n_dev = len(dev_TEXTFEATURES)
        
        params.n_test = len(test_TEXTFEATURES)

        
        params.num_epochs = 2000 # give a random big number
        params.when = 10 # reduce LR patience
        params.txt_dim = 300
        params.vid_dim = 2048
        params.aud_dim = 120
        params.pros_dim = 35
        count = 0

        print("학습시작")
        test_loss = train_tva_1.initiate(params, train_loader, dev_loader, test_loader)    
    else:
        print('pass')
        # label index -> folder name 매핑
        with open(id_to_numberid_path, 'r') as f:
            str_ids = [line.strip().split('\t')[0] for line in f.readlines()]

        # fold 분할
        with open(ids_number_in_folder, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            train_ids = [int(i) for i in lines[0].split()]
            dev_ids = [int(i) for i in lines[1].split()]
            test_ids = [int(i) for i in lines[2].split()]

        train_folders = [str_ids[i] for i in train_ids]
        dev_folders   = [str_ids[i] for i in dev_ids]
        test_folders  = [str_ids[i] for i in test_ids]

        # Dataset & DataLoader 생성
        train_dataset = MultimodalIEMOCAPDataset(train_folders, params.data_path, params)
        dev_dataset   = MultimodalIEMOCAPDataset(dev_folders,   params.data_path, params)
        test_dataset  = MultimodalIEMOCAPDataset(test_folders,  params.data_path, params)


        if params.devmode == "debugmode":
            train_dataset = Subset(train_dataset, list(range(10)))
            dev_dataset = Subset(dev_dataset, list(range(10)))
            test_dataset = Subset(test_dataset, list(range(10)))


        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)
        dev_loader   = DataLoader(dev_dataset,   batch_size=params.batch_size, shuffle=False, num_workers=4)
        test_loader  = DataLoader(test_dataset,  batch_size=params.batch_size, shuffle=False, num_workers=4)
        
        params.n_train = len(train_dataset)
        params.n_dev = len(dev_dataset)
        params.n_test = len(test_dataset)


        params.num_epochs = 200000 # give a random big number
        params.when = 10 # reduce LR patience
        params.txt_dim = 300
        params.vid_dim = 2048
        params.aud_dim = 120
        params.pros_dim = 35  # 쓰이지 않으면 제거해도 됨
        count = 0

        print("학습시작")
        test_loss = train_tva_1.initiate(params, train_loader, dev_loader, test_loader)
            
    
    
    
    # sessions = ['01', '02', '03', '04', '05']
    # sessions.remove(params.test_session)
    
    # dev_session = random.choice(sessions)
    # # print(sessions)
    # sessions.remove(dev_session)
    
    # # print(dev_session)
    # # print(sessions)
    # import os
    # test_session = params.test_session
    # train_sessions = sessions
    
    # assert not (test_session in train_sessions)
    # assert not (dev_session in train_sessions)
    # assert not (dev_session == test_session)
    # subfolders = [name for name in os.listdir(params.data_path)
    #           if os.path.isdir(os.path.join(params.data_path, name))]

    # train_folders = [name for name in subfolders if name[3:5] in train_sessions ]
    # dev_folders =  [name for name in subfolders if name[3:5] == dev_session]
    # test_folders =  [name for name in subfolders if name[3:5] == test_session]
    
import os
import sys
from transformers import AdamW
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from pytorch_metric_learning.losses import NTXentLoss
import numpy as np
import argparse 
from tqdm import tqdm
from datasets.patch_dataset import PatchDataset
import pdb
import pickle
import PIL
from const import map_tiles_dict
from tqdm import tqdm


DEBUG = False

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def load_model(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load a pretrained model and reset final fully connected layer.
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    num_ftrs = model.fc.in_features

    model.fc = torch.nn.Linear(num_ftrs, args.proj_dim) 

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

def extract_features(args, model,  cur_dir, jpg_list):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    

    embeddings_list = []
    for jpg_path in jpg_list:
        img_path = os.path.join(cur_dir, jpg_path)
        img = PIL.Image.open(img_path)
        img = preprocess(img).to(device)

        embed = model(torch.unsqueeze(img,0))
        embeddings_list.append(embed.detach().cpu().numpy())
        
        # pdb.set_trace()
    embeddings_list = np.concatenate(embeddings_list)
    return embeddings_list
   
def run_one_map_inner_tiles(args, model, external_id):
    cur_dir = os.path.join(args.crop_patch_dir, external_id)
    jpg_list = sorted(os.listdir(cur_dir))

    # crop folder might be empty. if empty, get a random image folder
    if len(jpg_list) == 0:
        return 

    h_max, w_max = jpg_list[-1].split('.jpg')[0].split('_')
    h_max, w_max = int(h_max[1:]), int(w_max[1:])

 
    # get one random patch from the map image dir 
    if h_max > 1 and w_max > 1: 
        inner_jpg_list = []
        for i in range(1, h_max):
            for j in range(1, w_max):
                inner_jpg_list.append('h'+str(i) + '_w'+str(j) + '.jpg')

        jpg_list = inner_jpg_list



    embeddings_list = extract_features(args, model, cur_dir, jpg_list)   

    return embeddings_list 


def run_one_map_all_tiles(args, model, external_id):
    # external_id = '12240034' # '6903219' # '1773092'
    cur_dir = os.path.join(args.crop_patch_dir, external_id)
    jpg_list = sorted(os.listdir(cur_dir))


    embeddings_list = extract_features(args, model, cur_dir, jpg_list)   

    return embeddings_list 


def run_one_map_selected_tiles(args, model,  external_id, tile_list):
    cur_dir = os.path.join(args.crop_patch_dir, external_id)
    # use selected  tiles
    jpg_list = [a+'.jpg' for a in tile_list]


    embeddings_list = extract_features(args, model, cur_dir, jpg_list)

    return embeddings_list


def test1(args, model):
    # contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning
    # https://arxiv.org/pdf/2002.05709.pdf


    for external_id, map_tile_list in map_tiles_dict.items():

        embeddings_list = run_one_map_selected_tiles(args, model, external_id, map_tile_list)
        
        # print(embedding_list1.shape, embedding_list1.dtype)
        with open('/data2/rumsey_clustering/analyze2/embedding_'+external_id+'.pkl', 'wb') as f:
            pickle.dump({'embed':embeddings_list}, f)

def test2(args, model):

    for external_id in ['12054009','12054010','12054011','12054012']:
        embeddings_list = run_one_map_all_tiles(args, model, external_id)
        
        # print(embedding_list1.shape, embedding_list1.dtype)
        with open('/data2/rumsey_clustering/analyze3/embedding_'+external_id+'.pkl', 'wb') as f:
            pickle.dump({'embed':embeddings_list}, f)        

def test3(args, model):
    # get external_id from crop folder
    # call run_all_maps_inner_tiles with external_id as input arg 
    # check return value, if not None,  then no cropped image. otherwise save the output to pkl 
    external_id_list = os.listdir(args.crop_patch_dir)

    print('num of maps', len(external_id_list))
    
    for i in tqdm(range(len(external_id_list))): 
        external_id = external_id_list[i]
        embeddings_list = run_one_map_inner_tiles(args,model, external_id)
        
        if embeddings_list is not None:
            # print(embedding_list1.shape, embedding_list1.dtype)
            with open('/data2/rumsey_clustering/all-maps/embedding_'+external_id+'.pkl', 'wb') as f:
                pickle.dump({'embed':embeddings_list}, f)      
        
           
       
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=2000)

    parser.add_argument('--proj_dim', type=int, default=256)

    

    parser.add_argument('--lr', type=float, default = 1e-3)

   
    # parser.add_argument('--with_type', default=False, action='store_true')

    parser.add_argument('--crop_patch_dir',type=str, default = '/data2/rumsey_output/57k_maps_r2/crop')
    parser.add_argument('--model_save_dir', type=str, default='/data2/rumsey_clustering/weights')
    parser.add_argument('--checkpoint_path', type=str, default='/data2/rumsey_clustering/weights/ep194_iter00706_0.0433.pth')
    

    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')


    # out_dir not None, and out_dir does not exist, then create out_dir
    if args.model_save_dir is not None and not os.path.isdir(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    model = load_model(args)
    test3(args, model)

if __name__ == '__main__':
    main()


# CUDA_VISIBLE_DEVICES='0' python train.py  --batch_size=64
# # watch -n 1 nvidia-smi


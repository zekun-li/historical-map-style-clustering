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

DEBUG = False

# TODO
# Transforms ( flipping)
# set temperature
# hard negative sampling
# validation
# freeze/unfreeze backbone 


def training(args):
    # contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning
    # https://arxiv.org/pdf/2002.05709.pdf

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    train_val_dataset = PatchDataset(crop_patch_dir = args.crop_patch_dir, 
                                            transform = preprocess)

    percent_80 = int(len(train_val_dataset) * 0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [percent_80, len(train_val_dataset) - percent_80])

    train_loader = DataLoader(train_dataset, batch_size= args.batch_size, num_workers=args.num_workers,
                                shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size= args.batch_size, num_workers=args.num_workers,
                                shuffle=False, pin_memory=True, drop_last=False)

    # img_path1 = os.path.join(cur_dir, 'h'+str(h_randoms[0]) + '_' + 'w' + str(w_randoms[0]))+'.jpg'
    # img1 = PIL.Image.open(img_path1)

    # img_path2 = os.path.join(cur_dir, 'h'+str(h_randoms[1]) + '_' + 'w' + str(w_randoms[1]))+'.jpg'
    # img2 = PIL.Image.open(img_path2)


    # if self.transform:
    #     img1 = self.transform(img1)
    #     img2 = self.transform(img2)

    # return {'img1':img1, 'img2':img2}


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load a pretrained model and reset final fully connected layer.
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    num_ftrs = model.fc.in_features
    # print(num_ftrs) # 2048
    model.fc = torch.nn.Linear(num_ftrs, args.proj_dim) 

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    
    # setup loop with TQDM and dataloader
    loop = tqdm(val_loader, leave=True)
    iter = 0
    embeddings_list1 = []
    embeddings_list2 = []
    map_idx_list = []
    for batch in loop:
        # pull all tensor batches required for training
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        map_idx = batch['idx']


        embed1 = model(img1)
        embed2 = model(img2)

        # embeddings = torch.cat((embed1, embed2))
        # indices = torch.arange(0, embed1.size(0), device=embed1.device)
        # labels = torch.cat((indices, indices))
        # print(embeddings.shape, labels.shape) # 128, 256; [128]

        embeddings_list1.append(embed1.detach().cpu().numpy())
        embeddings_list2.append(embed2.detach().cpu().numpy())
        map_idx_list.append(map_idx)
        
        # pdb.set_trace()
    embeddings_list1 = np.concatenate(embeddings_list1)
    embeddings_list2 = np.concatenate(embeddings_list2)
    map_idx_list = [item for sublist in map_idx_list for item in sublist]
    # pdb.set_trace()
       
    # print(embedding_list1.shape, embedding_list1.dtype)
    with open('/data2/rumsey_clustering/analyze/embeddings.pkl', 'wb') as f:
        pickle.dump({'embed1':embeddings_list1, 'embed2':embeddings_list2, 'map_idx':map_idx_list}, f)
        
        
# def validating(val_loader, model, device):

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

    training(args)

if __name__ == '__main__':
    main()


# CUDA_VISIBLE_DEVICES='0' python train.py  --batch_size=64
# # watch -n 1 nvidia-smi


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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load a pretrained model and reset final fully connected layer.
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    # print(num_ftrs) # 2048
    model.fc = torch.nn.Linear(num_ftrs, args.proj_dim) 

    # initialize optimizer
    optim = AdamW(model.parameters(), lr = args.lr)

    model.to(device)
    model.train()

    criterion = NTXentLoss(temperature=0.10)

    for epoch in range(args.epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(train_loader, leave=True)
        iter = 0
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)


            embed1 = model(img1)
            embed2 = model(img2)

            embeddings = torch.cat((embed1, embed2))
            indices = torch.arange(0, embed1.size(0), device=embed1.device)
            labels = torch.cat((indices, indices))

            
            loss = criterion(embeddings, labels)

            loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix({'loss':loss.item()})
            
            if DEBUG:
                print('ep'+str(epoch)+'_' + '_iter'+ str(iter).zfill(5), loss.item() )

            iter += 1

            if iter % args.save_interval == 0 or iter == loop.total:
            #     loss_valid = validating(val_loader, model, device)

                save_path = os.path.join(args.model_save_dir, 'ep'+str(epoch) + '_iter'+ str(iter).zfill(5) \
                + '_' +str("{:.4f}".format(loss.item())) +'.pth' )

                # save_path = os.path.join(model_save_dir, 'ep'+str(epoch) + '_iter'+ str(iter).zfill(5) \
                # + '_' +str("{:.4f}".format(loss.item())) + '_val' + str("{:.4f}".format(loss_valid)) +'.pth' )

                torch.save(model.state_dict(), save_path)
            #     print('validation loss', loss_valid)
                print('saving model checkpoint to', save_path)


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


# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:40:56 2021

@author: Michael
"""
import os
import sys
import random
from itertools import permutations
import os.path as osp
import argparse
from datetime import datetime

# Add src directory to Python path before importing diffusion_net
sys.path.append(osp.join(os.getcwd(),'src'))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import scipy.io as sio

import diffusion_net
from matching_dataset import MatchingDataset
# from torch_geometric.nn import knn
import scipy
import scipy.sparse.linalg as sla
from diffusion_net.utils import dist_mat, nn_search

#shrec19
def calculate_geodesic_error_1(dist_x, corr_x,  p2p, return_mean=True):
    ind21 = np.stack([corr_x, p2p], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = np.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err
# erro
#faust、scape、smal、tosca、topkids、dt4d
def calculate_geodesic_error(dist_x, corr_x, corr_y, p2p, return_mean=True):
    ind21 = np.stack([corr_x, p2p[corr_y]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = np.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err

# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
args = parser.parse_args()


# system things
device = torch.device('cuda:0')
dtype = torch.float32


# model 
# input_features = args.input_features # one of ['xyz', 'hks']
input_features = 'hks'
k_eig = 128

# training settings
train =not args.evaluate
n_epoch = 5
lr = 1e-3

# Important paths
base_path = osp.dirname(__file__)
dataset_path = osp.join(base_path, 'data','smal')
pretrain_path = osp.join(" ".format(input_features))



# Load the train dataset
if train:
    train_dataset = MatchingDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    now = datetime.now()
    folder_str = now.strftime("%Y_%m_%d__%H_%M_%S")
    model_save_dir=osp.join(dataset_path,'save_models',folder_str)
    diffusion_net.utils.ensure_dir_exists(model_save_dir)

# === Create the model

C_in={'xyz':3, 'hks':16, 'wks':128}[input_features] # dimension of input features



model = diffusion_net.layers.MDND(C_in=C_in,C_out=256)

model = model.to(device)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train_epoch(epoch):
    # Set model to 'train' mode

    if epoch > 4 and epoch % 1 == 0:
        global lr 
        lr *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 

    global iter
    global min_erro
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0.0
    total_num = 0
    loss_history=[]
    for data in tqdm(train_loader):
        # for _ in range(1000):
            # Get data
            descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
                descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y=data
            
            # Move to device
            descs_x=descs_x.to(device)
            # verts_x=descs_x.float()
            massvec_x=massvec_x.to(device)
            evals_x=evals_x.to(device)
            evecs_x=evecs_x.to(device)
            gs_x=gs_x.to(device)
            gradX_x=gradX_x.to(device) 
            gradY_x=gradY_x.to(device) #[N,N]
            elevals_x=elevals_x.to(device)
            elevecs_x=elevecs_x.to(device)

            descs_y=descs_y.to(device)
            # descs_y=descs_y.float()
            massvec_y=massvec_y.to(device)
            evals_y=evals_y.to(device)
            evecs_y=evecs_y.to(device)
            gs_y=gs_y.to(device)
            gradX_y=gradX_y.to(device)
            gradY_y=gradY_y.to(device)
            elevals_y=elevals_y.to(device)
            elevecs_y=elevecs_y.to(device)

            # Apply the model
            loss= model(descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
                descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y)


          
            loss.requires_grad_(True)
            loss.backward()

            # print(feat_x.grad)


            # track accuracy
            total_loss+=loss.item()
            # loss_history.append(loss.item())
            total_num += 1
            iter+=1

            # Step the optimizer
            optimizer.step()
            optimizer.zero_grad()


            # optimizer1.step()
            # optimizer1.zero_grad()
            if total_num%100==0:
                print('Iterations: {:02d}, train loss: {:.4f}'.format(total_num, total_loss / total_num))
                total_loss=0.0
                total_num=0
          
   
            if iter%1==0:
                avg_erro=test()
                print(avg_erro)
                model_save_path = osp.join(model_save_dir, 'ckpt_ep{best}.pth')
                if avg_erro < min_erro:
                        torch.save(model.state_dict(), model_save_path)
                        min_erro = avg_erro 
        

def test():
    test_dataset = MatchingDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True)
    test_loader = DataLoader(test_dataset, batch_size=None)


    model.eval()
    with torch.no_grad():
        count=0
        erro=0
        # count_1=0
        for data in tqdm(test_loader):
            # Get data
        
            descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
                descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y=data


    


            # Move to device
            # verts_x=verts_x.to(device)
            descs_x=descs_x.to(device)
            # verts_x=descs_x.float()
            massvec_x=massvec_x.to(device)
            evecs_x=evecs_x.to(device)
            evals_x=evals_x.to(device)
            gs_x=gs_x.to(device)
            gradX_x=gradX_x.to(device)
            gradY_x=gradY_x.to(device) #[N,N]
            # labels_x=labels_x.to(device)
            # L_x=L_x.to(device)
            elevals_x=elevals_x.to(device)
            elevecs_x=elevecs_x.to(device)

            # verts_y=verts_y.to(device)
            descs_y=descs_y.to(device)
            # descs_y=descs_y.float()
            massvec_y=massvec_y.to(device)
            evecs_y=evecs_y.to(device)
            evals_y=evals_y.to(device)
            gs_y=gs_y.to(device)
            gradX_y=gradX_y.to(device)
            gradY_y=gradY_y.to(device)
            # labels_y=labels_y.to(device)
            # L_y=L_y.to(device)
            elevals_y=elevals_y.to(device)
            elevecs_y=elevecs_y.to(device)

            # Apply the model
            p12= model.model_test_opt(descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
                               descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y)
          
            p12=p12.cpu()
            p12=np.array(p12)
        
            data_x=sio.loadmat(""+name_y+'.mat')
            dist_x=data_x["dist"]

            corr_x_file=""+name_y+'.vts'
            arrays_corr_x = []
            with open(corr_x_file, 'r') as file:
               for line in file:
                 numbers = int(line.strip())
                 arrays_corr_x.append(numbers)
            arrays_corr_x = [x-1 for x in arrays_corr_x]
            arrays_corr_x = np.array( arrays_corr_x)

            corr_y_file=""+name_x+'.vts'
            arrays_corr_y = []
            with open(corr_y_file, 'r') as file:
               for line in file:
                 numbers = int(line.strip())
                 arrays_corr_y.append(numbers)
            arrays_corr_y = [x-1 for x in arrays_corr_y]
            arrays_corr_y = np.array( arrays_corr_y)
            erro_i=calculate_geodesic_error(dist_x, arrays_corr_x,  arrays_corr_y,p12, return_mean=True)

            count+=1
            erro=erro+erro_i

        avg_erro=erro/count
 
        return avg_erro

if train:
    print("Training...")
    iter=0
    min_erro=100
    # min_loss=1e10
    for epoch in range(n_epoch):
        # torch.cuda.empty_cache()
        # start_time = time.time()
        train_epoch(epoch)

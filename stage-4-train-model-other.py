import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import sys
from models import get_models
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from numpy.random import shuffle
from sklearn.model_selection import KFold


def main(args):
    
    class Concatdataset(torch.utils.data.Dataset):
        def __init__(self, *datasets):
            self.datasets = datasets

        def __getitem__(self, i):
            return tuple(d[i] for d in self.datasets)

        def __len__(self):
            return min(len(d) for d in self.datasets)

    
    ## HYPERPARAMETER CELL 
    subject= int(args[1])
    frame_skip = 5
    model_num = 1
    
    print('Subject:',subject,'model num:',model_num)

    sequence_len = int(90/frame_skip)
    h5_filename = '../data/bw-'+'sub-'+str(subject)+'-len-'+str(sequence_len)+'.h5'#data file to train
    store_dir = '../models/'+'sub-'+str(subject)+'-model-'+str(model_num)+'/' #dir to store trained 
    
    other1_path = '../data/bw-'+'sub-'+str(1)+'-len-'+str(sequence_len)+'.h5'
    other2_path = '../data/bw-'+'sub-'+str(2)+'-len-'+str(sequence_len)+'.h5'
    other4_path = '../data/bw-'+'sub-'+str(4)+'-len-'+str(sequence_len)+'.h5'
    other6_path = '../data/bw-'+'sub-'+str(6)+'-len-'+str(sequence_len)+'.h5'
    other_paths = [other1_path,other2_path,other4_path,other6_path]
    
    
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    load_model = False 
    if load_model:
        load_epoch = 2000
        LOAD_PATH = store_dir+str(load_epoch)+'.torch'


    ###############################################


    f = h5py.File(h5_filename,'r')
    data = f['state']
    label = f['action']
    data_all = np.array(data)
    label_all = np.array(label)
    temp_labels = label_all.reshape(-1)
    main_dataset = Concatdataset(data_all,label_all)
    

    #########################
    test_datasets = []
    f = h5py.File(other_paths[0],'r')
    data = f['state']
    label = f['action']
    data_all0 = np.array(data)
    label_all0 = np.array(label)
    dataset0 = Concatdataset(data_all0,label_all0)
    test_datasets.append(dataset0)
    
    f = h5py.File(other_paths[1],'r')
    data = f['state']
    label = f['action']
    data_all1 = np.array(data)
    label_all1 = np.array(label)
    dataset1 = Concatdataset(data_all1,label_all1)
    test_datasets.append(dataset1)
    
    f = h5py.File(other_paths[2],'r')
    data = f['state']
    label = f['action']
    data_all2 = np.array(data)
    label_all2 = np.array(label)
    dataset2 = Concatdataset(data_all2,label_all2)
    test_datasets.append(dataset2)
    
    f = h5py.File(other_paths[3],'r')
    data = f['state']
    label = f['action']
    data_all3 = np.array(data)
    label_all3 = np.array(label)
    dataset3 = Concatdataset(data_all3,label_all3)
    test_datasets.append(dataset3)
    
    ######################################
    
    
    # training hyperparameters
    batch_size = 50
    learning_rate = 0.00005
    print_every = 100
    num_epochs = 3100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ###############
    train_loader = DataLoader(dataset=main_dataset, shuffle=True, batch_size=batch_size)
    
    
    test_loaders = []
    for i in range(4):
        loader = DataLoader(dataset=test_datasets[i], shuffle=True, batch_size=batch_size)
        test_loaders.append(loader)
    
    ################
    model = get_models(model_num,sequence_len)
    model = model.to(device)
    
    h,b = np.histogram(temp_labels,bins=64,range=(0,64))
    cutoff = 20
    scaling_factor=np.ones(64)
    for i in range(64):
        if h[i]>cutoff:
            factor = h[i]/cutoff
            scaling_factor[i] = 1/factor


    weight=torch.from_numpy(scaling_factor).float().to(device=device)
    criterion = nn.NLLLoss(weight=weight)
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)


    # In[28]:


    if load_model:
        saved_state = torch.load(LOAD_PATH)
        model.load_state_dict(saved_state)
        print('Loaded stored state',flush=True)


    # In[29]:

    if load_model:
        start_epoch = load_epoch
    else:
        start_epoch = 0
        
        
    best_accu = np.zeros(4)
    best_epoch = np.zeros(4)
    
    for epoch in range(start_epoch,num_epochs):
        for batchid,traindata in enumerate(train_loader):

            data,labels = traindata
            model.train()
            optimizer.zero_grad()
            model.zero_grad()
            labels = labels.reshape(-1).long().to(device=device)
            data = data.float().to(device=device)

            out,_ = model(data)
            loss = criterion(out,labels)
            loss.backward()
            optimizer.step()

        if (epoch)%print_every==0:
            model.eval()
            
            for sub in range(4):
                test_loader = test_loaders[sub]
                
                test_accuracy=0.0
                test_loss=0.0
                with torch.no_grad():
                    for batchid,traindata in enumerate(test_loader):
                        data,labels = traindata
                        labels = labels.reshape(-1).long().to(device=device)
                        data = data.float().to(device=device)
                        out,_ = model(data)
                        probs = torch.exp(out)                
                        top_p, top_class = probs.topk(1, dim=1)
                        loss = criterion(out, labels)
                        test_loss += loss.item()
                        equals = top_class == labels.view(*top_class.shape)
                        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                test_accuracy=test_accuracy/len(test_loader)
                test_loss=test_loss/len(test_loader)
                
                if test_accuracy>best_accu[sub]:
                    best_accu[sub] = test_accuracy
                    best_epoch[sub] = epoch
            

            with torch.cuda.device(0):
                state_to_save = model.state_dict()
                storename= store_dir+str(epoch)+'.torch'
                torch.save(state_to_save,storename)
                
    np.save('../temp_files/sub-'+str(subject)+'-stage-4-best-accu.npy',best_accu)
    np.save('../temp_files/sub-'+str(subject)+'-stage-4-best-epoch.npy',best_epoch)

if __name__ == '__main__':
    main(sys.argv[:]) 
    

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as Data
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
from sklearn.model_selection import KFold,LeaveOneGroupOut


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
    validation_fold = int(args[2])
    
    frame_skip = 5
    model_num = 1
    
    print('Subject:',subject,'model num:',model_num)

    sequence_len = int(90/frame_skip)
    h5_filename = '../data/bw-'+'sub-'+str(subject)+'-len-'+str(sequence_len)+'.h5'#data file to train
    store_dir = '../models/'+'sub-'+str(subject)+'-model-'+str(model_num)+'-set-'+str(validation_fold)+'/' #dir to store  
    df_filepath_stage1 = '../temp_files/'+'sub-'+str(subject)+'-stage-1-df.pkl'
    runshape_file_path = '../temp_files/'+'sub-'+str(subject)+'-stage-2-runshapes-'+str(sequence_len)+'.pkl'

    
    
    
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
    session = f['session']

    data_all = np.array(data)
    label_all = np.array(label)
    session_all = np.array(session)
    
    temp_labels = label_all.reshape(-1)


    h,b = np.histogram(temp_labels,bins=64,range=(0,64))
    cutoff = 20
    scaling_factor=np.ones(64)
    for i in range(64):
        if h[i]>cutoff:
            factor = h[i]/cutoff
            scaling_factor[i] = 1/factor


    main_dataset = Concatdataset(data_all,label_all)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # training hyperparameters
    batch_size = 50
    learning_rate = 0.00005
    print_every = 50
    num_epochs = 3100

    logo = LeaveOneGroupOut()
    index_splits = []
    for train_index, test_index in logo.split(data_all,groups =  session_all.reshape(-1)):
        index_splits.append((train_index, test_index))
    print(validation_fold,index_splits[validation_fold][1],test_index.shape)
    
    train_sampler = Data.SubsetRandomSampler(index_splits[validation_fold][0])
    valid_sampler = Data.SubsetRandomSampler(index_splits[validation_fold][1])
    train_loader = torch.utils.data.DataLoader(main_dataset,batch_size=batch_size,sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(main_dataset, batch_size=batch_size,sampler=valid_sampler)


    
    model = get_models(model_num,sequence_len)
    model = model.to(device)


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
        
        
    best_epoch1000 = 0
    best_epoch2000 = 0
    best_epoch3000 = 0
    best_accuracy = 0
    
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

            with torch.no_grad():
                
                test_accuracy=0.0
                test_loss=0.0
                for batchid,traindata in enumerate(validation_loader):
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
                test_accuracy=test_accuracy/len(validation_loader)
                test_loss=test_loss/len(validation_loader)

                train_accuracy=0.0
                train_loss=0.0
                
                for batchid,traindata in enumerate(train_loader):
                    data,labels = traindata
                    labels = labels.reshape(-1).long().to(device=device)
                    data = data.float().to(device=device)
                    out,_= model(data)

                    probs = torch.exp(out)                
                    top_p, top_class = probs.topk(1, dim=1)
                    loss = criterion(out, labels)
                    train_loss += loss.item()

                    equals = top_class == labels.view(*top_class.shape)
                    train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_accuracy=train_accuracy/len(train_loader)
                train_loss=train_loss/len(train_loader)
            
            
            if epoch<=1000:
                if best_accuracy<test_accuracy:
                    best_epoch1000 = epoch
                    best_epoch2000 = epoch
                    best_epoch3000 = epoch
                    best_accuracy = test_accuracy
                    
            
            if epoch<=2000:
                if best_accuracy<test_accuracy:
                    best_epoch2000 = epoch
                    best_epoch3000 = epoch
                    best_accuracy = test_accuracy
            
            if epoch<=3000:
                if best_accuracy<test_accuracy:
                    best_epoch3000 = epoch
                    best_accuracy = test_accuracy

            
            print("Epoch: %d | train_loss: %.3f train_accu: %.3f testloss: %.3f test_accuracy: %.3f"%(epoch,train_loss,train_accuracy,test_loss,test_accuracy),flush=True)

        if epoch%print_every==0:
            with torch.cuda.device(0):
                state_to_save = model.state_dict()
                storename= store_dir+str(epoch)+'.torch'
                torch.save(state_to_save,storename)
    
    best_epochs = np.array([best_epoch1000,best_epoch2000,best_epoch3000])
    np.save(store_dir+'bestepoch.npy',best_epochs)
    

if __name__ == '__main__':
    main(sys.argv[:]) 
    

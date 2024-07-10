#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
from gym import wrappers
import numpy as np
import os
import pickle
import retro
import sys
import time
import warnings
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from PIL import Image, ImageOps


# In[2]:
## HYPERPARAMETER CELL 
def main(args):

    subject= int(args[1])
    
    frame_skip = 5
    sequence_length = int(90/frame_skip) 
    df_file_name_stage1 = '../temp_files/sub-'+str(subject)+'-stage-1-df.pkl'
    h5_store_name = '../data/bw-sub-'+str(subject)+'-len-'+str(sequence_length)+'.h5' 
    runshapes_store_name = '../temp_files/sub-'+str(subject)+'-stage-2-runshapes-'+str(sequence_length)+'.pkl'
    bk2dir = '../data/shinobi.fmriprep/sourcedata/shinobi/'
    output_image_shape = (1,50,100)
    
    print('subject:',subject)

    def process_state(obs):
        obs = obs[50:-10,:,:] #crop
        obs = Image.fromarray(obs.astype(np.uint8))
        obs = ImageOps.grayscale(obs)
        obs = np.array(obs.resize((output_image_shape[2], output_image_shape[1])))
        obs = obs/255 # normalize
        obs = np.expand_dims(obs,2)
        obs = obs.transpose(2,0,1) # HWC to CHW
        return obs
    
    def process_action(action):
        '''
        Keys =  [attack , special,?, ?, up, down, left, right, jump,?,?,?]
        '''

        # converting to 6 actions 
        new_action = [0,0,0,0,0,0]

        if action[0]==True:
            new_action[0]=1
        if action[4]==True:
            new_action[1]=1
        if action[5]==True:
            new_action[2]=1
        if action[6]==True:
            new_action[3]=1
        if action[7]==True:
            new_action[4]=1
        if action[8]==True:
            new_action[5]=1

        new_action = int("".join(str(x) for x in new_action), 2) 
        return new_action


    # In[3]:


    max_episode_length = 30000
    
    gym_folder = '/project/rrg-pbellec/ani686/env/lib/python3.6/site-packages/retro/data/stable/ShinobiIIIReturnOfTheNinjaMaster-Genesis/'
    # In[5]:


    allstates = np.empty((0,sequence_length,1, output_image_shape[1], output_image_shape[2]),dtype=np.float32)
    allactions = np.empty((0,sequence_length,1),dtype=np.float32)
    allsections = np.empty((0,1),dtype=np.float32)
    allruns = np.empty((0,1),dtype=np.float32)
    allreps = np.empty((0,1),dtype=np.float32)
    allrunnumber = np.empty((0,1),dtype=np.float32)



    hf = h5py.File(h5_store_name,'a')
    hf.create_dataset('state', data=allstates,chunks=True, maxshape=(None,sequence_length,1,output_image_shape[1], output_image_shape[2]))
    hf.create_dataset('action', data=allactions,chunks=True, maxshape=(None,sequence_length,1))
    hf.create_dataset('session', data=allsections,chunks=True, maxshape=(None,1))
    hf.create_dataset('run', data=allruns,chunks=True, maxshape=(None,1))
    hf.create_dataset('rep', data=allreps,chunks=True, maxshape=(None,1))
    hf.create_dataset('runnumber', data=allrunnumber,chunks=True, maxshape=(None,1))


    # In[6]:

    df = pd.read_pickle(df_file_name_stage1)


    # In[7]:


    all_files=[]
    for i in range(len(df)):
        t = bk2dir+df['bk2'][i]
        all_files.append(t)


    # In[ ]:


    cf=0
    all_shapes = []
    for f in all_files[:]:



        sess = df['session'][cf]
        run = df['run'][cf]
        rep = df['rep'][cf]
        level = df['level'][cf]
        runnumber = df['run_number'][cf]

        print(cf)
        cf+=1
        filename = f


        file = filename.replace('.bk2', '')
        key_log = retro.Movie(filename)


        frames=[]
        actions=[]
        times=[]
    
        if level==1:
            env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level1-0',scenario = gym_folder+'scenario1-0.json')
        if level==4:
            env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level4-1',scenario = gym_folder+'scenario4-1.json')
        if level==5:
            env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level5-0',scenario = gym_folder+'scenario5-0.json')
        
        state = env.reset()
        start_action=False

        for i in range(max_episode_length):

            if '.bk2' in filename:
                key_log.step()
                action = [key_log.get_key(i, 0) for i in range(env.num_buttons)]


            prev_state = process_state(state)
            dec_action = process_action(action)

            state, _, done, info = env.step( action )

            if True:
                if i%frame_skip == 0:
                    frames.append(prev_state)
                    actions.append(dec_action)


            if done:
                break





        if True:
            frames = np.float32(np.array(frames))
            actions_arr = np.float32(np.array(actions))
            actions_arr = actions_arr.reshape((actions_arr.shape[0],1))

            num_batches = int(frames.shape[0]/sequence_length)
            final_index = num_batches*sequence_length

            frames = frames[:final_index]
            frames = frames.reshape((-1,sequence_length,frames.shape[1],frames.shape[2],frames.shape[3]))
            actions_arr = actions_arr[:final_index]
            actions_arr = actions_arr.reshape((-1,sequence_length,1))

            sess_arr = np.array([sess]*actions_arr.shape[0]).reshape(-1,1)
            run_arr = np.array([run]*actions_arr.shape[0]).reshape(-1,1)
            rep_arr = np.array([rep]*actions_arr.shape[0]).reshape(-1,1)
            runnumber_arr = np.array([runnumber]*actions_arr.shape[0]).reshape(-1,1)


            hf["state"].resize((hf["state"].shape[0] + frames.shape[0]), axis = 0)
            hf["state"][-frames.shape[0]:] = frames

            hf["action"].resize((hf["action"].shape[0] + actions_arr.shape[0]), axis = 0)
            hf["action"][-actions_arr.shape[0]:] = actions_arr

            hf["session"].resize((hf["session"].shape[0] + sess_arr.shape[0]), axis = 0)
            hf["session"][-sess_arr.shape[0]:] = sess_arr

            hf["run"].resize((hf["run"].shape[0] + run_arr.shape[0]), axis = 0)
            hf["run"][-run_arr.shape[0]:] = run_arr

            hf["rep"].resize((hf["rep"].shape[0] + rep_arr.shape[0]), axis = 0)
            hf["rep"][-rep_arr.shape[0]:] = rep_arr
            
            hf["runnumber"].resize((hf["runnumber"].shape[0] + runnumber_arr.shape[0]), axis = 0)
            hf["runnumber"][-runnumber_arr.shape[0]:] = runnumber_arr


            print(frames.shape,sess,runnumber,run,rep,level,flush=True)
            all_shapes.append(frames.shape[0])
            del frames
            del actions
            del actions_arr

        env.close()
    hf.close()


    # In[ ]:

    with open(runshapes_store_name, 'wb') as f:
        pickle.dump(all_shapes, f)

if __name__ == '__main__':
    main(sys.argv[:])








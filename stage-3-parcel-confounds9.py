import nibabel as nib
from nilearn import input_data, plotting,datasets
from load_confounds import Params9
import numpy as np
import pandas as pd
import pickle
import sys

def main(args):
    
    subject = int(args[1])
    
    
    dataset = datasets.fetch_atlas_basc_multiscale_2015()
    atlas_filename = dataset.scale444   
    df_filename_stage1 = '../temp_files/sub-'+str(subject)+'-stage-1-df.pkl'

    df = pd.read_pickle(df_filename_stage1)

    # In[9]:


    all_fmri_paths = []
    for i in range(len(df)):
        sess = df['session'][i]
        run= df['run'][i]

        if len(str(sess))<2:
            path = '/project/rrg-pbellec/ani686/shinobi_ridgereg/data/shinobi.fmriprep/sub-0'+str(subject)+'/ses-00'+str(sess)+'/func/sub-0'+str(subject)+'_ses-00'+str(sess)+'_task-shinobi_run-'+str(run)+'_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
        else:
            path = '/project/rrg-pbellec/ani686/shinobi_ridgereg/data/shinobi.fmriprep/sub-0'+str(subject)+'/ses-0'+str(sess)+'/func/sub-0'+str(subject)+'_ses-0'+str(sess)+'_task-shinobi_run-'+str(run)+'_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'


        all_fmri_paths.append(path)


    # In[ ]:


    masker = input_data.NiftiLabelsMasker(labels_img=atlas_filename,t_r=1.49,smoothing_fwhm=8,standardize=True,detrend=False,memory='nilearn_cache')


    # In[ ]:


    masker.fit() 


    # In[ ]:


    all_data = []
    i=0
    for path in all_fmri_paths:
        print(i,flush=True)
        i+=1
        confounds = Params9().load(path)
        masked_data = masker.transform(path,confounds=confounds) 
        all_data.append(masked_data)


    # In[ ]:


    store_name = '../temp_files/sub-'+str(subject)+'-stage-3-parcel-confounds9-nohigh.pkl'
    with open(store_name, 'wb') as f:
        pickle.dump(all_data, f)

    
    
if __name__ == '__main__':
    main(sys.argv[:]) 
    

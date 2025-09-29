#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nilearn import masking
from nilearn import maskers
from joblib import Parallel,delayed
import numpy as np 
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import xgboost as xgb
import pickle
import shap
import scipy.sparse
import tempfile
import json
import pandas as pd
import spacy
from collections import Counter


# In[2]:


#Construction Mask
masks = glob(r"/media/sunjc/program/tzq/tpl-MNI152NLin2009cAsym/*.nii") #Mask provided by the data set
mask = masking.intersect_masks(masks,threshold=1)   


# In[4]:


func = maskers.NiftiMasker(mask,smoothing_fwhm=6,standardize = True,detrend = True)
func.fit()


# In[ ]:


#10071 voxel number obtained by ISC method
volume_10071 = np.load(r"/media/sunjc/program/tzq/signal/volume_10071.npy")
#Subject_id
subj_id = ["075","131","190","258","260","268","270","271"]


# In[ ]:


#Extraction of fMRI data
subj_data = {}
for id in subj_id:
    tmp_data = func.transform_single_imgs(r"/media/sunjc/program/tzq/afni-nosmooth/story/"+id+".nii")
    subj_data[id] = tmp_data[:,volume_10071]


# In[ ]:


#Find 1742 non-repeating words
sytear_csv = r"/media/sunjc/program/tzq/align_csv/styear.csv" #The csv file provided with the dataset, written with the word and time information
tmp_csv = pd.read_csv(sytear_csv)
styear_words = []
for i in range(len(tmp_csv)):
    if tmp_csv.loc[i,"a"] == "<unk>" or str(tmp_csv.loc[i,"a"])=="nan":
        styear_words.append(tmp_csv.loc[i,"A"])
    else:
        styear_words.append(tmp_csv.loc[i,"a"])

uni_word = []
for word in styear_words:
    if word not in uni_word:
        uni_word.append(word)


#Constructing HRF
time_length = 30.0
frame_times = np.linspace(0, time_length, 301)
onset, amplitude, duration = 0.0, 1.0, 1.0
exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)
oversampling = 16
signal, _labels = compute_regressor(
        exp_condition,
        "spm",
        frame_times,
        con_id="main",
        fir_delays = [5.5],
        oversampling=oversampling,
    )
# src/utils.py
## Import necessary libraries
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from pyts.image import GramianAngularField
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd 
from .model_preprocessing import res_ind,Model_1,Kfloding,mark_img,kf_flod_three



def main_script_binary():
    coc_1=pd.read_csv('Subject1/COCO.csv')
    img_1=pd.read_csv('Subject1/ImageNet.csv')
    sun_1=pd.read_csv('Subject1/SUN.csv')
    df_c=coc_1.iloc[:,:16]
    df_c['label']=0 #COCO
    df_s=sun_1.iloc[:,:16]
    df_s['label']=1 #SUN
    df_C_S=pd.concat([df_c,df_s])
    df_C_S=res_ind(df_C_S)
    # Transform data into images and print shapes
    x,y=mark_img(df_C_S)
    print('x.shape y.shape', x.shape,y.shape)
    # Create and train the binary classification model
    model_1 = Model_1()
    kf=10
    floder_path='/content/out1/'
    model_file='model_coco_vs_sun'
    Kfloding(kf,model_1,x,y,floder_path,model_file,'subject_1',20,batch)

def main_script_threeclass():
    coc_1=pd.read_csv('Subject1/COCO.csv')
    img_1=pd.read_csv('Subject1/ImageNet.csv')
    sun_1=pd.read_csv('Subject1/SUN.csv')
    df_c=coc_1.iloc[:,:16]
    df_c['label']=0
    df_i=img_1.iloc[:,:16]
    df_i['label']=1
    df_s=sun_1.iloc[:,:16]
    df_s['label']=2
    df_total=pd.concat([df_i,df_c,df_s])
    df_total=res_ind(df_total)
    dat = df_total.copy()
    # Transform data into images and print shapes
    x,y=mark_img(dat)
    print('x.shape y.shape', x.shape,y.shape)
    kf=10
    floder_path='/content/out1/'
    model_file='model_coco_vs_imagenet_sun'
    kf_flod_three(kf,x,y,floder_path,model_file,'subject_1',50,batch)


# Main script execution logic here
if __name__ == "__main__":
    # Uncomment the desired main script function
    # main_script_binary()  # For binary classification using GAF summation
    # main_script_threeclass()  # For multi-class classification with combination checks
        



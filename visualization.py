#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 00:48:39 2019

@author: amir
"""
from extraction import *

import numpy as np
#import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import threading
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d, Axes3D

#pca=sklearnPCA(n_components=200)
#x=[v[j][0] for v in X2]
#x=ar(x)
#trans=pca.fit_transform(x)
#plt.scatter(trans[:,0],trans[:,1],c=Y3)


ar= np.array
#lda=LDA(n_components=1)
#trans=lda.fit_transform(x,y)
#j=2
#fig = plt.figure(figsize=(8, 6))
#ax = fig.add_subplot(111, projection='3d')
#ax = Axes3D(fig)
#for x,y,c2,c1,k in [(X,Y,'khaki','gold',0),(X2,Y2,'aqua','teal',1),(X3,Y3,'violet','magenta',2)]:
#    x = [v[j][0] for v in x]
#    x = lda.transform(x)
#    y=ar(y)
##    plt.plot(x[:])
#    ax.scatter(x,np.zeros(len(x)),np.zeros(len(x))+k,c=[c1 if i==1 else c2 for i in y ] , s=50, alpha=0.6, edgecolors='w')
#
#plt.show()

# before it, run pca lines first -- j=?


ds_addr= '/home/amirhossein/Desktop/implement/dataset/fall detection dataset/'
data_addr = 'data/'
options={'downsample_rate':4 , 'background_sub':False, 'stride':2}
#uradl = extraction('ur_fall', options , 'ur_fall_ds4__str2') # dataset folder in ds_addr
#uradl.load()
#f,l = uradl.load_layer(1)
#print(f.shape, l)
datasets =['ur_fall','ur_adl','Office','Home_02','Home_01','Coffee_room_01'] # os.listdir(ds_addr)
dataset_list=[]
layer= 2
X=[];Y=[];colors=[]; cmap1 = sns.color_palette("bright", 2*len(datasets))[:len(datasets)];
cmap2 = sns.color_palette("bright", 2*len(datasets))[len(datasets):len(datasets)*2];
for i,ds in enumerate(datasets):
    ds=extraction(ds, options , ds+'_ds4__str2',ds_addr,data_addr)
#    dataset_list+=[ds]
    ds.load()
    x,y=ds.load_layer(layer)
    colors+=[ cmap1[i] if yy==1 else cmap2[i] for yy in y]
    print(x.shape)
    X+=[x];Y+=[y]


# =================
j=0
#x = [v[j][0] for v in X]+[v[j][0] for v in X2]+[v[j][0] for v in X3]
pca=sklearnPCA(n_components=200)
X = np.concatenate(X, axis=0)
sh = X.shape[2]
X=pca.fit_transform(X.reshape(-1,sh))
print('pca done!')
print('pca.explained_variance_ratio_',pca.explained_variance_ratio_[:10])
#y= Y+Y2+Y3
#j=2
X_embedded = TSNE(n_components=3).fit_transform(X)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)
#for x,y,c2,c1,k,a in [(X,Y,'aqua','teal',0,0),(X2,Y2,'khaki','gold',1,len(X)),(X3,Y3,'violet','magenta',2,len(X)+len(X2))]:
##    x = [v[j][0] for v in x]
##    x = TSNE.transform(x)

#    b = a+len(y)
#    y=ar(y)
##    plt.plot(x[:])
#    ax.scatter(X_embedded[a:b,0],X_embedded[a:b,1],X_embedded[a:b,2],c=[c1 if i==1 else c2 for i in y ] , s=50, alpha=0.6, edgecolors='w')
ax.scatter(X_embedded[:,0],X_embedded[:,1],X_embedded[:,2],c= colors , s=5, alpha=0.6)
plt.show()
#ax.set_xlabel('a')
#ax.set_ylabel('b')
#ax.set_zlabel('c')







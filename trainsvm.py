""" 
Grid search for cross validation in (SVM parameters , layer of C3D) 


how to combine datasests for CV (whole ds in train or test or valid. or part) -> part( will nego)
-> now i split trainds for train,valid. (20% valid)
how should i use train,valid seperately or in a loop... (grid search cv)
some ds just have nagative labels(home2 mostly) -> pay attention in detemining TrainTestValid...


using reweighting
using pca -> first train,test,valid date dimensions reduced w PCA.
"""

from extraction import *

import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
ar = np.array

model_addr = 'models/train_svm/'
ds_addr= '/home/amirhossein/Desktop/implement/dataset/fall detection dataset/'
data_addr = 'data/'
options={'downsample_rate':4 , 'background_sub':False, 'stride':2}
#uradl = extraction('ur_fall', options , 'ur_fall_ds4__str2') # dataset folder in ds_addr
#uradl.load()
#f,l = uradl.load_layer(1)
#print(f.shape, l)
datasets =['ur_fall','ur_adl','Office','Home_02','Home_01','Coffee_room_01'] # os.listdir(ds_addr)
dataset_list=[]

#layer= 2
layers = [2,0,1,3]
gridCV = {
    'C': [0.001,0.1,1,10,20,30,40,60,80,100],
    'gamma': [0.1,0.01,0.001,0.0001,0.00006,0.00003,0.00001],
        }
#trainds = ['ur_fall','ur_adl','Coffee_room_01']; testds = ['Office','Home_02','Home_01']; # validds=[]
datasets = [[['ur_fall','ur_adl','Coffee_room_01'],['Office','Home_02','Home_01']],\
            [['Office','Home_02','Home_01'],['ur_fall','ur_adl','Coffee_room_01']]]
for trainds, testds in datasets:
    layerscores = []        
    for layer in layers:
        print('\n Layer {}   train_dataset:{}    test_dataset:{}'.format(layer, trainds, testds))
        X=[];Y=[];colors=[]; 
        #cmap1 = sns.color_palette("bright", 2*len(datasets))[:len(datasets)];
        #cmap2 = sns.color_palette("bright", 2*len(datasets))[len(datasets):len(datasets)*2];
        Xtrain=[]; Xtest = []
        Ytrain=[]; Ytest = []
        
        for i,ds in enumerate(trainds):
            ds=extraction(ds, options , ds+'_ds4__str2',ds_addr,data_addr)
       #     dataset_list+=[ds]
            ds.load()
            x,y=ds.load_layer(layer)
    #        colors+=[ cmap1[i] if yy==1 else cmap2[i] for yy in y]
            print(x.shape)
            Xtrain+=[x]; Ytrain+=y
        Xtrain = np.concatenate(Xtrain, axis=0)[:,0,:]
        Xtrain, Xvalid, Ytrain,Yvalid = train_test_split(Xtrain, Ytrain, test_size=0.2 )
        
        for i,ds in enumerate(testds):
            ds=extraction(ds, options , ds+'_ds4__str2',ds_addr,data_addr)
       #     dataset_list+=[ds]
            ds.load()
            x,y=ds.load_layer(layer)
    #        colors+=[ cmap1[i] if yy==1 else cmap2[i] for yy in y]
            print(x.shape)
            Xtest+=[x]; Ytest+=y
        Xtest = np.concatenate(Xtest, axis=0)[:,0,:]

        del ds

        pca=sklearnPCA(n_components=200)
        X = np.concatenate([Xtrain, Xtest, Xvalid])
        pca.fit(X)
        Xtrain = pca.transform(Xtrain)
        Xvalid = pca.transform(Xvalid)
        Xtest = pca.transform(Xtest)        
        del X; del pca

        scorel=[]
        param=[]
        for c in gridCV['C']:
            for g in gridCV['gamma']:
                svclassifier1 = SVC(kernel='rbf', C=c, gamma=g,class_weight='balanced')
                svclassifier1.fit(Xtrain,Ytrain)
                Ypred = svclassifier1.predict(Xvalid)
                falsealarm=0; total = len(Ypred); miss=0; alarm=0;
                for i in range(len(Ypred)):
                    if Ypred[i]==1 and Yvalid[i]==-1:
                        falsealarm+=1
                    if Ypred[i]==-1 and Yvalid[i]==1:
                        miss+=1
                    if Ypred[i]==1 and Yvalid[i]==1:
                        alarm+=1                
               
               
                s = alarm/(miss+alarm)* (1-falsealarm/(total-alarm-miss))#svclassifier1.score(xvalid, yvalid)
                scorel +=[s]
                param  +=[[c,g,s]]
#                print([c,g,s])
        bestparam= param[np.argmax(scorel)]
        print('best parameters in validation C {}   gamma {} score(sens*falarm {}'.format(bestparam[0], bestparam[1],bestparam[2]))
       
        svclassifier1 = SVC(kernel='rbf', C=bestparam[0], gamma=bestparam[1],class_weight='balanced')
        svclassifier1.fit(np.append(Xtrain,Xvalid,axis=0),np.append(Ytrain,Yvalid,axis=0))
        s = svclassifier1.score(Xtest, Ytest)
        Ypred = svclassifier1.predict(Xtest)
        falsealarm=0; total = len(Ypred); miss=0; alarm=0;
        for i in range(len(Ypred)):
            if Ypred[i]==1 and Ytest[i]==-1:
                falsealarm+=1
            if Ypred[i]==-1 and Ytest[i]==1:
                miss+=1
            if Ypred[i]==1 and Ytest[i]==1:
                alarm+=1
        print( '\nlayer:{}    train_dataset:{}    test_dataset:{}'.format(layer, trainds, testds))
        print('tot', total, 'alarm', alarm, 'f alrm', falsealarm,'miss',miss, 'else', total-alarm-falsealarm-miss)
        print('SVMscore{:.3} sens{:.3} falarm{:.3} '.format\
              (s, alarm/(miss+alarm), falsealarm/(total-alarm-miss)), \
              'c,g ', bestparam[:2])
        layerscores += [[layer , alarm/(miss+alarm)*(1- falsealarm/(total-alarm-miss))]]      
    n = np.argmax([l[1] for l in layerscores])            
    print('\nFinal Best Layer {}  sens*falarm {:.3f}'.format(layerscores[n][0], layerscores[n][1]))
















# ================================================

#dataset_list = [{'name':'urfall'+name_append, 'vid_s':1, 'vid_e':30, 
#              'addr':ur_addr+"/fall-{:02}-cam0-rgb/fall-{:02}-cam0-rgb-{:03}.png",'ur':True},
#            {'name':'uradl'+name_append, 'vid_s':1, 'vid_e':40, 
#              'addr':ur_addr+"/adl-{:02}-cam0-rgb/adl-{:02}-cam0-rgb-{:03}.png",'ur':True},
#            {'name':'office'+name_append, 'vid_s':1, 'vid_e':33, 
#              'addr':office_addr,'ur':False},
#            {'name':'home1'+name_append, 'vid_s':1, 'vid_e':30, 
#              'addr':home1_addr,'ur':False},
#            {'name':'home2'+name_append, 'vid_s':31, 'vid_e':60, 
#              'addr':home2_addr,'ur':False},

#             ]

#train_valid_test = {'train': [], 'valid':[], 'test'=[]}
## def trainsvm(dataset_list, data_addr, name_append, gridCV,train_valid_test):
#if True:
#    infile = open(data_addr+'labels'+name_append,'rb')
#    labels = pickle.load(infile)
#    infile.close()  
#    
#    Xtrain=[]; Xvalid =[]; Xtest = []
#    Ytrain=[]; Yvalid =[]; Ytest = []
#    # check whether samples and labels are equal length
#    for dataset in dataset_list:
#        name = dataset['name']
#        print(name)
#        Y= labels[name]
#        
#        infile = open(data_addr+ name,'rb')
#        all_features = pickle.load(infile)
#        infile.close()           
#           
#        if name in tvt['train']:
#            Xtrain += 
#            Ytrain +=
#        if name in tvt['valid']:
#            Xvalid += 
#            Yvalid +=
#        if name in tvt['test']:
#            Xtest += 
#            Ytest +=
#        
#        for k,v in all_features.items():
#            if len(v)!= len(Y[k]):
#                print('error=====',  name, len(v), len(Y[k]),k)
#                
#    del all_features 
#    
#    tvt = train_valid_test
#    
 
# del data



# for j in [3,2,0,1]:#(0,20,0.00007),(1,20,0.00002),(2,20,0.00007),(3,100,1)]:
#     print('==================fc', j)
#     pca=sklearnPCA(n_components=200)
#     x = [v[j][0] for v in X]+[v[j][0] for v in X2]+[v[j][0] for v in X3]
#     pca.fit(x)    
#     x1 = pca.transform([v[j][0] for v in X])
#     x2 = pca.transform([v[j][0] for v in X2])
#     x3 = pca.transform([v[j][0] for v in X3])
    
#     for xtrain,ytrain,xvalid,yvalid,xtest,ytest,datasets in [[x1,Y,x2,Y2,x3,Y3,'1,2,3']\
#         , [x2,Y2,x1,Y,x3,Y3,'2,1,3'], [x3,Y3,x2,Y2,x1,Y,'3,2,1']]:



#         scorel=[]
#         param=[]
#         for c in gridCV['C']:
#             for g in gridCV['gamma']:
#                 svclassifier1 = SVC(kernel='rbf', C=c, gamma=g,class_weight='balanced')
#                 svclassifier1.fit(xtrain,ytrain)
#                 ypred = svclassifier1.predict(xvalid)
#                 falsealarm=0; total = len(ypred); miss=0; alarm=0;
#                 for i in range(len(ypred)):
#                     if ypred[i]==1 and yvalid[i]==-1:
#                         falsealarm+=1
#                     if ypred[i]==-1 and yvalid[i]==1:
#                         miss+=1
#                     if ypred[i]==1 and yvalid[i]==1:
#                         alarm+=1                
                
                
#                 s = alarm/(miss+alarm)* (1-falsealarm/(total-alarm-miss))#svclassifier1.score(xvalid, yvalid)
#                 scorel +=[s]
#                 param  +=[[c,g,s]]
# #                print([c,g,s])

#         bestparam= param[np.argmax(scorel)]
        
#         svclassifier1 = SVC(kernel='rbf', C=bestparam[0], gamma=bestparam[1],class_weight='balanced')
#         svclassifier1.fit(np.append(xtrain,xvalid,axis=0),np.append(ytrain,yvalid,axis=0))
#         s = svclassifier1.score(xtest, ytest)
#         ypred = svclassifier1.predict(xtest)
#         falsealarm=0; total = len(ypred); miss=0; alarm=0;
#         for i in range(len(ypred)):
#             if ypred[i]==1 and ytest[i]==-1:
#                 falsealarm+=1
#             if ypred[i]==-1 and ytest[i]==1:
#                 miss+=1
#             if ypred[i]==1 and ytest[i]==1:
#                 alarm+=1
#         print('tot', total, 'alarm', alarm, 'f alrm', falsealarm,'miss',miss, 'else', total-alarm-falsealarm-miss)
#         print('sc{:.3} sens{:.3} falarm{:.3} '.format\
#               (s, alarm/(miss+alarm), falsealarm/(total-alarm-miss)), \
#               'c,g ', bestparam[:2] ,'Dsets:', datasets)
        
            
            

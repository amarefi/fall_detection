'''
each dataset has a different annotation format! 
first it lists start and end frames of falliing in each video and each dataset
then according to dataset_list and options, labels are generated and saved as new files

in a clip of 64 frames, at leat 16 frames of falling and 16 frames of lying on the floor 
is needed to consider a clip as falling
'''

import numpy as np
#import cv2 as cv2
import pandas as pd
#import pickle
# to lower sample rate, lines 34 37 40 44 changed # dont forget to change 3 pickle names
ar = np.array

options={'downsample_rate':4 , 'background_sub':False, 'stride':2}
annot_addr = '../models/'
data_addr = 'data/'
name_append = '_dr4'

# def labeling(data_addr, annot_addr, options, name_append):
#try:
# UR dataset =====================
# parameters
nadl = 40 # adl videos
nfall = 30 # fall videos
uradl_annot = pd.read_csv(annot_addr+'urfall-cam0-adls.csv', header = None)
urfall_annot = pd.read_csv(annot_addr+'urfall-cam0-falls.csv', header = None)

ds_addr= '/home/amirhossein/Desktop/implement/dataset/fall detection dataset/'

# lets make dictionaries
ur_fall = {} # startofFALL, endofFALL, video length for each video
for i in range(1,nfall+1):
    t = urfall_annot.loc[urfall_annot[0]=='fall-{:02}'.format(i)]
    frame_labels = t.loc[:,2].values
    start_end_len = [ np.argwhere(frame_labels==0)[0][0], np.argwhere(frame_labels==1)[0][0], len(frame_labels)]
    ur_fall.update({i:start_end_len})    
f = open(ds_addr+'ur_fall/'+'ur_fall.txt', "w+")
for k,v in ur_fall.items():
    f.write('{} {} {}\n'.format(v[0],v[1],v[2]) )
f.close()

# ADL
ur_adl = {} #  video length for each video
for i in range(1,nadl+1):
    t = uradl_annot.loc[uradl_annot[0]=='adl-{:02}'.format(i)]
    length = t.loc[:,1].values[-1]
    if i==17:
        length=230 #problem in csv file 
    ur_adl.update({i:[0,0,length]})   # 0,0,... means no falling
f = open(ds_addr+'ur_adl/'+'ur_adl.txt', "w+")
for k,v in ur_adl.items():
    f.write('{} {} {}\n'.format(v[0],v[1],v[2]) )
f.close()

#stride = options['stride']
#c3d_len = 16 # 16 frames, form a clip, given to C3D model
#ds_rate = options['downsample_rate']
#fall_duration = 16 # num of frames to be considered as fall
#lying_duration = 4 






#urfall_labels={}
#for k,v in ur_fall.items(): # video num: [start, end, len]
#    urfall_labels[k]=labels4video(v[0],v[1],v[2])



#uradl_labels={}
#for k,v in ur_adl.items(): # video num: [start, end, len]
#    num_samples = (((v-1)//ds_rate)+1-c3d_len)// stride +1
#    uradl_labels[k]=[]
#    uradl_labels[k]+=[-1]*num_samples


## OFFICE dataset =================
f=open(annot_addr+'office1.txt', "r")
fl =f.readlines()

office_labels = {}
for j in range(1,33+1):
    spl = fl[j-1].split()
    start= int(spl[1]); end=int(spl[2]); length=int(spl[3])
    office_labels[j]=[start,end,length]#labels4video(start,end,length)
f = open(ds_addr+'Office/'+'Office.txt', "w+")
for k,v in office_labels.items():
    f.write('{} {} {}\n'.format(v[0],v[1],v[2]) )
f.close()
    
            
# HOME, COFFEE dataset ==================
# home1 
home1_labels={}
for i in range(1,30+1):
    try:
        f=open(annot_addr+'home1/video ({}).txt'.format(i), "r")
    except:
        print("couldnt open the file", i)    
    fl =f.readlines()
    start = int(fl[0])
    end   = int(fl[1])
    try:
        length= int(fl[-1].split(',')[0]) #TODO check some files have same last lines.
    except:
        length= int(fl[-2].split(',')[0])
    home1_labels[i]= [start,end,length]#labels4video(start, end, length)
f = open(ds_addr+'Home_01/'+'Home_01.txt', "w+")
for k,v in home1_labels.items():
    f.write('{} {} {}\n'.format(v[0],v[1],v[2]) )
f.close()

# home2
home2_labels={}
for i in range(31,60+1):
    try:
        f=open(annot_addr+'home2/video ({}).txt'.format(i), "r")
    except:
        print("couldnt open the file", i)    
    fl =f.readlines()
    start = int(fl[0])
    end   = int(fl[1])
    try:
        length= int(fl[-1].split(',')[0]) #TODO check some files have same last lines.
    except:
        length= int(fl[-2].split(',')[0])
    home2_labels[i]= [start,end,length]#labels4video(start, end, length)
f = open(ds_addr+'Home_02/'+'Home_02.txt', "w+")
for k,v in home2_labels.items():
    f.write('{} {} {}\n'.format(v[0],v[1],v[2]) )
f.close()


# coffee dataset
coffee_labels={}
for i in range(1,48+1):
    try:
        f=open(annot_addr+'coffee1/video ({}).txt'.format(i), "r")
    except:
        print("couldnt open the file", i)    
    fl =f.readlines()
    start = int(fl[0])
    end   = int(fl[1])
    try:
        length= int(fl[-1].split(',')[0]) #TODO check some files have same last lines.
    except:
        length= int(fl[-2].split(',')[0])
    coffee_labels[i]= [start,end,length]#labels4video(start, end, length)
f = open(ds_addr+'Coffee_room_01/'+'Coffee_room_01.txt', "w+")
for k,v in coffee_labels.items():
    f.write('{} {} {}\n'.format(v[0],v[1],v[2]) )
f.close()

#all_labels = {'coffee'+name_append:coffee_labels,'home1'+name_append:home1_labels,\
#              'home2'+name_append:home2_labels, 'office'+name_append:office_labels,\
#              'uradl'+name_append:uradl_labels, 'urfall'+name_append:urfall_labels}
#filename = data_addr+'labels'+name_append
#outfile = open(filename,'wb')
#pickle.dump(all_labels ,outfile)
#outfile.close()

    
#except:
#    pass
    
    

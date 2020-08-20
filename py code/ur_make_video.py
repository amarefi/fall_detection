
"""
Downloaded UR dataset, is composed of frames that saved seperately.
in this code, i make a video out of frames, in order to be compatible with other datasets. 

"""

import numpy as np
ar = np.array
#from skimage.transform import rescale, resize, downscale_local_mean
#import matplotlib.pyplot as plt
#from imread import imread, imsave
#import skimage
import cv2
import glob
import os

ur_frames = "/home/amirhossein/Desktop/implement/dataset/ur dataset fall/"
ur_videos = "/home/amirhossein/Desktop/implement/dataset/fall detection dataset/ur_dataset/"
frame_rate = 32

for i in range(1,31):
    frames_addr = ur_frames+'fall-{:02}-cam0-rgb/'.format(i)
    frames = os.listdir(frames_addr)
    frames.sort()
    img_array = []
    for filename in frames:
        img = cv2.imread(frames_addr+filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    print("fall {}   size {}   length {}".format(i,size,len(frames)))
    out = cv2.VideoWriter(ur_videos+'fall{:02}.avi'.format(i),cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
for i in range(1,41):
    frames_addr = ur_frames+'adl-{:02}-cam0-rgb/'.format(i)
    frames = os.listdir(frames_addr)
    frames.sort()
    img_array = []
    for filename in frames:
        img = cv2.imread(frames_addr+filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    print("adl {}   size {}   length {}".format(i,size,len(frames)))
    out = cv2.VideoWriter(ur_videos+'adl{:02}.avi'.format(i),cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
    

"""
this code extracts C3D features out of files in ds_addr
using class extraction. with specified frame rate and ...
if data is not available in data folder, then model C3D model is loaded and calculated.

if a video has not enough frames(less than 16) it is not included in samples. 
a clip(of 16 downsampled frames) must have enough fall_duration and lying_duration in order to have + label.

"""

import numpy as np
ar = np.array
#from skimage.transform import rescale, resize, downscale_local_mean
#import matplotlib.pyplot as plt
#from imread import imread, imsave
#import skimage
import cv2
import pickle
from sklearn import svm
from scipy.signal import find_peaks
import os; import glob
# diff bw datasets within code: how to read them.
# *** changed for backg detection

c3d_len = 16

#address = 'videos/labels.txt'
#f=open(address, "r")
	#labels =f.readlines()
def load_c3d():
#    import tensorflow.keras as keras
    import tensorflow as tf
    from sport1m_model import create_model_functional, create_model_sequential
    from tensorflow.keras.models import Sequential, Model
    model = create_model_sequential()
    try:
        model.load_weights('models/C3D_Sport1M_weights_keras_2.2.4.h5')
    except OSError as err:
        print('Check path to the model weights\' file!\n\n', err)
    except :
        print("errooooor")

    C3D = Model(inputs=model.input,
                      outputs=[ model.get_layer('fc6').output,
                               model.get_layer('flat5').output,
                               model.get_layer('fc7').output,
                               model.get_layer('fc8').output])
    del model
    return C3D
                
        
        
# ===========================
def ext(l,c3d):
    
    # 16 black frames with 3 channels
    frames16 = l[:16].reshape((1,16,112,112,3)).astype('int') -100 # .astype('int')*255
    features_4layers = c3d.predict(frames16)
    return features_4layers
    


#def showclip(inp,sf):
#    capture = cv2.VideoCapture(inp)
#    num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
#    
#    capture.set(1, sf)
#    res, frame = capture.read()
#    f = frame
#    mf = sf+31
#    capture.set(1, mf)
#    res, frame = capture.read()
#    f = np.append( f, frame, axis=1)
#    ef = sf+63
#    if ef>=num_frame:
#        print('last frames reached...')
#        ef-=4
#    capture.set(1, ef)
#    res, frame = capture.read()
#    f = np.append( f, frame, axis=1)    
#    
#    print('start frame ', sf, '   total',num_frame)
#    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
#    plt.imshow(f)
#    plt.show()
#        
#def showclipur(l,sf):

#    f = l[sf//4]

#    f = np.append( f, l[sf//4+7], axis=1)

#    f = np.append( f, l[sf//4+15], axis=1)
#    
#    num_frame = len(l)*4
##    print('n frame', num_frame)
#    print('start frame ', sf, '   total',num_frame)
#    f = np.array(f*255,dtype='uint8')
#    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
#    plt.imshow(f)
#    plt.show()



class extraction():
    def __init__(self, dataset_name, options, label,ds_addr,data_addr, fall_duration=16,lying_duration=16):
        self.ds_addr = ds_addr; self.data_addr = data_addr; 
        self.dataset_name = dataset_name; 
        self.label = label
        self.options = options
        self.dsrate = options['downsample_rate']
        self.stride = options['stride'] #2 # was 2 meaning 8 frame steps. # 2 when stride=4
        self.fall_duration = fall_duration
        self.lying_duration = lying_duration

    def load(self):
        # if not exist first extract then load... 
#        if np.loadz(...)
        try:
#            a = np.load(self.data_addr+self.label)
            filename =self.data_addr+ self.label
            infile = open(filename,'rb')
            self.features_all = pickle.load(infile)
            infile.close()
            
            print(self.label, "loaded successfully")
        except:
            self.extract()
        f = open(self.ds_addr+ self.dataset_name+'/'+self.dataset_name+'.txt', "r+")
        self.annot = f.readlines()
        f.close()
#        self.features_all        
    
    def load_layer(self,layer): # loads data of nth layer of C3D with labels.
        i=0
        layer_features=[];labels=[]
        for k,v in self.features_all.items():
            start,end,length = self.annot[i].split()
            i+=1
            labels+=self.labels4video(int(start),int(end),int(length), len(v))
            layer_features +=[clipdata[layer] for clipdata in v]
            if len(labels)!=len(layer_features):
                print("error,loadlayer",len(labels),len(layer_features))
        return np.array(layer_features),labels
            
    def extract(self):
        print(self.label, "extracting ...")
        if self.dataset_name not in os.listdir(self.ds_addr):
            print("error , folder with dataset_name was not found\n\n")
#        np.savez_compressed(
        
        C3D = load_c3d()        
        
        
        features_all = {}
        f = open(self.ds_addr+ self.dataset_name+'/'+self.dataset_name+'.txt', "r+")
        annot = f.readlines()
        f.close()
        videos = glob.glob(self.ds_addr+ self.dataset_name+'/*.avi')
        videos.sort()
        for i,video in enumerate(videos):
            print(video, annot[i])

            capture = cv2.VideoCapture( video )
            num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if int(annot[i].split()[-1])!=num_frame:
                print("error, num frames", num_frame)
            if not capture.isOpened:
                print('Unable to open: ' + inp)
                exit(0)

            backSub = cv2.createBackgroundSubtractorMOG2()
            frame16= []
            i=0
            for i in range(0, num_frame):
                ret, frame = capture.read()
                if self.options['background_sub']:
                    mask = backSub.apply(frame)/255.0
                    mask = np.repeat(mask, 3).reshape(mask.shape[0],mask.shape[1],3)
                    masked = np.multiply(mask , frame).astype('uint8')
                    frame = masked
                    
#                if i%8==0:
#                    plt.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
#                    plt.show()
                
                if i==0:
                    r,c,_ = frame.shape # here the biggest square in the middle of frame is cropped
                    d = (c-r)//2
                    sc = d
                    ec = r+d
                    print(frame.shape,'  ', sc,ec)
                if i% self.dsrate==0:
            #        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame[:, sc:ec], (112, 112), interpolation = cv2.INTER_AREA)
                    if np.random.randint(10)<1:plt.imshow(frame_resized);plt.savefig('temp/{}.jpg'.format(i))
                    frame16 += [frame_resized]
                if frame is None:
                    break
        #============================================
            frame16 = ar(frame16)
            c3dlength = 16
            if len(frame16)<16 :
                print("error... not enough frames")
                continue
            sample_number = (len(frame16)-c3dlength)// self.stride +1
            
            features = []
            for i in range(sample_number):
                f = ext(frame16[i*self.stride:i*self.stride+16],C3D)
                features += [f]
            features_all.update({video:features})
            
        filename =self.data_addr+ self.label
        outfile = open(filename,'wb')
        pickle.dump(features_all ,outfile)
        outfile.close()
        self.features_all = features_all
        
    def labels4video(self, start,end,length,num):
#        num = (((length-1)// self.dsrate)+1-c3d_len)// stride +1
#        sample_number = (len(frame16)-c3dlength)// self.stride +1
        labellist = []
        for i in range(num):
            if i*self.stride* self.dsrate<end-self.fall_duration and  end+self.lying_duration<i*self.stride* self.dsrate+c3d_len* self.dsrate:
                labellist+=[1]
            else:
                labellist+=[-1]    
#        print(labellist)
        return labellist

    
#options={'downsample_rate':4 , 'background_sub':False, 'stride':2}
#uradl = extraction('ur_fall', options , 'ur_fall_ds4__str2') # dataset folder in ds_addr
#uradl.load()
#f,l = uradl.load_layer(1)
#print(f.shape, l)
#datasets = os.listdir(ds_addr)
#dataset_list=[]
#for ds in datasets:
#    ds=extraction(ds, options , ds+'_ds4__str2')
#    dataset_list+=[ds]
#    ds.load()
#    dl.load_layer(1)

    

#In [4]: a = d['/home/amirhossein/Desktop/implement/dataset/fall detection dataset/ur_fall/fall01.avi']

#In [5]: len(a)
#Out[5]: 13

#In [6]: a[0][0].shape
#Out[6]: (1, 4096)

#In [7]: a[0][2].shape
#Out[7]: (1, 4096)

#In [8]: a[0][3].shape
#Out[8]: (1, 487)

#In [9]: a[0][1].shape
#Out[9]: (1, 8192)

    
#            if ur:
#                frame16 = [] # array of 16 frames are given to C3D model.
#                backSub = cv2.createBackgroundSubtractorMOG2()
#            
#                i=0 # frame number
#                while 1:
#                    i+=1
#                    frame = cv2.imread(dataset_addr.format(video,video,i))
#                    if np.all(frame) == None:
#                        break
#                    if options['background_sub']:
#                        mask = backSub.apply(frame)/255.0
#                        mask = np.repeat(mask, 3).reshape(mask.shape[0],mask.shape[1],3)
#                        masked = np.multiply(mask , frame).astype('uint8')
#                        frame = masked
#    #                if i%5==1:
#    #                    plt.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
#    #                    plt.show()
#                    if (i-1)% options['downsample_rate']==0:
#                        frame_resized = cv2.resize(frame[:, 80:560], (112, 112), interpolation = cv2.INTER_AREA)
#                        frame16 += [frame_resized]  
#                del frame16
            #===========
#            else:            

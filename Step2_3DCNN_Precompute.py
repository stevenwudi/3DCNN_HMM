#-------------------------------------------------------------------------------
# Name:        Starting Kit for ChaLearn LAP 2014 Track3
# Purpose:     Show basic functionality of provided code
#
# Author:      Xavier Baro
# Author:      Di Wu: stevenwudi@gmail.com
# Created:     24/03/2014
# Copyright:   (c) Chalearn LAP 2014
# Licence:     GPL3
#-------------------------------------------------------------------------------
import sys, os,random,numpy,zipfile
from shutil import copyfile
import matplotlib.pyplot as plt
import cv2
import time
import cPickle
import numpy
import pickle
import scipy.io as sio  
from scipy.misc import imresize
from numpy import log

from ChalearnLAPEvaluation import evalGesture,exportGT_Gesture
from ChalearnLAPSample import GestureSample

print("Extracting the training files")
data_path=os.path.join("I:\Kaggle_multimodal\Training\\")  
# Get the list of training samples
samples=os.listdir(data_path)
STATE_NO = 10
meta = pickle.load(open(r'.\ConvNet_3DCNN\storage_sk_final\batches.meta'))
data_mean = meta['data_mean']


# pre-allocating the memory
IM_SZ = 90
debug_show = True

for file_count, file in enumerate(samples):        
    if  not file_count<650:
        time_tic = time.time()  
        print("\t Processing file " + file)
        # Create the object to access the sample
        smp=GestureSample(os.path.join(data_path,file))
        # ###############################################
        # USE Ground Truth information to learn the model
        # ###############################################
        # Get the list of actions for this frame
        total_frame = smp.getNumFrames()
        ##################################################
        # obtain the shift and scaling according to
        shift, scale = smp.get_shift_scale_sk( start_frame=total_frame-100, end_frame=total_frame-10)
        print ("shift: {} scaling: {:.4}".format(numpy.array(shift), scale))
        if numpy.isnan(scale):
            scale = 1.0

        cuboid = numpy.zeros((IM_SZ, IM_SZ, total_frame), numpy.uint8)
        frame_count_temp = 0 
        for x in range(1, smp.getNumFrames()):
            [img, flag] = smp.get_shift_scale_depth_sk_normalize(shift, scale, x, IM_SZ, show_flag=False)
            cuboid[:, :, frame_count_temp] =  img
            frame_count_temp +=1
            if False: # show the segmented images here
                print x
                cv2.imshow('image', img)
                cv2.waitKey(1)   

        print "Elapsed time %d sec" % int(time.time() - time_tic)
        save_dir=r'.\ConvNet_3DCNN\Depth_Precompute'
        save_path= os.path.join(save_dir,file)
        out_file = open(save_path, 'wb')
        dic = {'cuboid':cuboid}
        cPickle.dump(dic, out_file, protocol=cPickle.HIGHEST_PROTOCOL)
        out_file.close()
     # ###############################################
        ## delete the sample
        del smp
        
                          
                    

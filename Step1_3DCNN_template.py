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
from ChalearnLAPEvaluation import evalGesture,exportGT_Gesture
from ChalearnLAPSample import GestureSample
from utils import IsLeftDominant
from utils import Extract_feature_normalized
from utils import Extract_feature
import time
import cPickle

""" 
We choose the Sample3 as the template...
A bit random
"""
# Data folder (Training data)
print("Extracting the training files")
data=os.path.join("I:\Kaggle_multimodal\Training\\")  
# Get the list of training samples
samples=os.listdir(data)
STATE_NO = 10


for file_count, file in enumerate(samples):
    time_tic = time.time()      
    if (file_count>2):
        print("\t Processing file " + file)
        # Create the object to access the sample
        smp=GestureSample(os.path.join(data,file))
        # ###############################################
        # USE Ground Truth information to learn the model
        # ###############################################
        # Get the list of actions for this frame
        gesturesList=smp.getGestures()

        for gesture in gesturesList:
            gestureID,startFrame,endFrame=gesture
            cuboid = numpy.zeros(( 90, 90, endFrame-startFrame+1), numpy.uint8)
            frame_count_temp = 0 
            for x in range(startFrame, endFrame):
                img = smp.getDepth3DCNN(x, ratio=0.25)
                cuboid[:, :, frame_count_temp] = img
                frame_count_temp +=1
  
            fr_no = endFrame-startFrame+1
            for frame in range(fr_no-3):
                Feature_all[:,:,:,cuboid_count] = cuboid[:,:, frame:frame+4]
                state_no = numpy.floor(frame*(STATE_NO*1.0/(fr_no-3)))
                Targets[cuboid_count] = state_no+STATE_NO*(gestureID-1)
                cuboid_count += 1
        # ###############################################
        ## delete the sample
        del smp        
        print "Elapsed time %d sec" % int(time.time() - time_tic)
                                          

        if(not file_count%50 and file_count>0):
            random_idx = numpy.random.permutation(range(cuboid_count))
            save_dir = '.\storage\\'
            print "batch: %d" % batch_num
            data_temp = Feature_all[:,:,:,random_idx[:]]
            data_id = Targets[random_idx[:]]
            
            save_path= os.path.join(save_dir,'data_batch_'+str(batch_num))
            out_file = open(save_path, 'wb')
            dic = {'batch_label':['batch 1 of'+ str(batch_num)], 'data':data_temp, 
                            'data_id':data_id}
            cPickle.dump(dic, out_file)
            out_file.close()

            # pre-allocating the memory
            batch_num += 1
            Feature_all =  numpy.zeros(shape=(90, 90, 4, 100000), dtype=numpy.uint8 )
            Targets = numpy.zeros(shape=(100000, 1), dtype=numpy.uint8)
            cuboid_count = 0


#data_mean = data_temp.mean(axis=3)
#dic = {'data_dim': 90*90*4,'data_in_rows':True, 'data_mean': data_mean}
#file_name ='.\storage\\batches.meta'
#out_file = open(file_name, 'wb')
#cPickle.dump(dic, out_file)
#out_file.close()


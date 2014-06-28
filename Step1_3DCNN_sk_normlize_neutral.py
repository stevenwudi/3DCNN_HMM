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
import time
import cPickle
from scipy.misc import imresize
""" Main script. Show how to perform all competition steps
    Access the sample information to learn a model. """
# Data folder (Training data)
print("Extracting the training files")
data=os.path.join("I:\Kaggle_multimodal\Training\\")  
# Get the list of training samples
samples=os.listdir(data)

STATE_NO = 10
batch_num = 29

# pre-allocating the memory
IM_SZ = 90

Feature_all =  numpy.zeros(shape=(IM_SZ, IM_SZ, 4, 20000), dtype=numpy.uint8)
Targets = numpy.zeros(shape=(100000, 1), dtype=numpy.uint8)
cuboid_count = 0


for file_count, file in enumerate(samples):
    time_tic = time.time()      
    if (file_count<650 and file_count!=155  and file_count!=538 and file_count>560):
        print("\t Processing file " + file)
        # Create the object to access the sample
        smp=GestureSample(os.path.join(data,file))
        # ###############################################
        # USE Ground Truth information to learn the model
        # ###############################################
        # Get the list of actions for this frame
        gesturesList=smp.getGestures()

        total_frame = smp.getNumFrames()
        ##################################################
        # obtain the shift and scaling according to
        shift, scale = smp.get_shift_scale_sk( start_frame=total_frame-100, end_frame=total_frame-10)
        print ("shift: {} scaling: {:.4}".format(numpy.array(shift), scale))
        if numpy.isnan(scale) or scale>3:
            scale = 1
  
        for gesture in gesturesList:
            gestureID,startFrame,endFrame=gesture
            ## extract first 4 frames
            cuboid = numpy.zeros(( IM_SZ, IM_SZ, 4), numpy.uint8)
            frame_count_temp = 0 
            if startFrame-5 > 1:
                valid_count = 0
                for x in range(startFrame-5, startFrame-1):
                    [img, flag] = smp.get_shift_scale_depth_sk_normalize(shift, scale, x, IM_SZ, show_flag=True)
                    cuboid[:, :, frame_count_temp] = img
                    frame_count_temp +=1  
                    if flag:# we actually detect some region
                        valid_count += 1
                if valid_count==4: # all 4 frames are valid
                    Feature_all[:,:,:,cuboid_count] = cuboid
                    Targets[cuboid_count] = 201
                    cuboid_count += 1

            ## extract last 5 frames
            cuboid = numpy.zeros(( IM_SZ, IM_SZ, 4), numpy.uint8)
            frame_count_temp = 0 
            if endFrame+5 < smp.getNumFrames():
                valid_count = 0
                for x in range(endFrame+1, endFrame+5):
                    [img, flag] = smp.get_shift_scale_depth_sk_normalize(shift, scale, x, IM_SZ, show_flag=True)
                    cuboid[:, :, frame_count_temp] = img
                    frame_count_temp +=1  
                    if flag:# we actually detect some region
                        valid_count += 1
                if valid_count==4: # all 4 frames are valid
                    Feature_all[:,:,:,cuboid_count] = cuboid
                    Targets[cuboid_count] = 201
                    cuboid_count += 1

        # ###############################################
        ## delete the sample
        del smp        
        print "Elapsed time %d sec" % int(time.time() - time_tic)
                                          

        if(not file_count%20 and file_count>0) or file_count==649:
            random_idx = numpy.random.permutation(range(cuboid_count))
            save_dir = '.\ConvNet_3DCNN\storage_neutral_sk_normalize\\'
            print "batch: %d" % batch_num
            data_temp = Feature_all[:,:,:,random_idx[:]]
            data_id = Targets[random_idx[:]]
            
            save_path= os.path.join(save_dir,'data_batch_neutral_'+str(batch_num))
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

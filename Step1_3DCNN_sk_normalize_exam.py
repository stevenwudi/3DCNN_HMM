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
from utils import Extract_feature
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
batch_num = 1

# pre-allocating the memory
IM_SZ = 90

Feature_all =  numpy.zeros(shape=(IM_SZ, IM_SZ, 4, 20000), dtype=numpy.uint8)
Targets = numpy.zeros(shape=(100000, 1), dtype=numpy.uint8)
cuboid_count = 0
###############################


for file_count, file in enumerate(samples):

    time_tic = time.time()      
    if (file_count<650 and file_count!=155  and file_count!=538  and file_count>567):
        print("Processing file " + file)
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
        #shift, scale = smp.get_shift_scale(template, ref_depth, start_frame=total_frame-100, end_frame=total_frame-10)
        #shift, scale = smp.get_shift_scale_user_sk( start_frame=total_frame-100, end_frame=total_frame-10)
        shift, scale = smp.get_shift_scale_sk( start_frame=total_frame-100, end_frame=total_frame-10)
        print ("shift: {} scaling: {:.4}".format(numpy.array(shift), scale))
        if numpy.isnan(scale):
            scale = 1
        for gesture in [gesturesList[0]]: # for examine purposes
            gestureID,startFrame,endFrame=gesture
            cuboid = numpy.zeros((IM_SZ, IM_SZ, endFrame-startFrame+1), numpy.uint8)
            frame_count_temp = 0 
            for x in range(startFrame, endFrame):
                depth_original = smp.getDepth(x)
                mask = numpy.mean(smp.getUser(x), axis=2) > 150
                if mask.sum() < 1000: # Kinect detect nothing
                    print "skip "+ str(x)
                else:
                    depth_user = depth_original * mask
                    depth_user_median = cv2.medianBlur(depth_user,5)

                    depth_user_normalized = depth_user_median * 1.0 / float(smp.data['maxDepth'])
                    depth_user_normalized = depth_user_normalized *255 /depth_user_normalized.max()

                    rows,cols = depth_user_normalized.shape
                    M = numpy.float32([[1,0, shift[1]],[0,1, shift[0]]])
                    affine_image = cv2.warpAffine(depth_user_normalized, M,(cols, rows))
                    resize_image = imresize(affine_image, scale)


                    rows, cols = resize_image.shape
                    image_crop = resize_image[max(0,rows/2-200):min(rows/2+160,rows), max(9,cols/2-180):min(cols,cols/2+180)] # we move up a bit
                    cuboid[:, :, frame_count_temp] =  imresize(image_crop, (IM_SZ,IM_SZ))
                    frame_count_temp +=1
                    if True: # show the segmented images here
                        cv2.imshow('image',image_crop)
                        cv2.waitKey(10)
            fr_no = frame_count_temp -1
            for frame in range(fr_no-3):
                Feature_all[:,:,:,cuboid_count] = cuboid[:,:, frame:frame+4]
                state_no = numpy.floor(frame*(STATE_NO*1.0/(fr_no-3)))
                Targets[cuboid_count] = state_no+STATE_NO*(gestureID-1)
                cuboid_count += 1
        # ###############################################
        ## delete the sample
        del smp        
        print "Elapsed time %d sec" % int(time.time() - time_tic)
                                          







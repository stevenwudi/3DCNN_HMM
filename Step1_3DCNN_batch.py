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
# load prestore template
ref_depth = numpy.load('distance_median.npy')
template = cv2.imread('template.png')
template = numpy.mean(template, axis=2)
template /= template.max()

for file_count, file in enumerate(samples):

    time_tic = time.time()      
    if (file_count<650 and file_count>-1):
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
        shift, scale = smp.get_shift_scale(template, ref_depth, start_frame=total_frame-100, end_frame=total_frame-10)
        if numpy.isnan(scale):
            scale = 1


        for gesture in gesturesList:
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

                    depth_user_normalized = depth_user * 1.0 / float(smp.data['maxDepth'])
                    depth_user_normalized = depth_user_normalized *255 /depth_user_normalized.max()

                    rows,cols = depth_user_normalized.shape
                    M = numpy.float32([[1,0, shift[1]],[0,1, shift[0]]])
                    affine_image = cv2.warpAffine(depth_user_normalized, M,(cols, rows))
                    resize_image = imresize(affine_image, scale)
                    resize_image_median = cv2.medianBlur(resize_image,5)

                    rows, cols = resize_image_median.shape
                    image_crop = resize_image_median[rows/2-160:rows/2+160, cols/2-160:cols/2+160]
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
                                          

        if(not file_count%20 and file_count>0) or file_count==649:
            random_idx = numpy.random.permutation(range(cuboid_count))
            save_dir = '.\storage\\'
            print "batch: %d" % batch_num
            data_temp = Feature_all[:,:,:,random_idx[:]]
            data_id = Targets[random_idx[:]]
            
            save_path= os.path.join(save_dir,'data_batch_'+str(batch_num))
            out_file = open(save_path, 'wb')
            dic = {'batch_label':['batch 1 of'+ str(batch_num)], 'data':data_temp, 
                            'data_id':data_id}
            cPickle.dump(dic, out_file, protocol=cPickle.HIGHEST_PROTOCOL)
            out_file.close()

            # pre-allocating the memory
            batch_num += 1
            Feature_all =  numpy.zeros(shape=(IM_SZ, IM_SZ, 4, 20000), dtype=numpy.uint8)
            Targets = numpy.zeros(shape=(20000, 1), dtype=numpy.uint8)
            cuboid_count = 0






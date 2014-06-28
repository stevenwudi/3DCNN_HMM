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
############### viterbi path import
sys.path.append(r'I:\Kaggle_multimodal\StartingKit_track3\Final_project')
from utils import viterbi_path, viterbi_path_log
from utils import viterbi_colab_clean
from utils import createSubmisionFile
from utils import imdisplay
from ChalearnLAPEvaluation import evalGesture,exportGT_Gesture
from ChalearnLAPSample import GestureSample

from shownet_codalab import ShowConvNet
""" Main script. Show how to perform all competition steps
    Access the sample information to learn a model. """
# Data folder (Training data)
print("Extracting the training files")
data_path=os.path.join("I:\Kaggle_multimodal\Training\\")  
# Get the list of training samples
samples=os.listdir(data_path)
STATE_NO = 10


##################################
### load the pre-store normalization constant
## Load Prior and transitional Matrix
dic=sio.loadmat('Transition_matrix.mat')
Transition_matrix = dic['Transition_matrix']
Prior = dic['Prior']

outPred='./training/pred/'
####################################
##################load CNN here######
import getopt as opt
from gpumodel import IGPUModel
from options import *

op = ShowConvNet.get_options_parser()
op.options['load_file'].value=r'.\tmp\tmp\ConvNet__2014-05-14_21.42.41'
op.options['feature_path'].value=r'.\prediction_feature'
op.options['test_batch_range'].value=1
op.options['write_features'].value ='probs'
load_dic =  IGPUModel.load_checkpoint(op.options["load_file"].value)
old_op = load_dic["op"]
old_op.merge_from(op)
op = old_op
op.eval_expr_defaults()
#op, load_dic = IGPUModel.parse_options(op)
op.options['train_batch_range'].value=[1]
op.options['test_batch_range'].value=[1]
model = ShowConvNet(op, load_dic)
meta = pickle.load(open(r'.\storage\batches.meta'))
data_mean = meta['data_mean']

###############################
# load prestore template
ref_depth = numpy.load('distance_median.npy')
template = cv2.imread('template.png')
template = numpy.mean(template, axis=2)
template /= template.max()

# pre-allocating the memory
IM_SZ = 90

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
        shift, scale = smp.get_shift_scale(template, ref_depth, start_frame=total_frame-100, end_frame=total_frame-10)
        if numpy.isnan(scale):
            scale = 1


        cuboid = numpy.zeros((IM_SZ, IM_SZ, total_frame), numpy.uint8)
        frame_count_temp = 0 
        for x in range(1, smp.getNumFrames()):
            depth_original = smp.getDepth(x)
            mask = numpy.mean(smp.getUser(x), axis=2) > 150
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
            if False: # show the segmented images here
                cv2.imshow('image',image_crop)
                cv2.waitKey(10)

     
        Feature_all = numpy.zeros(( IM_SZ, IM_SZ, 4, cuboid.shape[-1]-3), numpy.uint8)
        for frame in range(cuboid.shape[-1]-3):
            Feature_all[:,:,:,frame] = cuboid[:,:, frame:frame+4]

        ### now the crucial part to write feature here###########
        data = numpy.array(Feature_all, dtype=numpy.single)

        #(data dimensionality)x(number of cases). !!!!!!!!!!!!!!
        data = numpy.transpose(data, (2,0,1,3))
        data = numpy.reshape(data, (4*90*90, data.shape[-1]))
        data = numpy.array(data - data_mean[:,numpy.newaxis], dtype=numpy.single)
        ftrs = model.do_write_features(data)

        ##########################
        # viterbi path decoding
        #####################
        log_observ_likelihood = log(ftrs.T) 
        log_observ_likelihood[-1, 0:5] = 0
        log_observ_likelihood[-1, -5:] = 0

        print("\t Viterbi path decoding " )
        # do it in log space avoid numeric underflow
        [path, predecessor_state_index, global_score] = viterbi_path_log(log(Prior), log(Transition_matrix), log_observ_likelihood)
        #[path, predecessor_state_index, global_score] =  viterbi_path(Prior, Transition_matrix, observ_likelihood)
        
        # Some gestures are not within the vocabulary
        [pred_label, begin_frame, end_frame, Individual_score, frame_length] = viterbi_colab_clean(path, global_score, threshold=-100, mini_frame=19)

        #begin_frame = begin_frame + 1
        end_frame = end_frame + 5 # note here 3DCNN should add 5 frames because we used 4 frames
        ### plot the path and prediction
        if False:
            im  = imdisplay(global_score)
            plt.imshow(im, cmap='gray')
            plt.plot(range(global_score.shape[-1]), path, color='c',linewidth=2.0)
            plt.xlim((0, global_score.shape[-1]))
            # plot ground truth
            gesturesList=smp.getGestures()
            for gesture in gesturesList:
            # Get the gesture ID, and start and end frames for the gesture
                gestureID,startFrame,endFrame=gesture
                frames_count = numpy.array(range(startFrame, endFrame+1))
                pred_label_temp = ((gestureID-1) *10 +5) * numpy.ones(len(frames_count))
                plt.plot(frames_count, pred_label_temp, color='r', linewidth=5.0)
            
            # plot clean path
            for i in range(len(begin_frame)):
                frames_count = numpy.array(range(begin_frame[i], end_frame[i]+1))
                pred_label_temp = ((pred_label[i]-1) *10 +5) * numpy.ones(len(frames_count))
                plt.plot(frames_count, pred_label_temp, color='#ffff00', linewidth=2.0)

            plt.show()
        else:
            print "Elapsed time %d sec" % int(time.time() - time_tic)

            pred=[]
            for i in range(len(begin_frame)):
                pred.append([ pred_label[i], begin_frame[i], end_frame[i]] )

            smp.exportPredictions(pred,outPred)

     # ###############################################
        ## delete the sample
        del smp          
        

  

TruthDir='./training/gt/'
final_score = evalGesture(outPred,TruthDir)         
print("The score for this prediction is " + "{:.12f}".format(final_score))
# Submision folder (output)
outSubmision='./training/submision/'
# Prepare submision file (only for validation and final evaluation data sets)
createSubmisionFile(outPred, outSubmision)




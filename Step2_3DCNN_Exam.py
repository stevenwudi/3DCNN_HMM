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

####################################
##################load CNN here######
sys.path.append(r'I:\Kaggle_multimodal\StartingKit_track3\Final_project\ConvNet_3DCNN')
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

outPred='./ConvNet_3DCNN/training/pred_sk_norm/'
####################################
##################load CNN here######
import getopt as opt
from gpumodel import IGPUModel
from options import *
Flag_multiview = 0

op = ShowConvNet.get_options_parser()
op.options['load_file'].value=r'.\ConvNet_3DCNN\tmp\ConvNet__2014-05-28_01.59.00'
op.options['write_features'].value ='probs'
load_dic =  IGPUModel.load_checkpoint(op.options["load_file"].value)
old_op = load_dic["op"]
old_op.merge_from(op)
op = old_op
op.eval_expr_defaults()
op.options['train_batch_range'].value=[1]
op.options['test_batch_range'].value=[1]
op.options['data_path'].value=r'.\ConvNet_3DCNN\storage_sk_final'

model = ShowConvNet(op, load_dic)
model.crop_border = 0


meta = pickle.load(open(r'.\ConvNet_3DCNN\storage_sk_final\batches.meta'))
data_mean = meta['data_mean']


# pre-allocating the memory
IM_SZ = 90

debug_show = True

for file_count, file in enumerate(samples):        
    if  not file_count<652:
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
        #for x in range(1, smp.getNumFrames()):
        for x in range(150+28*4, 300, 4):
            [img, flag] = smp.get_shift_scale_depth_sk_normalize(shift, scale, x, IM_SZ, show_flag=False)
            cuboid[:, :, frame_count_temp] =  img
            frame_count_temp +=1
            if False: # show the segmented images here
                #print x
                #cv2.imshow('image', img)
                #cv2.waitKey(1) 
                import matplotlib.pyplot as plt
                plt.imshow(img)
                plt.show()              

        Feature_all = numpy.zeros(( IM_SZ, IM_SZ, 4, cuboid.shape[-1]-3), numpy.uint8)
        for frame in range(cuboid.shape[-1]-3):
            Feature_all[:,:,:,frame] = cuboid[:,:, frame:frame+4]

        ### now the crucial part to write feature here###########
        data = numpy.array(Feature_all, dtype=numpy.single)

        #(data dimensionality)x(number of cases). !!!!!!!!!!!!!!
        data = numpy.transpose(data, (2,0,1,3))
        data = numpy.reshape(data, (4*90*90, data.shape[-1]))
        data = numpy.array(data - data_mean[:,numpy.newaxis], dtype=numpy.single)

        if Flag_multiview:#self.multiview:
            data = data.reshape((4,90,90, data.shape[-1]))
            target = numpy.zeros((4*82*82,  data.shape[-1]*5), dtype=numpy.single)
            border_size = 4
            start_positions = [(0,0),  (0, border_size*2),
                                (border_size, border_size),
                                (border_size*2, 0), (border_size*2, border_size*2)]
            end_positions = [(sy+82, sx+82) for (sy,sx) in start_positions]
            for i in xrange(5):
                pic = data[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                target[:,i * data.shape[-1]:(i+1)* data.shape[-1]] = pic.reshape((4*82*82, data.shape[-1]))
            data = numpy.require(target,  dtype=numpy.single)

        #ftrs = model.do_write_features(data)

 

        if True:
            data = data.reshape((4,90,90, data.shape[-1]))
            target = data[:,4:86, 4:86,:]
            data = target.reshape((4*82*82,data.shape[-1]))

        ftrs = model.do_write_features_conv1(data)
        

        ##########################
        # viterbi path decoding
        #####################
        if Flag_multiview:
            accumulate_prob = numpy.zeros((ftrs.shape[0]/5, ftrs.shape[1]))
            for i in range(5):
                accumulate_prob += ftrs[i*ftrs.shape[0]/5:(i+1)*ftrs.shape[0]/5,:]

            accumulate_prob = accumulate_prob/5.0
            log_observ_likelihood = log(accumulate_prob.T + numpy.finfo(numpy.float32).eps) 
        else:
            log_observ_likelihood = log(ftrs.T + numpy.finfo(numpy.float32).eps) 
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
        if True:
            im  = imdisplay(global_score)
            plt.clf()
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

        ### save the result
        print "Elapsed time %d sec" % int(time.time() - time_tic)

     # ###############################################
        ## delete the sample
        del smp          
        

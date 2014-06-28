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

outPred='./ConvNet_3DCNN/training/pred/'
####################################
##################load CNN here######
import getopt as opt
from gpumodel import IGPUModel
from options import *

op = ShowConvNet.get_options_parser()
#op.options['load_file'].value=r'I:\Kaggle_multimodal\Code_for_submission\Final_project\ConvNet_3DCNN\tmp\ConvNet__2014-05-28_01.59.00'
op.options['load_file'].value=r'I:\Kaggle_multimodal\Code_for_submission\Final_project\ConvNet_3DCNN\tmp\ConvNet__2014-06-18_01.39.52'
Flag_multiview = 0

#op.options['test_batch_range'].value=1
op.options['write_features'].value ='probs'
load_dic =  IGPUModel.load_checkpoint(op.options["load_file"].value)
old_op = load_dic["op"]
old_op.merge_from(op)
op = old_op
op.eval_expr_defaults()
#op, load_dic = IGPUModel.parse_options(op)
op.options['train_batch_range'].value=[1]
op.options['test_batch_range'].value=[1]
op.options['data_path'].value=r'.\ConvNet_3DCNN\storage'
model = ShowConvNet(op, load_dic)

meta = pickle.load(open(r'.\ConvNet_3DCNN\storage\batches.meta'))
data_mean = meta['data_mean']

###############################
# load prestore template
ref_depth = numpy.load('distance_median.npy')
template = cv2.imread('template.png')
template = numpy.mean(template, axis=2)
template /= template.max()

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
        shift, scale = smp.get_shift_scale(template, ref_depth, start_frame=total_frame-20, end_frame=total_frame-10, debug_show=False)
        if numpy.isnan(scale):
            scale = 1.0

        cuboid = numpy.zeros((IM_SZ, IM_SZ, total_frame), numpy.uint8)
        frame_count_temp = 0 
        for x in range(1, smp.getNumFrames()):
            ### If we do it on a frame basis, it will be very expensive
            #print "frame is "+str(x)
            #shift, scale, depth_user = smp.get_shift_scale(template, ref_depth, start_frame=x, end_frame=x+1)
            #if numpy.isnan(scale):
            #    scale = 1.0
            #    depth_user_normalized = numpy.zeros(shape=(480,640))
            depth_original = smp.getDepth(x)
            mask = numpy.mean(smp.getUser(x), axis=2) > 150
            depth_user = depth_original * mask
            if  mask.sum() < 1000:
                depth_user_normalized = numpy.zeros(shape=(480,640))
            else:
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
                print x
                startY, startX = numpy.random.randint(0,4*2 + 1),numpy.random.randint(0,4*2 + 1)
                endY, endX = startY + 82, startX + 82
                pic = cuboid[startY:endY,startX:endX, frame_count_temp-1]
                cv2.imshow('image', pic)
                cv2.waitKey(1)
                

     
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

        ftrs = model.do_write_features(data)

        ##########################
        # viterbi path decoding
        #####################
        if Flag_multiview:
            accumulate_prob = numpy.zeros((ftrs.shape[0]/5, ftrs.shape[1]))
            for i in range(5):
                accumulate_prob += ftrs[i*ftrs.shape[0]/5:(i+1)*ftrs.shape[0]/5,:]

            accumulate_prob = accumulate_prob/5
            log_observ_likelihood = log(accumulate_prob.T) 
        else:
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

            from pylab import *
            if True:
                save_dir=r'.\training\Depth_path'
                save_path= os.path.join(save_dir,file)
                savefig(save_path, bbox_inches='tight')
            else: 
                plt.show()

        ### save the result
        print "Elapsed time %d sec" % int(time.time() - time_tic)

        save_dir=r'.\training\Depth'
        save_path= os.path.join(save_dir,file)
        out_file = open(save_path, 'wb')
        dic = {'log_observ_likelihood':log_observ_likelihood}
        cPickle.dump(dic, out_file, protocol=cPickle.HIGHEST_PROTOCOL)
        out_file.close()


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
#The score for this prediction is 0.637848231042 -3DCNN  ConvNet__2014-05-28_01.59.00

#Sample ID: Sample0651, score 0.699171
#Sample ID: Sample0652, score 0.707909
#Sample ID: Sample0653, score 0.915447
#Sample ID: Sample0654, score 0.566418
#Sample ID: Sample0655, score 0.756774
#Sample ID: Sample0656, score 0.698210
#Sample ID: Sample0657, score 0.802261
#Sample ID: Sample0658, score 0.707345
#Sample ID: Sample0659, score 0.684510
#Sample ID: Sample0660, score 0.513607
#Sample ID: Sample0661, score 0.799875
#Sample ID: Sample0662, score 0.392395
#Sample ID: Sample0663, score 0.891096
#Sample ID: Sample0664, score 0.943837
#Sample ID: Sample0665, score 0.403225
#Sample ID: Sample0666, score 0.529919
#Sample ID: Sample0667, score 0.587213
#Sample ID: Sample0668, score 0.767841
#Sample ID: Sample0669, score 0.863379
#Sample ID: Sample0670, score 0.538094
#Sample ID: Sample0671, score 0.201223-------------------
#Sample ID: Sample0672, score 0.348593
#Sample ID: Sample0673, score 0.400181
#Sample ID: Sample0674, score 0.602465
#Sample ID: Sample0675, score 0.092397-----------------------
#Sample ID: Sample0676, score 0.728147
#Sample ID: Sample0677, score 0.558078
#Sample ID: Sample0678, score 0.772195
#Sample ID: Sample0679, score 0.863033
#Sample ID: Sample0680, score 0.620209
#Sample ID: Sample0681, score 0.310052
#Sample ID: Sample0682, score 0.872131
#Sample ID: Sample0683, score 0.598099
#Sample ID: Sample0684, score 0.657778
#Sample ID: Sample0685, score 0.511750
#Sample ID: Sample0686, score 0.713403
#Sample ID: Sample0687, score 0.358252
#Sample ID: Sample0688, score 0.764004
#Sample ID: Sample0689, score 0.849279
#Sample ID: Sample0690, score 0.621822
#Sample ID: Sample0691, score 0.883785
#Sample ID: Sample0692, score 0.190132----------------------
#Sample ID: Sample0693, score 0.779292
#Sample ID: Sample0694, score 0.754025
#Sample ID: Sample0695, score 0.551357
#Sample ID: Sample0696, score 0.258955---------------------
#Sample ID: Sample0697, score 0.801815
#Sample ID: Sample0698, score 0.860441
#Sample ID: Sample0699, score 0.241819-----------------------
#Sample ID: Sample0700, score 0.777985
#The score for this prediction is 0.626224524197


######################################################
######CONVNET  -          ConvNet__2014-05-26_03.40.18
#Extracting the training files
#=========================
#Importing pyconvnet C++ module
#	 Processing file Sample0651
#[x, y, shift, distance, scaling]
#[160, 312, [80, 8], 2578.5999999999999, 1.3282913414303703]
#	 Processing file Sample0652
#[x, y, shift, distance, scaling]
#[259, 344, [-19, -24], 2136.0, 1.1002987300454785]
#	 Viterbi path decoding 
#Elapsed time 132 sec
#	 Processing file Sample0653
#[x, y, shift, distance, scaling]
#[214, 335, [26, -15], 1911.0, 0.98439647617832848]
#	 Viterbi path decoding 
#Elapsed time 120 sec
#	 Processing file Sample0654
#[x, y, shift, distance, scaling]
#[190, 335, [50, -15], 2786.0, 1.435127463439468]
#	 Viterbi path decoding 
#Elapsed time 196 sec
#	 Processing file Sample0655
#[x, y, shift, distance, scaling]
#[204, 309, [36, 11], 2473.0, 1.2738945502820544]
#	 Viterbi path decoding 
#Elapsed time 164 sec
#	 Processing file Sample0656
#[x, y, shift, distance, scaling]
#[211, 315, [29, 5], 1840.5999999999999, 0.94813194874611784]
#	 Viterbi path decoding 
#Elapsed time 120 sec
#	 Processing file Sample0657
#[x, y, shift, distance, scaling]
#[223, 304, [17, 16], 1802.2, 0.92835129741945766]
#	 Viterbi path decoding 
#Elapsed time 125 sec
#	 Processing file Sample0658
#[x, y, shift, distance, scaling]
#[188, 342, [52, -22], 2066.8000000000002, 1.0646523479672263]
#	 Viterbi path decoding 
#Elapsed time 115 sec
#	 Processing file Sample0659
#[x, y, shift, distance, scaling]
#[226, 324, [14, -4], 2171.9000000000001, 1.1187915785513929]
#	 Viterbi path decoding 
#Elapsed time 135 sec
#	 Processing file Sample0660
#[x, y, shift, distance, scaling]
#[192, 307, [48, 13], 2852.0, 1.4691254579071653]
#	 Viterbi path decoding 
#Elapsed time 193 sec
#	 Processing file Sample0661
#[x, y, shift, distance, scaling]
#[218, 321, [22, -1], 1745.3, 0.89904090521927604]
#	 Viterbi path decoding 
#Elapsed time 136 sec
#	 Processing file Sample0662
#[x, y, shift, distance, scaling]
#[194, 314, [46, 6], 2907.0, 1.4974571199635798]
#	 Viterbi path decoding 
#Elapsed time 198 sec
#	 Processing file Sample0663
#[x, y, shift, distance, scaling]
#[235, 317, [5, 3], 1943.0, 1.0008803522838787]
#	 Viterbi path decoding 
#Elapsed time 116 sec
#	 Processing file Sample0664
#[x, y, shift, distance, scaling]
#[218, 355, [22, -35], 2139.3000000000002, 1.1019986297688635]
#	 Viterbi path decoding 
#Elapsed time 122 sec
#	 Processing file Sample0665
#[x, y, shift, distance, scaling]
#[198, 342, [42, -22], 3012.5999999999999, 1.5518539111118954]
#	 Viterbi path decoding 
#Elapsed time 210 sec
#	 Processing file Sample0666
#[x, y, shift, distance, scaling]
#[169, 305, [71, 15], 2216.0, 1.1415084203093542]
#	 Viterbi path decoding 
#Elapsed time 142 sec
#	 Processing file Sample0667
#[x, y, shift, distance, scaling]
#[198, 342, [42, -22], 3051.0, 1.5716345624385557]
#	 Viterbi path decoding 
#Elapsed time 222 sec
#	 Processing file Sample0668
#[x, y, shift, distance, scaling]
#[201, 305, [39, 15], 2545.0, 1.3109832715195426]
#	 Viterbi path decoding 
#Elapsed time 189 sec
#	 Processing file Sample0669
#[x, y, shift, distance, scaling]
#[240, 333, [0, -13], 1983.0, 1.0214851974158166]
#	 Viterbi path decoding 
#Elapsed time 127 sec
#	 Processing file Sample0670
#[x, y, shift, distance, scaling]
#[195, 307, [45, 13], 2907.0, 1.4974571199635798]
#	 Viterbi path decoding 
#Elapsed time 224 sec
#	 Processing file Sample0671
#[x, y, shift, distance, scaling]
#[197, 336, [43, -16], 2570.5999999999999, 1.3241703724039828]
#	 Viterbi path decoding 
#Elapsed time 128 sec
#	 Processing file Sample0672
#[x, y, shift, distance, scaling]
#[244, 339, [-4, -19], 2980.5999999999999, 1.5353700350063453]
#	 Viterbi path decoding 
#Elapsed time 126 sec
#	 Processing file Sample0673
#[x, y, shift, distance, scaling]
#[194, 314, [46, 6], 2920.6000000000004, 1.5044627673084388]
#	 Viterbi path decoding 
#Elapsed time 218 sec
#	 Processing file Sample0674
#[x, y, shift, distance, scaling]
#[161, 323, [79, -3], 2081.5999999999999, 1.072276140666043]
#	 Viterbi path decoding 
#Elapsed time 151 sec
#	 Processing file Sample0675
#[x, y, shift, distance, scaling]
#[205, 375, [35, -55], 2076.3000000000002, 1.0695459986860616]
#	 Viterbi path decoding 
#Elapsed time 88 sec
#	 Processing file Sample0676
#[x, y, shift, distance, scaling]
#[243, 351, [-3, -31], 1847.0, 0.951428723967228]
#	 Viterbi path decoding 
#Elapsed time 152 sec
#	 Processing file Sample0677
#[x, y, shift, distance, scaling]
#[225, 315, [15, 5], 1858.2, 0.95719808060417055]
#	 Viterbi path decoding 
#Elapsed time 143 sec
#	 Processing file Sample0678
#[x, y, shift, distance, scaling]
#[199, 336, [41, -16], 2115.1999999999998, 1.0895842105768709]
#	 Viterbi path decoding 
#Elapsed time 142 sec
#	 Processing file Sample0679
#[x, y, shift, distance, scaling]
#[204, 336, [36, -16], 2168.0, 1.1167826061510289]
#	 Viterbi path decoding 
#Elapsed time 147 sec
#	 Processing file Sample0680
#[x, y, shift, distance, scaling]
#[172, 289, [68, 31], 1606.0, 0.82728453204730268]
#	 Viterbi path decoding 
#Elapsed time 137 sec
#	 Processing file Sample0681
#[x, y, shift, distance, scaling]
#[234, 326, [6, -6], 2168.0, 1.1167826061510289]
#	 Viterbi path decoding 
#Elapsed time 212 sec
#	 Processing file Sample0682
#[x, y, shift, distance, scaling]
#[218, 276, [22, 44], 1726.0, 0.88909906744311606]
#	 Viterbi path decoding 
#Elapsed time 135 sec
#	 Processing file Sample0683
#[x, y, shift, distance, scaling]
#[212, 347, [28, -27], 1943.0, 1.0008803522838787]
#	 Viterbi path decoding 
#Elapsed time 119 sec
#	 Processing file Sample0684
#[x, y, shift, distance, scaling]
#[214, 346, [26, -26], 2057.0, 1.0596041609099014]
#	 Viterbi path decoding 
#Elapsed time 112 sec
#	 Processing file Sample0685
#[x, y, shift, distance, scaling]
#[222, 324, [18, -4], 2128.8000000000002, 1.0965898579217299]
#	 Viterbi path decoding 
#Elapsed time 108 sec
#	 Processing file Sample0686
#[x, y, shift, distance, scaling]
#[190, 353, [50, -33], 1959.0, 1.0091222903366539]
#	 Viterbi path decoding 
#Elapsed time 174 sec
#	 Processing file Sample0687
#[x, y, shift, distance, scaling]
#[167, 310, [73, 10], 1606.0, 0.82728453204730268]
#	 Viterbi path decoding 
#Elapsed time 121 sec
#	 Processing file Sample0688
#[x, y, shift, distance, scaling]
#[202, 346, [38, -26], 2136.0, 1.1002987300454785]
#	 Viterbi path decoding 
#Elapsed time 135 sec
#	 Processing file Sample0689
#[x, y, shift, distance, scaling]
#[192, 327, [48, -7], 2248.0, 1.1579922964149043]
#	 Viterbi path decoding 
#Elapsed time 137 sec
#	 Processing file Sample0690
#[x, y, shift, distance, scaling]
#[187, 248, [53, 72], 1687.8, 0.86942144034211555]
#	 Viterbi path decoding 
#Elapsed time 111 sec
#	 Processing file Sample0691
#[x, y, shift, distance, scaling]
#[218, 324, [22, -4], 2104.0, 1.0838148539399284]
#	 Viterbi path decoding 
#Elapsed time 127 sec
#	 Processing file Sample0692
#[x, y, shift, distance, scaling]
#[217, 385, [23, -65], 1635.9000000000001, 0.84268665378342622]
#	 Viterbi path decoding 
#Elapsed time 133 sec
#	 Processing file Sample0693
#[x, y, shift, distance, scaling]
#[206, 321, [34, -1], 2537.0, 1.3068623024931549]
#	 Viterbi path decoding 
#Elapsed time 170 sec
#	 Processing file Sample0694
#[x, y, shift, distance, scaling]
#[196, 324, [44, -4], 1910.8, 0.98429345195266871]
#	 Viterbi path decoding 
#Elapsed time 131 sec
#	 Processing file Sample0695
#[x, y, shift, distance, scaling]
#[202, 316, [38, 4], 2473.0, 1.2738945502820544]
#	 Viterbi path decoding 
#Elapsed time 170 sec
#	 Processing file Sample0696
#[x, y, shift, distance, scaling]
#[248, 343, [-8, -23], 3170.4000000000001, 1.6331400251573902]
#	 Viterbi path decoding 
#Elapsed time 131 sec
#	 Processing file Sample0697
#[x, y, shift, distance, scaling]
#[258, 342, [-18, -22], 2168.0, 1.1167826061510289]
#	 Viterbi path decoding 
#Elapsed time 130 sec
#	 Processing file Sample0698
#[x, y, shift, distance, scaling]
#[198, 322, [42, -2], 1935.0, 0.99675938325749114]
#	 Viterbi path decoding 
#Elapsed time 131 sec
#	 Processing file Sample0699
#[x, y, shift, distance, scaling]
#[223, 295, [17, 25], 2069.5999999999999, 1.0660946871264618]
#	 Viterbi path decoding 
#Elapsed time 160 sec
#	 Processing file Sample0700
#[x, y, shift, distance, scaling]
#[168, 332, [72, -12], 2407.4000000000001, 1.2401026042656766]
#	 Viterbi path decoding 
#Elapsed time 172 sec
#Sample ID: Sample0651, score 0.663733
#Sample ID: Sample0652, score 0.575386
#Sample ID: Sample0653, score 0.892627
#Sample ID: Sample0654, score 0.528169
#Sample ID: Sample0655, score 0.576693
#Sample ID: Sample0656, score 0.389144
#Sample ID: Sample0657, score 0.374459
#Sample ID: Sample0658, score 0.650195
#Sample ID: Sample0659, score 0.658055
#Sample ID: Sample0660, score 0.576639
#Sample ID: Sample0661, score 0.664821
#Sample ID: Sample0662, score 0.254808
#Sample ID: Sample0663, score 0.796288
#Sample ID: Sample0664, score 0.947266
#Sample ID: Sample0665, score 0.326278
#Sample ID: Sample0666, score 0.508804
#Sample ID: Sample0667, score 0.484710
#Sample ID: Sample0668, score 0.580054
#Sample ID: Sample0669, score 0.849987
#Sample ID: Sample0670, score 0.397313
#Sample ID: Sample0671, score 0.203398
#Sample ID: Sample0672, score 0.391556
#Sample ID: Sample0673, score 0.378563
#Sample ID: Sample0674, score 0.635954
#Sample ID: Sample0675, score 0.045522
#Sample ID: Sample0676, score 0.438792
#Sample ID: Sample0677, score 0.419724
#Sample ID: Sample0678, score 0.655333
#Sample ID: Sample0679, score 0.854239
#Sample ID: Sample0680, score 0.714220
#Sample ID: Sample0681, score 0.272786
#Sample ID: Sample0682, score 0.053125
#Sample ID: Sample0683, score 0.734950
#Sample ID: Sample0684, score 0.615251
#Sample ID: Sample0685, score 0.510606
#Sample ID: Sample0686, score 0.597038
#Sample ID: Sample0687, score 0.664701
#Sample ID: Sample0688, score 0.804543
#Sample ID: Sample0689, score 0.872862
#Sample ID: Sample0690, score 0.577523
#Sample ID: Sample0691, score 0.792732
#Sample ID: Sample0692, score 0.153347
#Sample ID: Sample0693, score 0.635430
#Sample ID: Sample0694, score 0.634161
#Sample ID: Sample0695, score 0.502847
#Sample ID: Sample0696, score 0.293253
#Sample ID: Sample0697, score 0.703240
#Sample ID: Sample0698, score 0.850709
#Sample ID: Sample0699, score 0.244085
#Sample ID: Sample0700, score 0.750470
#The score for this prediction is 0.553927838114


#########local (bad, bad, bad):  I:\Kaggle_multimodal\Code_for_submission\Final_project\ConvNet_3DCNN\tmp\ConvNet__2014-06-09_00.33.03:   
#The score for this prediction is 0.553927838114                                  0.49
#Sample0651, score 0.645744
#Sample ID: Sample0652, score 0.559796
#Sample ID: Sample0653, score 0.533064
#Sample ID: Sample0654, score 0.485505
#Sample ID: Sample0655, score 0.890764
#Sample ID: Sample0656, score 0.369051
#Sample ID: Sample0657, score 0.049603
#Sample ID: Sample0658, score 0.555414
#Sample ID: Sample0659, score 0.613854
#Sample ID: Sample0660, score 0.311432
#Sample ID: Sample0661, score 0.535761
#Sample ID: Sample0662, score 0.265325
#Sample ID: Sample0663, score 0.920095
#Sample ID: Sample0664, score 0.935104
#Sample ID: Sample0665, score 0.378939
#Sample ID: Sample0666, score 0.461569
#Sample ID: Sample0667, score 0.458477
#Sample ID: Sample0668, score 0.415858
#Sample ID: Sample0669, score 0.849286
#Sample ID: Sample0670, score 0.261954
#Sample ID: Sample0671, score 0.222867
#Sample ID: Sample0672, score 0.247778
#Sample ID: Sample0673, score 0.410968
#Sample ID: Sample0674, score 0.498387
#Sample ID: Sample0675, score 0.048387
#Sample ID: Sample0676, score 0.235188
#Sample ID: Sample0677, score 0.531585
#Sample ID: Sample0678, score 0.403813
#Sample ID: Sample0679, score 0.773155
#Sample ID: Sample0680, score 0.629767
#Sample ID: Sample0681, score 0.309476
#Sample ID: Sample0682, score 0.093147
#Sample ID: Sample0683, score 0.571337
#Sample ID: Sample0684, score 0.596494
#Sample ID: Sample0685, score 0.700286
#Sample ID: Sample0686, score 0.626915
#Sample ID: Sample0687, score 0.336185
#Sample ID: Sample0688, score 0.695073
#Sample ID: Sample0689, score 0.817379
#Sample ID: Sample0690, score 0.617178
#Sample ID: Sample0691, score 0.664088
#Sample ID: Sample0692, score 0.136878
#Sample ID: Sample0693, score 0.602969
#Sample ID: Sample0694, score 0.658753
#Sample ID: Sample0695, score 0.584923
#Sample ID: Sample0696, score 0.124848
#Sample ID: Sample0697, score 0.623503
#Sample ID: Sample0698, score 0.697686
#Sample ID: Sample0699, score 0.245174
#Sample ID: Sample0700, score 0.667301

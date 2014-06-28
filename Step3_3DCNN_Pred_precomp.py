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

STATE_NO = 10
##################################
### load the pre-store normalization constant
## Load Prior and transitional Matrix
dic=sio.loadmat('Transition_matrix.mat')
Transition_matrix = dic['Transition_matrix']
Prior = dic['Prior']


####################################  PER CNN customize
Flag_multiview = 0
CNN_NAME = 'ConvNet__2014-05-26_03.40.18_155'
outPred='./ConvNet_3DCNN/training/Test_3DCNN_' + CNN_NAME
if not os.path.exists(outPred):
    os.makedirs(outPred)   
####################################
##################load CNN here######
import getopt as opt
from gpumodel import IGPUModel
from options import *


op = ShowConvNet.get_options_parser()
#op.options['load_file'].value=r'.\ConvNet_3DCNN\tmp\ConvNet__2014-05-28_01.59.00'
### old
op.options['load_file'].value=r'I:\Kaggle_multimodal\StartingKit_track3\Final_project\ConvNet_3DCNN\tmp\ConvNet__2014-05-26_03.40.18'

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

print("Extracting the training files")
data_path=os.path.join("I:\Kaggle_multimodal\Test\Test\\")  
cnn_path=os.path.join(".\ConvNet_3DCNN\Depth_Precompute_test\\")  
# Get the list of training samples
samples=os.listdir(data_path)
samples_cnn = os.listdir(cnn_path)

for file_count, file in enumerate(samples):        
    if file ==  samples_cnn[file_count]:
        time_tic = time.time()  
        print("\t Processing file " + file)
        # Create the object to access the sample
        smp=GestureSample(os.path.join(data_path,file))
        # ###############################################
        # USE Ground Truth information to learn the model
        # ###############################################
        load_path_cnn= os.path.join(cnn_path,file)
        dic_cnn = cPickle.load( open(load_path_cnn, "rb" ) )
        cuboid = dic_cnn['cuboid']             

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

            from pylab import *
            if True:
                save_dir=r'.\ConvNet_3DCNN\training\Test_Depth_path_'+ CNN_NAME
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path= os.path.join(save_dir,file)
                savefig(save_path, bbox_inches='tight')
                #plt.show()
            else: 
                plt.show()

        ### save the result
        print "Elapsed time %d sec" % int(time.time() - time_tic)

        save_dir=r'.\ConvNet_3DCNN\training\Test_Depth_'+ CNN_NAME
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)            
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
        

  

TruthDir=r'I:\Kaggle_multimodal\ChalearnLAP2104_EvaluateTrack3\input\ref//'
CNN_NAME = 'ConvNet__2014-05-28_01.59.00_150'
outPred='./ConvNet_3DCNN/training/Test_3DCNN_' + CNN_NAME
final_score = evalGesture(outPred,TruthDir)         
print("The score for this prediction is " + "{:.12f}".format(final_score))
# Submision folder (output)
outSubmision='./training/submision/'
# Prepare submision file (only for validation and final evaluation data sets)
createSubmisionFile(outPred, outSubmision)

#ConvNet__2014-06-25_18.21.33_250 The score for this prediction is 0.637141036068
#ConvNet__2014-06-25_18.56.59_162 The score for this prediction is 0.679523343823
#ConvNet__2014-05-28_01.59.00_150 The score for this prediction is 0.702863204283
#'I:\Kaggle_multimodal\StartingKit_track3\Final_project\ConvNet_3DCNN\tmp\ConvNet__2014-05-26_03.40.18'
# The score for this prediction is 0.702863204283
import sys, os,random,numpy,zipfile
from numpy import log
from shutil import copyfile
import matplotlib.pyplot as plt
import cv2
from ChalearnLAPEvaluation import evalGesture,exportGT_Gesture
from ChalearnLAPSample import GestureSample


from utils import normalize
from utils import imdisplay
from utils import viterbi_colab_clean
from utils import createSubmisionFile
import time
import cPickle
import numpy
import scipy.io as sio  

############### viterbi path import
from utils import viterbi_path, viterbi_path_log

#########################

outPred_cropped=os.path.join(r"./ConvNet_3DCNN\training\Test_Depth_ConvNet__2014-05-28_01.59.00_150/")  
outPred_cropped_2=os.path.join(r"./ConvNet_3DCNN\training\Test_Depth_ConvNet__2014-06-25_18.56.59_162/")  
outPred_depth =r'.\ConvNet_3DCNN\training\Test_Depth_ConvNet__2014-06-25_18.21.33_250/'
samples_cropped = os.listdir(outPred_cropped)
samples_cropped_2 = os.listdir(outPred_cropped_2)
samples_depth = os.listdir(outPred_depth)

predPath=r'./training/Test_combined_3DCNN_3/'
if not os.path.exists(predPath):
    os.makedirs(predPath)
## only for ploting
data_path=os.path.join("I:\Kaggle_multimodal\Test\Test\\")  
## Load Prior and transitional Matrix
dic=sio.loadmat('Transition_matrix.mat')
Transition_matrix = dic['Transition_matrix']
Prior = dic['Prior']



for file_count, file in enumerate(samples_cropped):
    if file_count > -1:
        if file ==  samples_depth[file_count] and file == samples_cropped_2[file_count]:
            print("\t Processing file " + file)
            smp=GestureSample(os.path.join(data_path,file))

            time_tic = time.time() 
            load_path_cropped= os.path.join(outPred_cropped,file)
            load_path_cropped_2= os.path.join(outPred_cropped_2,file)
            load_path_depth= os.path.join(outPred_depth,file)

            dic_cropped= cPickle.load( open(load_path_cropped, "rb" ) )
            dic_cropped_2= cPickle.load( open(load_path_cropped_2, "rb" ) )
            dic_depth = cPickle.load( open(load_path_depth, "rb" ) )

            log_observ_likelihood_cropped= dic_cropped['log_observ_likelihood']
            log_observ_likelihood_cropped_2= dic_cropped_2['log_observ_likelihood']
            log_observ_likelihood_depth = dic_depth['log_observ_likelihood']

            log_observ_likelihood = log_observ_likelihood_cropped + log_observ_likelihood_depth + log_observ_likelihood_cropped_2
            print "Viterbi path decoding " + file
            # do it in log space avoid numeric underflow
            [path, predecessor_state_index, global_score] = viterbi_path_log(log(Prior), log(Transition_matrix), log_observ_likelihood)
            #[path, predecessor_state_index, global_score] =  viterbi_path(Prior, Transition_matrix, observ_likelihood)
        
            # Some gestures are not within the vocabulary
            [pred_label, begin_frame, end_frame, Individual_score, frame_length] = viterbi_colab_clean(path, global_score, threshold=-100, mini_frame=19)
            ### In theory we need add frame, but it seems that the groutnd truth is about 3 frames more, a bit random
            end_frame = end_frame + 3

            print "Elapsed time %d sec" % int(time.time() - time_tic)
            prediction=[]
            for i in range(len(begin_frame)):
                prediction.append([ pred_label[i], begin_frame[i], end_frame[i]] )

            if True:
                import matplotlib.pyplot as plt
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
                    save_dir=r'.\ConvNet_3DCNN\training\Test_3Depth_path_combined'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)     
                    save_path= os.path.join(save_dir,file)
                    savefig(save_path, bbox_inches='tight')
                    #plt.show()
                else: 
                    plt.show()

            #def exportPredictions(self, prediction,predPath):
            """ Export the given prediction to the correct file in the given predictions path """
            if not os.path.exists(predPath):
                os.makedirs(predPath)
            output_filename = os.path.join(predPath,  file + '_prediction.csv')
            output_file = open(output_filename, 'wb')
            for row in prediction:
                output_file.write(repr(int(row[0])) + "," + repr(int(row[1])) + "," + repr(int(row[2])) + "\n")
            output_file.close()


TruthDir=r'I:\Kaggle_multimodal\ChalearnLAP2104_EvaluateTrack3\input/ref/'
final_score = evalGesture(predPath,TruthDir)         
print("The score for this prediction is " + "{:.12f}".format(final_score))



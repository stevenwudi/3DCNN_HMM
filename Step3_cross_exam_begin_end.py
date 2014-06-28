from ChalearnLAPEvaluation import evalGesture


predPath=r'.\training\test/'
predPath=r'.\ConvNet_3DCNN\training\Test_3DCNN_ConvNet__2014-06-25_18.21.33_250/'
#predPath=r'I:\Kaggle_multimodal\Code_for_submission\Final_project\ConvNet_3DCNN\training\pred_sk_norm'
#predPath=r'I:\Kaggle_multimodal\StartingKit_track3\Final_project\training\test_combined/'


TruthDir=r'I:\Kaggle_multimodal\ChalearnLAP2104_EvaluateTrack3\input/ref/'
#TruthDir=r'I:\Kaggle_multimodal\Valid_650_labels/'
final_score = evalGesture(predPath,TruthDir, begin_add=0, end_add=0) 
print("The score for this prediction is " + "{:.12f}".format(final_score))

#The score for this prediction is 0.816150551922--combined
#The score for this prediction is 0.787309630209--skeleton
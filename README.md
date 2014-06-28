Purpose
=============
This is the code the challenge"Chalearn Looking at People 2014“.
******************************************************************************************************
Gist:3DCNN(3D Convolutional Neural Networks) + Hidden Markov Networks
******************************************************************************************************
by Di WU: stevenwudi@gmail.com, 2015/05/27



Disclaimer
-------
Since I have little time to improve the performance, hence the project files are all over the place/messy. 
Apologies in advance if that causes any inconvenience for reading and please contact me for missing modules if there is any.
******************************************************************************************************

Dependency:
******************************************************************************************************
(1) Nvidia enabled GPU
			
(2) scikit learn: for Depth image template matching task: http://scikit-learn.org/stable/
	
(3) opencv for python: for depth image scaling and shifting: http://docs.opencv.org/trunk/doc/py_tutorials/py_tutorials.html

(4)	scipy for imresize and some reading matrix functionality

(5) cuda-convnet: for 3D convolutional neural networks	(https://code.google.com/p/cuda-convnet/)
				Note: compiling the library is kind of complicated for windows(and the library needs GPU on your computer)
					I have a blog about compiling the cuda library under windows environment if ever needed:(http://vision.group.shef.ac.uk/wordpress/?p=1)
	
	
Train
-------
To train the network, you first need to extract the relevant information and store the intermediate result:

1) 'Step1_3DCNN_template.py' --> extract template

2) 'Step1_3DCNN_batch.py' and'Step1_3DCNN_neutral_batch.py' --> extract normalized resize images using template matching method
(could also be the 'Step1_3DCNN_normalize_by_sk.py' and 'Step1_3DCNN_sk_normlize_neutral.py' using skeleton for normalization

3) 'Step1_transition_matrix.py' --> extract Transitional Matrix and Prior for Hidden Markov Models

4) 'Main_3DCNN.py' --> the main file for training the network.

Thank you!

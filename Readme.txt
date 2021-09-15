                    Traffic sign and signal recognition:

Traffic sign and signal recognition (TSSR) represents an important feature of
advanced driver assistance systems, contributing to the safety of the drivers,
pedestrians and vehicles as well. Developing TSSR systems requires the use
of computer vision techniques, which could be considered fundamental in the
field of pattern recognition in general. Despite all the previous works and
research that has been achieved, traffic sign detection and recognition still
remain a very challenging problem, precisely if we want to provide a real time
processing solution. We propose an approach for traffic sign and light
detection based on Convolutional Neural Networks (CNN). We first transform
the original image into the gray scale image by using support vector machines,
then use convolutional neural networks with fixed and learnable layers for
detection and recognition. The fixed layer can reduce the amount of interest
areas to detect, and crop the boundaries very close to the borders of traffic
signs. The learnable layers can increase the accuracy of detection significantly. 
Steps To execute:

1. For Running the Image Classsifier:

	1.1 Open command prompt from the directory where the source code is saved
	1.2  Run the python file using the command: "python model.py”
	1.3 After running the above command “ traffic_classifier.h5 “ will be generated.
	1.4 Run the python file using the command: "python ImageClassifier.py"
	1.5 Use upload button to upload the imge which is needed to be classified
	1.6 Click on Classify button to finally predict the label for the image

2. For Running the Image Classsifier:

	2.1 Open command prompt from the directory where the source code is saved
	2.2 Run the python file using the command: "python VideoClassifier.py"


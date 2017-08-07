#**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/new33.jpg "Traffic Sign 1"
[image5]: ./examples/new22.jpg "Traffic Sign 2"
[image6]: ./examples/new25.jpg "Traffic Sign 3"
[image7]: ./examples/new26.jpg "Traffic Sign 4"
[image8]: ./examples/new27.jpg "Traffic Sign 5"


---
###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is 32 x 32 x 3, RGB
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed in each class. I could found that the testing data is distributed proportionally to the distribution of the training data set.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. 

As a first step, I decided to convert the images to grayscale by using inner product between the 3 color values on each pixel and (0.299, 0.587, 0.114) because the color of the image doesn’t seem to matter for traffic sign at this case.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data by the function 0.1 + ((img - 0) * (0.9-0.1) / (255-0), because I need to arrange the input values as the similar level, which will affect to the entire learning rate of my network.

I did not decide to generate additional data because it is likely that I can get better result by adding some data which is tranformed by such as rotating images, considering the translation invariance, however, I found I succeeded to achieve more than 93% validation accuracy when I tried some iteration for customizing and checking accuracy on the model architecture and parameters.  


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution     	| 1x1 stride, outputs 28x28x8 	|
| RELU					|		
| Convolution		| 1x1 stride, outputs 14x14x10	|
| RELU					|
| Convolution		| 1x1 stride, outputs 10x10x24	|										|
| Max pooling	      	| 2x2 stride, outputs 5x5x24	|  				|
| Fully connected		|         									|
| Softmax				|      									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer, though I started from GradientDescentOptimizer at first. AdamOptimizer was faster for learning and better result. 
Also, I set epochs as 20, Batch_size as 128, and learning rate as 0.002, after trying several combinations.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.958
* test set accuracy of 0.940

If an iterative approach was chosen:
* At the beginning, I used Lenet-5 model.
* After trying different parameters and training, I found that Validation accuracy can reach at most 90%, even though Training accuracy is around 99%. It looks overfitting. 
* Then, I tried to make the depth of each convolution layer deeper. It got the validation accuracy to be improved, but still it was at most 92%, even though I make it deeper. 
* Lastly, when I add additional convolution layer, the valudation accuracy got improved, resulted in more than 96%.
* I also modified the parameters such as epoch, batch, and learning rate, for example, when I found that the number of training looked too short to train the model well, I increased the number of training (epoch). 
* I think the important design choices was adding convolution layer, because the issue I faced was mainly overfitting. In other words, Lenet-5 does not have enough capacity for accomodating the complexity of german traffic signs data at this time.
By adding more convolution layer, I think my model can express more complecated classification. At this time, fortunately the computing power for training was enough, but it may be likely that I need to consider additional approach such as dropout / max pooling if I faced what requires huge calculation. 


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first and second images look easy to classify because there is the sign only in the picture with good contrast.
The third to fifth are comparatively difficult because they include some noise, such as including backround image, masking and so on.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			       |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 33 Turn right ahead      		| 33 Turn right ahead	  									| 
| 22 Bumpy Road     			| 22 Bumpy Road 									|
| 25	Road Work      		| 22 Bumpy Road 					 				|
| 27	Pedestrians				| 27	Pedestrians											|
| 28	Children crossing		| 20 Dangerous curve to the right    							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This result is a little bit dissapointing, comparing to the test accuracy 94%. The test data for Image labels (No.20-35) are comparatively smaller set than other NO.s. I think this short of data set is one of cause this situation.    


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first to fourth mage, the model predict specific signis, in other words its probabilities more than 99%. 
The top five soft max probabilities were as follows. I also made bar chart on the Ipython notebook.

For the first image ...

| Image			                |     Prediction	         					| 
|-------------------------|------------------------------| 
| 33 Turn right ahead     | 0.99                									| 
| 39 Keep left     			    | 8.2e-32 	 								           |
| 13 Yield          		    | 8.9e-38 	          				 				 |
| 0	Speed limit (20km/h)		| 0                            |
| 1	Speed limit (30km/h)		| 0                            |



For the second image ... 

| Image			                          |     Prediction					| 
|-----------------------------------|--------------------| 
| 22 Bumpy Road         	          	| 0.99      									| 
| 3 Speed limit (60km/h)     			    | 4.1e-04  	 								|
| 31 Wild animals crossing     		   | 5.4e-05 					 				 |
| 20	Dangerous curve to the right			| 1.2e-07		 									|
| 29	Bicycles crossing		            | 1.4e-08     							|

For the third image ...

| Image			                          |     Prediction					| 
|:---------------------:|:---------------------------------------------:| 
| 22 Bumpy Roadd      		| 0.99  									| 
| 25 Road work    			| 1.2e-04 	 								|
| 29	Bicycles crossing     		| 2.1e-11 					 				 |
| 20	Dangerous curve to the right			| 7.8e-14		 									|
| 24	Road narrows on the right		| 5.4e-14    							|

For the forth image ...

| Image			                          |     Prediction					| 
|:---------------------:|:---------------------------------------------:| 
| 27	Pedestrians      		| 0.99  									| 
| 11 Right-of-way at the next intersection     			| 2.4e-07 	 								|
| 18 Yield     		| 1.2e-10 					 				 |
| 1	Speed limit (30km/h)			| 2.8e-14		 									|
| 30	Beware of ice/snow		| 5.8e-17    							|

For the fifth image ...

| Image			                          |     Prediction					| 
|:---------------------:|:---------------------------------------------:| 
| 20 Dangerous curve to the right      		| 0.99  									| 
| 39 Keep left     			| 2.4e-09 	 								|
| 11 Right-of-way at the next intersection     		| 7.9e-12				 				 |
| 29	Bicycles crossing			| 3.3e-12		 									|
| 25	Road work		| 8.1e-13    							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

It is difficult to find the specific characteristic from these maps, however, I felt immpression that the high value of feature will be gathered around the edge of the traffic sign, if my model is more sophisticated or trained well.




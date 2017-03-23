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
[image4]: ./examples/sign1.png "Traffic Sign 1"
[image5]: ./examples/sign2.png "Traffic Sign 2"
[image6]: ./examples/sign3.png "Traffic Sign 3"
[image7]: ./examples/sign4.png "Traffic Sign 4"
[image8]: ./examples/sign5.png "Traffic Sign 5"
[bar]: ./examples/bar.png "Distribution of classes"
[sign_examples]: ./examples/sign_examples.png "Examples of data"
[download]: ./examples/download.png "Fake data"
[original]: ./examples/original.png "Original picture"
[predictions]: ./examples/predictions.png "Predictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/skandium/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the fourth code cell of the IPython notebook, under "Data Set Summary & Exploration"

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The size of the validation set is 4410
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43
* The number of augmented images added is 173996

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fifth code cell of the IPython notebook, under "Include an exploratory visualization of the dataset"

Here is an exploratory visualization of the data set. It is a bar chart showing the class frequencies. Classes are clearly unbalanced in the dataset.


![alt text][bar]


Here we plot 100 signs from the training set

![alt text][sign_examples]


We see that they are not "perfect" pictures - sign and picture boundaries do not match, they are often blurry, with low brightess and in non-default poses. However, they are relatively well centered.

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.
The code for this step is contained in the seventh and eleventh code cells of the IPython notebook, under "Preprocessing" and "Model architecture".

The only preprocessing step that was tried on the original images was normalization in the following form:

````python
print("Normalizing")
X_train = np.array(X_train / 255.0 - 0.5 )
X_test = np.array(X_test / 255.0 - 0.5 )
X_valid = np.array(X_valid / 255.0 - 0.5 )
````
Surprisingly, it did not help and actually made training the model more difficult (parameters got stuck in a region of low accuracy.)

We can think of grayscaling as a weighted average of the color channels with fixed weights. Instead of this, I use a 1x1x1 convolution over the whole image to transform color information to one dimension. This procedure learns the weights for the transformation and has the added value of providing a nonlinear ReLu layer which adds additional discriminatory power to the classifier.
Transformation into one dimension is done because, as per LeCun's paper, quite counterintuitively the color channels do not add accuracy, their model is more accurate on grayscale images.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Training, validation and test sets **were already set up** when downloading the files from Udacity. The training set consists of 34799 images, test set of 12630 and validation set of 4410 images.
If it were not split already, I would have simply used scikit-learn's train-test split function.

Data augmentation was heavily used in the seventh code cell: for every image, 5 augmented images were created and added to the data set. These help to reduce the generalization error, on the one hand because estimating 8.5 million parameters needs a lot of data and secondly because you introduce the model to more circumstances. I use affine transformations, namely rotation, shearing and translation. All augmented pictures are then multiplied by a random brightness coefficient (they can become darker or lighter). We add 173996 augmented pictures to the training set, for a total size of 208795.
In practice, the augmentation process looks something like this: 
First, we have an original image:

![original][original]

We create 5 fakes:

![fakes][download]

Notably, I took no steps to alleviate unbalanced classes.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the eleventh cell of the ipython notebook, under "Model architecture". 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1x1x1     	| 1x1 stride, same padding, outputs 32x32x1 	|
| ReLu					|												|
| Convolution 3x3x64	    | 1x1 stride, same padding, outputs 32x32x64     |
| ReLu		|       									|
| Convolution 3x3x64	    | 1x1 stride, same padding, outputs 32x32x64     |
| ReLu		|       									|
| Convolution 3x3x64	    | 1x1 stride, same padding, outputs 32x32x64     |
| ReLu		|       									|
| Max Pool | 2x2 stride, same padding, outputs 16x16x64 |
| Fully connected | 512 units |
| ReLu | |
| Dropout | Keep probability = 0.2 | 
| Fully connected | 128 units | 
| ReLu| | 
| Softmax				| 								|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the twelth cell of the ipython notebook, under "Train, Validate and Test the model".

To train the model, I used an AdamOptimizer with a batch of 128, learning rate 0.0005, and L2 regularization with coefficient 0.01 for all FC weights. I train for 10 epochs.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the thirteenth and fourteenth cells of the Ipython notebook, under "Train, Validate and Test the model.".

I used the validation set accuracy to guide the training process. 

My final model results were:
* training set accuracy of 95.8%
* validation set accuracy of 97.9%
* test set accuracy of 95.8%

Considering that human level performance lies between 98.4-98.8% on the test set, we still have some way to go. 

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
##### I started with LeNet-5 because it was readily available from the course and because it seemed like a suitable model for simple classification in a low-dimensional image with centered images.
* What were some problems with the initial architecture?
##### The model capacity was not nearly enough, the ~82k parameters of LeNet-5 allowed for only about 87% accuracy on validation set. We needed to make the model more complex to reduce underfitting.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
##### As explained above, I begin with a 1x1x1 convolutional layer. There are a few other peculiarities of the network compared to LeNet-5. Note that the training images are already centered and of a similar shape - this means that signs take up approximately the same pixel space in all the images. For this reason, I have removed most of the wasteful max pooling layers - if the signs are in the same position, the weak spatial invariance property of max pooling is not helpful.
##### 3x3 kernels are used for two reasons: it has proven to be very succesfull ever since VGG Net and the images we are dealing with seem too small for bigger kernels. Only zero padding is used to not lose any information. Some trial and error was used to find the number of convolutional channels. Surprisingly, what worked well in practice was using 64 for all layers, rather than the progressive strategy found in literature. 
##### I use two fully connected layers of size 512 and 128 (also partly due to trial and error). Because I do not using max pooling layers, the model has very high capacity at 8.5 million parameters. We need to regularize it heavily. I use a L2 regularization on all the fully connected weights (including fc2 to output). In addition, I assign heavy dropout to the first FC layer (keep_prob=0.2 seems to work well in practice). Assigning dropout to FC2 did not seem to benefit the accuracy. A slightly smaller learning rate than the default one is used, as it seemed that the model validation accuracy was very sensitive to iterations. Other hyperparameters have default values.
##### Additional tricks that were tried but did not seem to help were Xavier initialization, multi-scale features as in LeCun paper, larger batch size, different number of fake images generated, different kernel sizes, Elu rather than ReLu for activation. It is very likely that some of these would help with enough patience to fine-tune the model.

* Which parameters were tuned? How were they adjusted and why?
##### A slightly smaller learning rate than the default one is used, as it seemed that the model validation accuracy was very sensitive to iterations.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
##### This is mostly explained in the first point. The only domain-specific measure I took, compared to benchmark CNN literature, was dropping some max pooling layers. If we were dealing with more troublesome images where signs are not centered, max pooling layers would become useful again.
##### It is perhaps worth noting how much every incremental change to architecture added to validation performance. Firstly, changing the model capacity (number of parameters) helped by far the most, boosting accuracy about 8% from LeNet-5 benchmark. Choosing the right layer sizes (by running many experiments) helped about 1-2%. Augmenting data added another 1%. Dropout, perhaps the most effective regularization technique, added 0.5%. Adding the 1x1x1 convolution to the beginning of the pipeline added a further 0.5%. Just in case, I added additional L2 regularization which seems to add about 0.2-0.3%. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

As we see, they are all with varying sizes, not always centered and with bounding noise - very different from our training and test data set. 

I use a naive approach for classification: resizing all images to 32x32, regardless of dimension.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the sixteenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop Sign									| 
| Children crossing    			| Road work									|
| Right-of-way at the next intersection					| Road work								|
| Wild animals	      		| 	Bumpy road				 				|
| Road work			| 	Road work  							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares poorly to the accuracy on the test set of 95.8% The model seems particularly fond of predicting the "road work" sign, perhps because of unbalanced training classes.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for doing this via visualizations is located in the 20th cell of the Ipython notebook.

![alt text][predictions]

We see that even on pictures where the classifier fails, it usually picks the correct option in the top 5 (children crossing, right-of-way, wild animals). What is worrisome is that it does not usually give much probability to the correct classes, even when the model is "unsure".

In general, this kind of performance would not be tolerable on the road: the correct sign has to be the top 1 choice with at least 99% probability. However, our CNN is likely a pretty decent classifier on cropped and centered images. We simply would need to add some localization method as the first step of the pipeline for these out of sample images and performance would surely increase. Ideally, this would be the same method that was used on our training data.
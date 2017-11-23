# **Behavioral Cloning** 

## The Writeup Template was used to create a report for the self driving car simulation project.

### Project submitted by Mona Moren


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior.  
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/figure1.png "original central image"
[image2]: ./examples/figure1-1.png "original left image"
[image3]: ./examples/figure1-2.png "original right image"
[image4]: ./examples/figure1-3.png "flipped central image"
[image5]: ./examples/figure1-4.png "flipped left image"
[image6]: ./examples/figure1-5.png "flipped right image"

[image01]: ./examples/"figure_1_using_udacity_data (succesful model).png" "Model Loss for model successfully trained on Udacity data"
[image02]: ./examples/"figure_1b_using_adam.png" "example of Model Loss for model unsuccessfully trained on Kamil's data"
[image03]: ./examples/"figure_1b_using_adam_5epochs_original_DNN_no_cropping.png" "example of Model Loss for model unsuccessfully trained on Kamil's data"

[image04]: ./examples/"figure_1_using_udacity_data_scatter_plot_prediction (successful model).png" "predictions scatter plot of Model successfully trained on Udacity data"
[image05]: ./examples/"figure_4_model2a.png" "example 1 of predictions scatter plot of Model unsuccessfully trained on Udacity data (without needing to test the model on the simulator)"
[image06]: ./examples/"figure_1_using_adam.png" "example 2 of predictions scatter plot of Model unsuccessfully trained on Udacity data (with needing to test the model on the simulator)"



## Rubric Points

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

My project includes the following files:
* model.py containing the script to create and train the model
* model_v2.py is the second submission to my project, after integrating some of the comments from the reviewer.
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video_training_on_Kamil_dataset.mp4 and video_training_on_Udacity_dataset, videos demonstrating the driving skills achieved by my network. This also demonstrates the importance of training data set example quality and quantity

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My final model consists of a convolution neural network with the following convolution layer types: 3 layers using a 5x5 filter size and one layer of 3x3 filter sizes and depths between 20 and 128 (model.py lines 163-166) 
  
The convolution layers include RELU activation to introduce nonlinearity (code lines 163-166), and the data is normalized directly at the second layer in the model by using a Keras lambda layer (code line 161). 

The final architecture is the result achieved after exploring several different network architecture possibilities.  It required a lot of trial and error, both validating the models statistically, and testing the trained model on the track.

Here are examples of the network models I have tried to train on my computer, and never managed to get a good result on the simulator in automatic mode:

#model adapted from alemenis on github, due to issues at the time to actually run Keras without bug reports on my computer, who had successfully trained his own network on AWS servers):

	model = Sequential()
	model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(nb_rows, nb_cols, 3), name='Normalization'))
	model.add(Conv2D(36, 5, 5, border_mode='same', activation='relu'))
	model.add(MaxPooling2D((4, 4), (4, 4), 'same'))
	model.add(Conv2D(6, 5, 5, border_mode='same', activation='relu'))
	model.add(MaxPooling2D((4, 4), (4, 4), 'same'))
	model.add(Conv2D(6, 5, 5, border_mode='same', activation='relu'))
	model.add(MaxPooling2D((4, 4), (4, 4), 'same'))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))

# Second example using Kamil's architecture directly in my model.py script, when investigating the sources of why my own model.py code would not yield a trained model that could drive around the track.

	nb_rows  = 160 # image size, see center_image for the number of rows to use
	nb_cols  = 320
	#steering_theta = 0.3 # side cameras correction angle
	#test_samples = 20   # use few images to test the CNN
	epochs = 10
	batch_size = 512
	# Build the model # testing Kamil's network on my data processing
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))) #Normalize the images
	model.add(Cropping2D(cropping=((70,25),(0,0)))) #Crop the images (top and bottom parts)
	model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu")) #Add Conv net with a stride and relu activation
	model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu")) #Add Conv net with a stride and relu activation
	model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu")) #Add Conv net with a stride and relu activation
	model.add(Convolution2D(64,3,3,activation="relu")) #Add Conv net and relu activation
	model.add(Convolution2D(64,3,3,activation="relu")) #Add Conv net and relu activation     
	model.add(Flatten())
	model.add(Dense(100)) #Add a fully connected layer
	model.add(Dense(50)) #Add a fully connected layer
	model.add(Dense(10)) #Add a fully connected layer
	model.add(Dense(1))  
	model.summary() # print model summary
	
# 3rd example of network structure tested (the network just before the one I used that worked). One can see the different layers I had progressively tried to introduce and experiment with using dropout and pooling.  It was also in this architecture that I had experimented with additinal layers in varying sizes of 2D convolution:

	nb_rows  = 160 # image size, see center_image for the number of rows to use
	nb_cols  = 320
	#steering_theta = 0.3 # side cameras correction angle
	#test_samples = 20   # use few images to test the CNN
	epochs = 5 #10
	batch_size = 512
	# Build the model
	model = Sequential()
	model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(nb_rows, nb_cols, 3), name='Normalization')) 
	#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))) #Normalize the images
	model.add(Cropping2D(cropping=((90,25),(0,0)))) #Crop the images (top and bottom parts)

	model.add(Conv2D(16, 5, 5, subsample=(2,2), border_mode='same', activation='relu'))
	model.add(Conv2D(32, 5, 5, subsample=(2,2), border_mode='same', activation='relu'))
	#model.add(Dropout(0.4)) #added this dropout layer after being fed up with models that cannot generalize...
	model.add(Conv2D(32, 5, 5, subsample=(2,2), border_mode='same', activation='relu'))
	model.add(Conv2D(64, 5, 5, subsample=(2,2), border_mode='same', activation='relu'))
	#model.add(Dropout(0.4))
	#model.add(MaxPooling2D((4, 4), (4, 4), 'same'))
	#model.add(Dropout(0.4)) #added this dropout layer after being fed up with models that cannot generalize...
	model.add(Conv2D(64, 3, 3, subsample=(2,2), border_mode='same', activation='relu'))
	#model.add(Conv2D(64, 5, 5, border_mode='same', activation='relu'))
	model.add(Conv2D(64, 3, 3, subsample=(2,2), border_mode='same', activation='relu'))
	#model.add(Dropout(0.4)) #added this dropout layer after being fed up with models that cannot generalize...
	#model.add(MaxPooling2D((4, 4), (4, 4), 'same'))
	#model.add(Conv2D(6, 5, 5, border_mode='same', activation='relu'))
	#model.add(MaxPooling2D((4, 4), (4, 4), 'same'))

	model.add(Flatten())
	model.add(Dense(240))
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))  #  --> this guy here is supposed to give out a steering indication...

	model.summary() # print model summary

#### 2. Attempts to reduce overfitting in the model

Attempts to reduce the overfitting in the model came from tuning the following parameters:
	-the number of epochs used (having experimented between anywhere from 5 to 30 epochs)
	-the image cropping.
	-increase and then limit the number of dataset samples used (Udacity, Kamil's or my own recorded sets - this became obvious from obtaining the same driving instability on both of my laptops used for this project)

The original attempt to overfitting the model was to crop the input images used (code line 160).  These parameters were tuned progressively:
	cropping=((80,0),(0,0))
	cropping=((80,40),(0,0))
	cropping=((80,30),(0,0))
	All of these tuning parameters did not give satisfaction, since my model would not manage to converge when testing it directly on the simulator. 
	The image_plot function has commented code lines (line 22), the previous code from data_loader has been removed. These have helped me to visualize the data treatment prior to training my model.
	It looked like my model was seeing too much road, or also when I was getting familiar with the cropping tool that it was not looking to the right section of the image. 
	
The model was trained and validated on different data sets to ensure that the model was not overfitting, this is why I have used two separate databases for the training (Kamil's database) and the testing (Udacity's) of the model during the training phase. 
The model was tested by running it through the simulator in the "automatic mode" and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

My models would converge and train, yielding very promising statistics in both training and validation phases (25% of the data was provided to validation), yet never managed to reliably drive the car beyond the first bend after crossing the bridge.  
Here are examples of the statistics my models would train to, comparing an example of a successful training to examples of unsuccessful trainings:
![alt text][image01]
![alt text][image02]
![alt text][image03]

This was independant of:
	-the number of convolution layers included (having tried networks with as much as up to 6 convolution layers, see example networks above)
	-the number of dropout layers peppered through the layers.
	-the actual combination of convolution layers, with or without additional layers such as dropout and predicts
	-the type of estimator used will affect the quality of the output.
	
Following are a few additional notes on certain parameter tuninng insights gained:
	
The model contained dropout layers and pooling layers, however these were all later removed, since none of the attempts yielded smooth and acceptable steering beyond the bridge when testing directly the model on the track. 
	Dropout did not yield the results I was hoping for, since it appeared that adding a dropout layer required increasing the number of epochs for the network to learn. The model learning would also plateau relatively quickly, without it reflecting in the statistics.  This would become apparent in the testing phase around the track, where the driving was never much better or different, and the car would fail to stay on track anywhere around the first two bends in the road.
	Pooling, I fear that the way I have used it here only increased the noise in the processing. Once this was also removed, I felt that my network had improved its driving skills. Maybe used in a different way it could better help finetune the network learning. 
	
The convolution layers themselves were also tricky.  This led to the following insight from my experience:
	It was better to use a pyramidal filter structure, starting with large filters that are relatively 'shallow' (lines 163-165) , and finishing with a small deep filter (line 166). 
	Stacking up convolution layers of the same size or "inverting" the filter pyramide simply yielded models that at best could not converge (i.e. train) or at worst that would diverge (statistically sound, but predictions in the scatter plots would be completely off the mark).
	My final model initially had 5 convolution layers, with the last two layers being 64x3x3.  Replacing these last two with just one convolution layer of the same "size" 128x3x3 ultimately gave the good convergence to my model.

Using an appropriate estimator also makes a difference in the network's capability to learn:
	The optimizers used and tested were: adam, adamax, SGD.
	Originally starting with SGD, it became apparent that the model learning would eventually plateau.  The test driving on the track would never be smooth.
	Moving onto an adam optimizer yielded better results.  The car would drive, and stay more reliably on track.  Yet it would have at times very shaky steering.
	At last, using the adamax training yielded the smoothest steering results.

In the end, the best method I had found for reducing the overfitting in the model were the following parameter tunings:
	-appropriately crop the input images: logically aim for a section of the road that is close to the car, and provides some additional image depth (to look towards the road ahead, and not stare at the clouds)
	-crop the images within the network rather than as a function outside the network (this was rather for the flexibility of testing out different cropping settings)
	-impose as a design item to keep the original number of epochs for training at 5 epochs. If the network needed more epochs, then the risk of overfitting the model increased.
	-rely on the statistics I visualize only as a rule of thumb to "see" how the model could be doing ( depending on the shape of my scatter plot it was immediately obvious if it was not worth my time to test on the simulation track)
	-get rid of the dropout layers and pooling layers, as they seemed to add noise to my network, and limit its learning fairly quickly, regardless of the optimizer used.
	-use progressively narrower and deeper convolution filters 
	-privelege convolution filter size and depth to number of convolution filter layers. 
	-use an appropriate estimator. Adamax performed better than Adam, which performed better than classical SGD. Each time the resulting steering from the trained model would become progressively better. 


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.  I have created my data using both the computer keypad and a mouse for steering.

I did not use the data I had created, since I was not confident that the computer I had used was capturing it appropriately. This was because throughout my initial series of network training phases my network had never achieved the capability to drive.
My initial troubleshooting thought is that the Udacity dataset should be sufficient to achieve smooth steering, if I was using an appropriate network.  I have also relied on the same thought process, when using Kamil's dataset for training my network.

Kamil's dataset was created by normally driving around the track twice (with the car as close to the middle as possible), followed by distinct driving segments demonstrating recovery from a car that had drifted to one side of the road or the other.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start off with a convolution network that would run on my computer. This did require re-installing my python environment a few times.  
On my old computer (CPU only), this required a bit of background search, since my python installation was not stable and created unexpected exceptions when running my code. 
On my new computer it required getting familiar with new technology (GPU processing) and drivers (CUDA).
Once all the python data packages and imports could be called successfully, I then started trying out different network architectures. 

My first step was to use a convolution neural network model similar to the nvidia architecture. I thought this model might be appropriate because it already has shown results and is commercially available.
I then simplified it to obtain an architecture that my old computer could compile and run (see computer related issues described above), before trying out additional tweaks to the network (dropout, pooling, sampling, parameter tuning...). 
My experimentation was limited on my old computer, and really took off using a GPU to run my architecture.  What previously required a 2 hour wait could now be achieved in 5 minutes.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.  Initially I split the original dataset into training (75%)/testing(25%), with a further split of the training set into training (75%)/validation (25%).  
It took a bit of experimentation, including trying out architectures I knew did work and converge, to realize that this split was too severe, and my network simply didn't have enough examples to train on.
The result was not obvious in the statistics, since both final training and validation losses were low (maximum 3% and usually even less then 1%), but testing on the track revealed unpredictable driving.

The first step to resolve this was to attempt tuning the training/validation split parameter from .25 to a lower value, including .1 or less.  This was not sufficient to to successfully and reliably train my network. 
It was after several attempts, including experimenting with a network that I knew did achieve results, that I realized that my network would also need to train on the data originally reserved for statistical testing.
That was when I decided to split training and testing data by selecting two completely different data collections, rather than splitting one original collection.  This had yielded results that improved the distance the car would run without crashing, yet the network was still not optimal.
The final balance in the training/validation data split I found was using a 80%/20% split.

But this was not the only issue my network was facing in terms of overfitting.  It required in parallel additional parameter tuning, as described in the previous section in order to achieve the desired result. 

The final step in each network training iteration was to run the simulator to see how well the car was driving around track one. 
There were a few spots where the vehicle fell off the track: anywhere between the begining of the test drive and the first sand pit.  Sometimes this was quite spectacular, since the driving was perfect around the first bend, and for an unknown reason the car would suddenly swerve back and forth as if an oil patch had been hit, and finally crash into a tree. 
In order to improve the driving behavior in these cases, I had initially started out with trying to collect more training data, and then later simply concentrated on the other parameters available for tuning (anywhere from data augmentation techniques such as image cropping and flipping, to network adjustments such as reworking the size and depth of the convolution filter layers)

At the end of the process, with the final model architecture, the vehicle is able to drive autonomously around the track without leaving the road, and with smooth steering.

#### 2. Final Model Architecture

The final model architecture (model.py lines 158-174) consisted of a convolution neural network with a total of four 2D-convolution layers, that begin as wide and shallow (5x5x(depths of 20, 32 and 64), and end narrow and deep (3x3x128).  
The resulting dataset is then flattened in the flatten layer, and progressively squeezed through several dense layers, into a final layer that yields the steering command.
	# Parameters

	nb_rows  = 160 # image size, see center_image for the number of rows to use
	nb_cols  = 320
	epochs = 5 #10
	batch_size = 512
	# Build the model
	model = Sequential()

	model.add(Cropping2D(input_shape=(nb_rows, nb_cols, 3),cropping=((70,25),(0,0)))) #Crop the images (top and bottom parts)
	model.add(Lambda(lambda x: x/255 - 0.5, name='Normalization')) 

	model.add(Conv2D(20, 5, 5, subsample=(2,2), border_mode='same', activation='relu'))
	model.add(Conv2D(32, 5, 5, subsample=(2,2), border_mode='same', activation='relu'))
	model.add(Conv2D(64, 5, 5, subsample=(2,2), border_mode='same', activation='relu'))
	model.add(Conv2D(128, 3, 3, subsample=(2,2), border_mode='same', activation='relu'))

	model.add(Flatten())
	model.add(Dense(240))
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))  #  --> this guy here is supposed to give out an appropriate steering indication...

	model.summary() # print model summary


#### 3. Creation of the Training Set & Training Process

I had used two different datasets to separately train my network and try the results.  Note each time I did train on one data set (Udacity's or Kamil's), I used the other to create the predictions scatterplot. 

To augment the data set, I used all camera images available: right, center and left. I augmented the steering angle for the images coming from either the left or the right camera by .20(lines 72-73)
Improvement brought in model_v2.py: setting the angle correction from 0.20 --> 0.25.  The car stays within the yellow lines on track, however the driving became more swervy around the first bend before the bridge (it did not "feel" safe or comfortable).
![alt text](image1)
![alt text](image2)
![alt text](image3)
 
I also flipped the images, in order to balance out the steering angles:  with the original dataset, the car driving in one same direction would "always" turn left. Flipping the image and multiplying the corresponding steering angle by -1 would allow to give the model an easy example of turning right, without collecting any additional data around the track.
![alt text](image4)
![alt text](image5)
![alt text](image6)

After the collection process, using the Udacity dataset, I had 3584 driving samples in the driving log (with an absolute steering angle value larger than 0.01).  The total number of samples used for the training is 10752 (=3584*3 (center, left and right images))
I then preprocessed the data by flipping the dataset, bringing the total number of training samples available to the network to 21504.

Using Kamil's database for training my model, I had 5426 driving samples in the driving log (with an absolute steering angle value larger than 0.01).  Following the same computation logic as previously, this provided a total of 32556 image samples for training to the network.

I finally randomly shuffled the data set and put 20% of the data into a validation set (see above for the insight on tuning the parameter for splitting the dataset for validation). 

Both datasets provided sufficient data to train the network to stay on track.  However, Kamil's database provided a richer ammount of information for training to my database, by approximately 11000 additional driving examples compared to using the Udacity database for training.
This manifested itself in the final result of testing the car on the track, where using more driving examples helped the model steer more smoothly, and more often towards the center of the track once the car has passed the bridge and the second bend in the track 
(see videos "training on Udacity data set" and "training on Kamil's dataset" for comparisson). The car also hardly swerved around the middle of the road, when training with Kamil's dataset - whereas with Udacity's dataset the car seemed to be searching more often where to place itself along the track.
In conclusion, even with a good network, when it comes to network training, this demonstrates that additional "experience" also counts in finetuning the model!

Unseen bug pointed out by the first review of my code:
 - BGR2RGB transformation did in my image plotter for visualization, but not on the images that were treated and used for training the network.  This was corrected to transform each image immediately after it was loaded, so that the network will learn on images that resemble what it would be fed with the command PIL() in drive.py.

future developments to add to this project:
1) implement additional augmentation techniques to the original data set.  Ideas seen in the example reports provided simulate:
 - different road shapes and angles (by deforming the image, such as stretching vertically or horizontally), 
 - different lighting conditions by randomly changing the luminosity of the images already available, in order to simulate different times of the day 
 - simulate shaddows, where the lighting of the road will be different, yet safe to drive through

2) create additional recovery examples which would simulate obstacle avoidance (swerving around an object, slowing down or stopping)

3) incorporate additional parameters to train in the model: throttle, break and speed 
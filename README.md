# Behavioral Cloning with Driving Simulator

<p align="center">
	<img src="/media/output.gif" alt="result"
	title="result"  />
</p>

---

The goal of this project are the following:
- collect training data by driving laps in Udacity Driving Simulator
- filter and sample data for model input
- augment data to help the model generalize
- train a CNN to predict steering angles from image data
- test model to drive around the lap


The project consists out of the following files:
- imgs directory containing sampled training data on smooth driving
- add_imgs directory containing additional raw training data on hard curves
- data.ipynb as the original notebook for data exploration, filtering and sampling
- model.py containing the model and ImageDataGenerator
- model.h5 saved model for project
- drive.py as the controller for Udacity Simulator
- video.py to create video from captured images during testing
- video.mp4 as project video


## 1. Collection of training data

As the goal of the project is predicting steering angles from images, there has to go some though in HOW to record the training data. The Udacity Simulator captures Image Data from 3 Cameras (center, left, right) as well as the steering angle (our label) and throttle. All recorded images were captured in the highest graphic quality available, since some other options do not features some challenges like shadows. From basic physics it is quite simple to assume that a faster speed also means a bigger steering angle. As there is no way to manually stay a given speed other than max speed, I only captured training data while driving at full speed to keep the angles somewhat proportional to the speed. With this premise I drove around the track 5-7 times in either direction and saved those images to my hard drive. Steering was done with mouse input, since the dragging yields in a much more 'natural' way of driving than hard keyboard input, which results in steering spikes followed by a lot of zeros. Half of the training data features smooth 'in the middle' driving, while some proportion focusses on correcting the angle by wiggeling between the lines. Additionally I captured some more data on just the curves in the lap, since the model was always a bit shy with some higher steering angles.

## 2. Filter and Sample Data


With my recordings I had a total of 21323 center images available. This count could later be increased by flipping the images and labels around and thereby doubling the total amount of training data. Therefore we could only check the absolute distribution. Additionally with an ImageDataGenerator by Keras, we can generalize the images with some augmentation during training.

<p align="center">
	<img src="/media/obs_with_label.png" alt="result"
	title="result"  />
</p>

After a short display of some images with the label on them, it can be clearly seen that left curves require a negative steering angle, wheras right curves require a positive angle. This information can be used to take left and right images into account when training the model, because the perspective in those images have a different center and are looking towards the outer edges. With a correction of ~0.25 the training data would triple and also feature another perspective.

<p align="center">
	<img src="/media/obs_dist.png" alt="result"
	title="result"  />
</p>

As can be seen the steering angle ranges from 0-1, where 1 represents an 25° angle. The data is heavily skewed towards 0, therefore some kind of sampling needs to be applied to accurately represent curves. Therefore I create a linspace between 0 and 1 with 200 bins and sort each observation within those buckets. Afterwards I take a random sample of max 150 observations per bin (if bin has >150, else just all) and store those in a list. This list is then used to filter the dataframe for corresponding images.

Afterwards I create the dataframe for the ImageDataGenerator. This features all the selected images within the bins created before.
- take left, center and right image
- add .25, 0, -.25 to create incentive to go back to the center of the road in the perspective
- crop the image between 50:130 in the horizontal axis
- save the images/dataframe to upload on server
- shuffle dataframe

After saving all the images within a seperate directory, I got around 14000 images to start training the model. 

<p align="center">
	<img src="/media/sample_input.png" alt="result"
	title="result"  />
</p>


## 3. Augment Data to help model generalize

By using Keras ImageDataGenerator I can easily augment data inplace and dont need to save them. Therefore a lot of epochs does not neccisarily mean overfitting. The options included in the generator:
- rotation 10°
- width shift 10% (I am actually uncertain if this is good since the shift may interfere with the correct steering angle.. if car is too far left but gets shifted back)
- shear 10%
- zoom 10%
- vertical flip (excluded because I would also have to flip the label)

## 4. CNN to predict steering angles

I implemented a CNN consisting of 8 layers inspired by NVIDIA's CNN structure. I choose to drop one of the convolutions due because this challenge is not as complex as NVIDIA's was. Therefore the model architecture looks as follows:

- Input Layer
- Normalization Layer
- Conv2D (Filter 16, Kernel 5, Strides 2)
- Conv2D (Filter 32, Kernel 5, Strides 2)
- Conv2D (Filter 64, Kernel 5, Strides 2)
- Conv2D (Filter 128, Kernel 3, Strides 2)
- Fully-Connected (contains Dropout)
- Fully-Connected (contains Dropout)
- Fully-Connected (contains Dropout)
- Output: Steering angle

An Adam Optimizer with a learning rate of 1e-4 with decay of 1e-6 has been chosen.

<p align="center">
	<img src="/media/model_architecture.PNG" alt="result"
	title="result"  />
</p>


## 5. Test CNN on sample track

With the implemented ImageDataGenerator I trained the model for 10 Epochs and monitored its performance on the race track. Following things have been noticed:
- Driving seemed to be really smooth, vehicle stayed in middle of lane in straight ways and light curves
- Hard curves were difficult to handle since the model detected the curve pretty late and the steering was too soft
- Speed in drive.py has a strong influence on the steering angle, the model performed overall better at slow speed and started to wiggle between the boundaries

Since the most obvious reason for the upcoming problems were missing sample data for the curves, I added in some additions training images from just driving in the curves unfiltered to the training data (2700) and trained on them for 5 more Epochs. Afterwards the model was able to drive the track up to a speed of 25 mph. 

<p align="center">
	<img src="/media/output.gif" alt="result"
	title="result"  />
</p>

Some obvious improvements to the model could be:
- flip all images to double training data (flip label accordingly)
- collect more training data from curves, e.g. the challenge track where curves are not always visible due to going up or down
- adjust speed accoding to confidence, e.g. if not visible or really high steering prediction to help model get around corners

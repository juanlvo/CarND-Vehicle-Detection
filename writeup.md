##Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: /writeup_images/1.png
[image2]: /writeup_images/2.png
[image3]: /writeup_images/3.png
[image4]: /writeup_images/4.png
[image5]: /writeup_images/5.png
[image7]: /writeup_images/7.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

<b>Writeup / README</b>
<table>
	<tr>
		<th>Criteria</th>
		<th>Meets Specifications</th>
	</tr>
	<tr>
		<td>Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.</td>
		<td>The writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled.</td>
	</tr>
</table>

<b>Histogram of Oriented Gradients (HOG)</b>
<table>
	<tr>
		<th>Criteria</th>
		<th>Meets Specifications</th>
	</tr>
	<tr>
		<td>Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.</td>
		<td>The HOG feature is used in the extraction of the features, in vehicle_detection_videos.ipynb you can find the HOG parameter in the 5th box between lines 8 to 22, then after you can see the call of the function extract_features in lines 24 and 30, this function is located in lesson_function.py inside of the function you can find all the process related with HOG features, the parameters where choice with the recomendation of the lessons of Udacity so:
			- color_scape chose is YCrCb, the orientation.
			- HOG orient = 9  
			- HOG pix_per_cell = 8
			- HOG cell_per_block = 2
			- hog_channel = 'ALL'
			- Number of histogram bins hist_bins = 64
			- spatial_feat = True
			- Histogram features hist_feat = True
			- HOG features hog_feat = True
		</td>
	</tr>
	<tr>
		<td>Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).</td>
		<td>The HOG features extracted from the training data have been used to train a classifier, the clasifier chose was SVM. Features were scaled to zero mean and unit variance before training the classifier.</td>
	</tr>
</table>

<b>Sliding Window Search</b>
<table>
	<tr>
		<th>Criteria</th>
		<th>Meets Specifications</th>
	</tr>
	<tr>
		<td>Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?</td>
		<td>In the file vehicle_detection_videos.ipynb 3rd box is the function find_cars_svm where you can see the call of the function slide_window this function is declared in the file lesson_functions.py betweens lines 153 and 191 this implementation was the recomended by udacity. The scale is not use in this implementation. The overlap is 50%</td>
	</tr>
	<tr>
		<td>Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier?</td>
		<td>In this case is clear the detection is not good enough, for improve is important to implement HOG Sub-sampling windows but for the author it was pretty difficult to implement this functionality due an error of the line test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)) which never works and he should implement test_features = X_scaler.transform(np.hstack((hog_features)).reshape(1, -1)) but the result was pretty low, so at the end was better just to implement SVM due the waste of time. The images can be find below this rubrics</td>
	</tr>
</table>

<b>Video Implementation</b>
<table>
	<tr>
		<th>Criteria</th>
		<th>Meets Specifications</th>	
	</tr>
	<tr>
		<td>Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)</td>
		<td>The sliding-window search plus classifier (only SVM) has been used to search for and identify vehicles in the videos provided. Video output has been generated with detected vehicle positions drawn bounding boxes. the video is <a href="video_p5.mp4">here</a></td>
	</tr>
	<tr>
		<td>Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.</td>
		<td>There is the implementation of heatmap you can find the function mix_heat_map_image in the 4th box, this function is recibing the boxes and then is representing one big box with a filter of number of boxes in the line 19 of the box 4th</td>
	</tr>
</table>

<b>Discussion</b>
<table>
	<tr>
		<th>Criteria</th>
		<th>Meets Specifications</th>	
	</tr>
	<tr>
		<td>Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?</td>
		<td>The most difficult part was dettecting the problem in the function find_cars where in the execution of the line 403 of the lesson_function.py all the time was getting the error "ValueError: operands could not be broadcast together with shapes (1,5292) (8460,) (1,5292)" and after a lot of discussions in the slack chat he discover the line can work if he change for test_features = X_scaler.transform(np.hstack((hog_features)).reshape(1, -1)) this was quite complex to understan because the explanation was not enough to understand the function find_cars, even he read all the material several times he could not find where it came the problem, and this is the key for having a better detector of vehicles</td>
	</tr>
</table>


---
###Writeup / README

####1. Provide a Writeup 


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.
  
The HOG feature is used in the extraction of the features, in vehicle_detection_videos.ipynb you can find the HOG parameter in the 5th box between lines 8 to 22, then after you can see the call of the function extract_features in lines 24 and 30, this function is located in lesson_function.py inside of the function you can find all the process related with HOG features, the parameters where choice with the recomendation of the lessons of Udacity so:
			
			- color_scape chose is YCrCb, the orientation.
			- HOG orient = 9  
			- HOG pix_per_cell = 8
			- HOG cell_per_block = 2
			- hog_channel = 'ALL'
			- Number of histogram bins hist_bins = 64
			- spatial_feat = True
			- Histogram features hist_feat = True
			- HOG features hog_feat = True

I started by reading in all the `vehicle` and `non-vehicle` images in the box 5th line 1 with the call of the function test_data_loading 

![alt text][image1]


####2. Explain how you settled on your final choice of HOG parameters.

the parameters where choice with the recomendation of the lessons of Udacity, the author teied with different combinations but at the end the bestone were the parameters show in the lessons of udacity

Here an example of HOG features

![alt text][image3]

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The training you can find it in the box 5th lines 39 and 55 the clasifier is SVM with HOG feature and the color space used it is RGB

Here is an example of the features and the normalization

![alt text][image2]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Was decided to check every window position all over the image and trying to detecting a car

![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

 1. Train of the SVM
 ![alt text][image1]
 2. Take an Image and check every window of the image
 ![alt text][image4]
 3. Try to identify posible windows like in this image
 ![alt text][image5]
 4. Check the heatmap of the windows in the image and draw the final boxes
 ![alt text][image5]
 
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a <a href="video_p5.mp4">link to my video result</a>


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

There is the implementation of heatmap you can find the function mix_heat_map_image in the 4th box, this function is recibing the boxes and then is representing one big box with a filter of number of boxes in the line 19 of the box 4th



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most difficult part was dettecting the problem in the function find_cars where in the execution of the line 403 of the lesson_function.py all the time was getting the error "ValueError: operands could not be broadcast together with shapes (1,5292) (8460,) (1,5292)" and after a lot of discussions in the slack chat he discover the line can work if he change for test_features = X_scaler.transform(np.hstack((hog_features)).reshape(1, -1)) this was quite complex to understan because the explanation was not enough to understand the function find_cars, even he read all the material several times he could not find where it came the problem, and this is the key for having a better detector of vehicles 


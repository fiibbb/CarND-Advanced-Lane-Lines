##Writeup

###Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1_undistorted.jpg "Undistorted"
[image2]: ./output_images/test4_undistorted.jpg "Road Transformed"
[image3]: ./output_images/test4_thresholded.jpg "Binary Example"
[image4]: ./output_images/test6_perspective.jpg "Warp Example"
[image5]: ./output_images/test1_fit.jpg "Fit Visual"
[image6]: ./output_images/final_image_example.jpg "Output"

###Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook "./Advanced-Lane-Lines.ipynb", see `LaneLine.calibrate`.  

I loop through all images located in `./cameral_cal`, convert them to grayscale colorspace, and then pass into `cv2.findChessboardCorners`. After all images for which `cv2` is able to find the corners, I use `cv2.calibrateCamera` function to calcuate the distortion matrix and coefficient.

The calibration matrix and coefficient are stored in the `LaneLine` object for later use. An example of the undistort effect shown as below:

![alt text][image1]

###Pipeline (single images)

In this section I will demonstrate the pipeline to handle a single image. The pipeline is implemented in `LaneLine.process_img` function, which calls out to a couple other transformation functions in the `LaneLine` object in series.

####Distortion correction

The first stage of the pipeline is to apply the camera calibration as mentioned above to undistort the image. An example of the undistort effect shown as following:  

![alt text][image2]

####Thresholding

The second stage of the pipeline is to use various thresholding algorithms to transform the undistorted image into a binary image where the lane lines are clearly visible. In my case, I find the combination of absolute sobel gradient thresholding and HLS thresholding on the S channel to be particularly useful. As shown in `LaneLine.thresh_img`, I'm using `(20, 100)` for absolute sobel gradient threshold and `(90, 255)` for HLS thresholding on the S channel. An example of the thresholded image shown as below:

![alt text][image3]

####Perspectie transformation

The third stage of the pipeline is to use perspective transformation to extrat the sub-portion of the image where lane lines are included, and project them into a flat plane for easier lane fitting. This is implemented in `LaneLine.perspective` function. The perspective transformation matrix is calcuated in the constrctor of the `LaneLine` object. In this case, I'm using coordinates `[190,720], [1090,720], [600,440], [680,440]` as `source_bottom_left`, `source_bottom_right`, `source_top_left` and `source_top_right`. These coordinates are then mapped the top and bottom of the transformed image, with `dst_offset` from left and right of the transformed image. An example of the transformation shown as below:

![alt text][image4]

####Lane fitting

The fourth stage of the pipeline is to use the sliding window technique to map out the most possible lane line pixiels in the binary image after perspective transformation, and use a 2nd degree polynomial fitting to identify the lane lines. Note that histogram and sliding window is only used for the initial image when processing video. Once the first frame is processed, the subsequent frames are calcualted based on the lane line ploynomials found in the previous frame. Finally, the polygon highlighting the part identified as lane is drawn in green. An example of the processed image shown as below:

![alt text][image5]

####Curvature and Position

The curvature and the position of the vehicle with respect to the lane center are calculated in the `LaneLine.find_fits` function towards the end. Note that the former is based on coordinates before applying reverse perspective transform, and the latter is based on coordinates after applying reverse perspective transform.  

####Final result

After a first frame of the video is passed to the `LaneLine.process_img` function, the result is shown as the following:  

![alt text][image6]

---

###Pipeline (video)

Here's a [link to my video result](./output.mp4)

---

###Discussion

The overall result of the lane finding pipeline worked well, it was able to identify the lane correctly most of the time. Few small glitches in the video shows that the right side lane line jumps around a little bit from time to time. This is due to the fact that every frame of the video is processed separately. Ideally, I should take the average of the last `k` frames (where `k` might be roughly somewhere between 3-10), and calculate the fitting polynomials based on that. I believe that would adding a smoothing effect to the pipeline. Another aspect of it may worth tuning is the thresholding algorithm. It seems it is difficult to find a correct thresholding algorithm that works for all scenarios. When I tried applying the pipeline to some other videos where the lighting effect is drastically different, the effect is less than ideal. This may be due to the fact that I only use two thresholding technique. Adding more thresholding dimension to the pipeline may be able to help.

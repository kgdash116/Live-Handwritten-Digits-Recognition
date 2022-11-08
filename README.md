# Live Handwritten Digits detection

## Contents:
- The main py file is used for doing detections.
- Model.h5: is the weight file from the trained CNN model, that is trained on
  the MNIST dataset.
------------------------
## Method:
Two way detections are possible
a.  Through webcam feed
b.  Recognition on images<br><br>

In the start the program prompts the user to enter 1 for detection on images,
after which the user is prompted to add a file name. The second approach, that
is  the live recognition through webcam can be selected by pressing 2 when
prompted.<br><br>

The basic methodology is the same for both methods.
The input image is converted to a grayscale image, then the threshold is
applied, after this the contours are found using the threshold.
Through the contours we get the dimensions: x-point, y-point, height and width.
With the help of these points we make bounding boxes on the actual image.
The digit is extracted out from the image and then resized. It is then appended
to an array and for each digit a predication is made through the model,
utilizing the models weights. These predictions are then casted on the
image/feed.

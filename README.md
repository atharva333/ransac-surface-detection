# Road Surface Region Detection in 3D Stereo

Developed using Python and OpenCV

## Description

Computer vision assignment with university provided dataset of stereo images taken from a car. The task required students to use Python and OpenCV and find the best plane that fit the road in the given pair of images. Code is split into pre-processing, stereo matching, RANSAC and obstacle detection. 

## Pre-processing

* The colour images were processed to remove highly saturated green pixel points, this was done using HSV filtering and the colours were restricted to a certain range. Removing green from the points stops grass patches on the road plane from being falsely identified as part of the road.

![Road-without-green](https://github.com/atharva333/ransac-surface-detection/blob/master/images/green.png) 
Example of green parts being excluded from road plane.

* A mask was applied to remove the bonnet and top left and top right corners of the image. Removing these sections meant only the relevant pixels were considered for planar detection resulting in better performance and more accurate road detection.

![Mask](https://github.com/atharva333/ransac-surface-detection/blob/master/images/mask.png) 
Image of mask used to remove irrelevant parts from image

## Stereo matching

* Semi-global block matching was used to calculate the disparity map of the scene given left and right images.
![Stereo](https://github.com/atharva333/ransac-surface-detection/blob/master/images/stereo.png) 
Disparity map obtained from SGBM, lighter colours are closer to camera and darker are further away

* Many parameters were fit such as disparity noise filtering, maximum disparity value.
* Hole filling was used to improve patches of black pixels and gaps in the disparity maps obtained form the stereo matching. 

## RANSAC

* Over a range of trials 3 points were randomly sampled and a plane was fit using these points to define the coefficients of the plane. The plane that had the most points (from the stereo data)within a certain distance of the plane was selected as the best plane.
* The distance and number of trials were key parameters for accuracy of the resulting road plane.
* Only planes that had normals pointing upwards were considered because a road plane will have a relatively flat plane and therefore a normal that points upwards. This stopped walls and other large surfaces from being taken as roads.

## Obstacle detection

* The plane was drawn using a convex hull of all the points on the plane. Obstacles were shown using the points in the convex hull that had a disparity that was significantly different from the rest of the points in the convex hull. 

![Obstacle](https://github.com/atharva333/ransac-surface-detection/blob/master/images/obstacle.png)
Example of obstacle detection with pedestrian and traffic cone 





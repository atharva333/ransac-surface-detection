##############

# INSTRUCTIONS
# master_path_to_dataset is the variable that points to the filepath with the images dataset
# select filepath and assign to this variable
# running the script will then run the algorithm and show all the intermediate processing images as well
# the important windows are labeled finalObs (final road detection) and holeHist (hole filled and histogram equalised disparity map)

# n.b. commented code snippets show different experiments on image pre and post processing
##############

import cv2
import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt

master_path_to_dataset = "../TTBB-durham-02-10-17-sub10" # ** need to edit this **
directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = "" # set to timestamp to skip forward to

crop_disparity = False # display full or cropped disparity image
pause_playback = False # pause until key press after each image

#####################################################################

# fixed camera parameters for this stereo setup (from calibration)

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 184.0
image_centre_w = 474.5

#####################################################################

## project_disparity_to_3d : project a given disparity image
## (uncropped, unscaled) to a set of 3D points with optional colour

def project_disparity_to_3d(disparity, max_disparity, rgb=[]):
    ''' Projects points from 2d to 3d using disparity to calculate Z coordinates'''

    points = []

    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    height, width = disparity.shape[:2]
    
    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then get reasonable scaling in X and Y output

    #Zmax = ((f * B) / 2)

    for y in range(height): # 0 - height is the y axis index
        for x in range(width): # 0 - width is the x axis index

            # if we have a valid non-zero disparity

            if (disparity[y,x] > 0):

                # calculate corresponding 3D point [X, Y, Z]

                # stereo lecture - slide 22 + 25

                Z = (f * B) / disparity[y,x]

                X = ((x - image_centre_w) * Z) / f
                Y = ((y - image_centre_h) * Z) / f

                # add to points
                

                if(rgb.size > 0):
                    points.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]])
                else:
                    points.append([X,Y,Z])
    
    return points

#####################################################################

# project a set of 3D points back the 2D image domain

def project_3D_points_to_2D_image_points(points):

    points2 = []

    # calc. Zmax as per above

    #Zmax = (camera_focal_length_px * stereo_camera_baseline_m) / 2

    for i1 in range(len(points)):

        # reverse earlier projection for X and Y to get x and y again

        x = ((points[i1][0] * camera_focal_length_px) / points[i1][2]) + image_centre_w
        y = ((points[i1][1] * camera_focal_length_px) / points[i1][2]) + image_centre_h
        points2.append([x,y])

    return points2

#####################################################################

def ransac(points):

    ''' RANSAC function takes points and uses them for planar fitting
        Returns equation of plane and points that are on plane for visualisation'''

    trials = 200 # number of ransac trials

    maxMatches = 0 # intialise max matches to 0

    pointsOnPlane = []

    bestIndices = [] # initialise best indices list

    PP1, PP2, PP3 = 0 ,0, 0 # PP are plane points i.e the points we want

    points = np.array(points) # convert points to numpy array for easier slicing
    points = points[:,:3]

    # Run for number of trials

    for i in range(trials):


        # Checking for co linearity and the same point

        cross_product_check = np.array([0,0,0])
        while cross_product_check[0] == 0 and cross_product_check[1] == 0 and cross_product_check[2] == 0:
            # Creates list of 3 random integers in the given range
            randomList = random.sample(range(len(points)), 3)
            P1 = points[randomList[0]][:3]
            P2 = points[randomList[1]][:3]
            P3 = points[randomList[2]][:3]
            # make sure they are non-colinear
            cross_product_check = np.cross([i - j for i, j in zip(P1, P2)], [i - j for i, j in zip(P2, P3)])

        # how to - calculate plane coefficients from these points

        coefficients_abc = np.dot(np.linalg.inv(np.array([P1,P2,P3])), np.ones([3,1]))
        coefficient_d = math.sqrt(coefficients_abc[0]*coefficients_abc[0]+coefficients_abc[1]*coefficients_abc[1]+coefficients_abc[2]*coefficients_abc[2])

        # how to - measure distance of all points from plane given the plane coefficients calculated

        if ((abs(coefficients_abc[1]) > 4*abs(coefficients_abc[0])) and (abs(coefficients_abc[1]) > 2*abs(coefficients_abc[2]))):

            dist = abs((np.dot(points, coefficients_abc) - 1)/coefficient_d)

            # Calculate number of points that are under a certain threshold distance away from the plane
            # Later we'll calculate how many points match for each random plane formed

            # Condition produces array where distance is less than given threshold
            indices = (dist < 0.025)
            
            if (indices.sum() > maxMatches):
                coef = coefficients_abc / coefficient_d
                bestIndices = indices
                maxMatches = indices.sum()
                PP1, PP2, PP3 = P1, P2, P3

    for idx, boolean in enumerate(bestIndices):
        if boolean == True:
            #print(points[idx])
            pointsOnPlane.append(points[idx])

    return [PP1, PP2, PP3], pointsOnPlane, coef

#####################################################################

def applyMask(img):

    ''' Apply a mask to the given image
        Mask file is hard coded in here '''

    # Read file for mask

    mask = cv2.imread("cpqr93disparity-mask.png", 1)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

    #cv2.imshow("Mask", mask)

    res = cv2.bitwise_and(img,img,mask = mask)

    #cv2.imshow("Mask applied", res)

    return res

#####################################################################

def holeFilling(img):

    ''' Apply hole filling to depth map 
        Credit to paper from dallas uni '''

    kernelSize = 5
    #kernel = np.zero(5,5)

    imgF = img.copy()

    for row in imgF:
        for idx, pixel in enumerate(row):
            #print(pixel, end = "")
            if (pixel < 5 or pixel > 180) and idx < 1023:
                row[idx] = int(row[idx-1] )
                #print("changed")
    return imgF

#####################################################################

def drawNormal(normVec, point1, disp):

    ''' Draw normal vector using arrow on given image'''

    if holeHist[point1[1]][point1[0]] == 0:
        point1.append(1)
    else:
        point1.append(holeHist[point1[1]][point1[0]])

    #print(point1)
    # Take x,y points of the top of bonnet

    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    point1Z = (f * B) / point1[2]
    point1X = ((point1[0] - image_centre_w)*point1Z)/f
    point1Y = ((point1[1] - image_centre_h)*point1Z)/f

    point2 = [point1X-(normVec[0]*0.1), point1Y-(normVec[1]*0.1), point1Z-(normVec[2]*0.1)]

    points = [[point1X, point1Y, point1Z],point2]

    

    points2d = project_3D_points_to_2D_image_points(points)

    return points2d

#####################################################################

def colourPre(img):

    ''' Go through all the points obtained and select the green ones
        Take the inverse of the image and return the mask '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([30,100,0])
    upper1 = np.array([80,255,255])

    #Threshold the HSV image to get only blue colors
    mask1 = cv2.inRange(hsv, lower1, upper1)

    mask = mask1 

    #cv2.imshow('hsvmask',cv2.bitwise_not(mask))
   
    #Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)

    #cv2.imshow('res',res)

    return cv2.bitwise_not(mask)

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right)

full_path_filename_left = os.path.join(full_path_directory_left, "1506942480.483420_L.png")
full_path_filename_right = (full_path_filename_left.replace("left", "right")).replace("_L", "_R")

left_file_list = sorted(os.listdir(full_path_directory_left))

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)

max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(  0, max_disparity, 21)

for filename_left in left_file_list:

    # skip forward to start a file we specify by timestamp (if this is set)

    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = ""

    # from the left image filename get the correspondoning right image

    filename_right = filename_left.replace("_L", "_R")
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

    # check the files actually exist

    if (os.path.isfile(full_path_filename_left) and os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        #print("-- files loaded successfully")

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        hsv = colourPre(imgL)

 
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

        ##### GAUSSIAN BLUR APPLIED #####

        gbL = cv2.GaussianBlur(grayL, (7,7), 0)
        gbR = cv2.GaussianBlur( grayR, (7,7), 0)

        ##### LAPLACIAN APPLIED #####

        # lapL = cv2.Laplacian(gbL,cv2.CV_64F,7)
        
        # # lapL = np.absolute(lapL)
        # # cv2.imshow("Laplacian2", lapL)
        # lapL = np.uint8(lapL)
        # cv2.imshow("LaplacianL", lapL)

        # lapR = cv2.Laplacian(gbR,cv2.CV_64F,7)
        # # lapR = np.absolute(lapR)
        # lapR = np.uint8(lapR)
        # cv2.imshow("LaplacianR", lapR)

        ##### CANNY APPLIED #####

        # cannyL = cv2.Canny(grayL,50,149)
        # cm = applyMask(cannyL)
        # cannyR = cv2.Canny(grayR, 50, 149)
        #cv2.imshow("CannyL", cm)
        # cv2.imshow("CannyR", cannyR)

        # sobelx64fL = cv2.Sobel(grayL,cv2.CV_64F,0,1,ksize=5)
        # abs_sobel64fL = np.absolute(sobelx64fL)
        # sobel_8uL = np.uint8(abs_sobel64fL)

        # sobelx64fR = cv2.Sobel(grayR,cv2.CV_64F,0,1,ksize=5)
        # abs_sobel64fR = np.absolute(sobelx64fR)
        # sobel_8uR = np.uint8(abs_sobel64fR)

        ##### MASK APPLIED #####
 
        maskedL = applyMask(gbL)
        maskedR = applyMask(gbR)

        disparity = stereoProcessor.compute(maskedL,maskedR)

        # filter out noise and speckles (adjust parameters as needed)

        dispNoiseFilter = 12 # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

        # scale the disparity to 8-bit for viewing
        # divide by 16 and convert to 8-bit image (then range of values should
        # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
        # so we fix this also using a initial threshold between 0 and max_disparity
        # as disparity=-1 means no disparity available

        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
        disparity_scaled = (disparity / 16.).astype(np.uint8)
        

        #disparity_scaled = equalise(disparity_scaled)

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)

        cv2.imshow("disparity", (disparity_scaled * (255. / max_disparity)).astype(np.uint8))

        kernel = np.ones((5,5),np.uint8)
        #erosion = cv2.erode((disparity_scaled * (255. / max_disparity)).astype(np.uint8),kernel,iterations = 1)
        
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

        # HOLE FILLING PERFORMED

        hole = holeFilling((disparity_scaled * (255. / max_disparity)).astype(np.uint8))
        holeMask = applyMask(hole)

        #cv2.imshow("holeMask", holeMask)

        # HISTOGRAM EQUALISED

        holeHist =  cv2.equalizeHist(holeMask)
        cv2.imshow("holeHist", holeHist)

        ##### MORPHOLOGY APPLIED #####

        #morph = cv2.morphologyEx(disparity_scaled , cv2.MORPH_CLOSE, kernel)

        #cv2.imshow("morph",morph)

        ##### DILATION AND EROSION #####
        # Dilate to fill the holes and erode to return close to original 

        # kernelDE = np.ones((5,5), np.uint8)
        
        # img_dilation = cv2.dilate(morph, kernelDE, iterations=7)
        # img_erosion = cv2.erode(img_dilation, kernelDE, iterations=5)

        ##### HISTOGRAM EQUALISATION #####

        # histEq = cv2.equalizeHist(img_erosion)
        # cv2.imshow("hist",histEq)

        # maskHist = applyMask(histEq)

        #####

        # project to a 3D colour point cloud (with or without colour)

        # points = project_disparity_to_3d(disparity_scaled, max_disparity)
        points = project_disparity_to_3d(holeHist, max_disparity, imgL)

        # Results from RANSAC returned into this variable
        ransacRes = ransac(points)
    

        pts = project_3D_points_to_2D_image_points(ransacRes[1])
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1,1,2))

        # Create blank canvas for each point plot
        blank = np.zeros((544,1024,3), np.uint8)
        blankContour = np.zeros((544,1024,3), np.uint8)
        blankConvexHull = np.zeros((544,1024,3), np.uint8)


        # For every point on the plane draw a circle

        for point in pts:
            #print(point[0][0], point[0][1])
            cv2.circle(blank,(point[0][0],point[0][1]),2,(255,255,0), -1)

        hsvPoints = cv2.bitwise_and(blank, blank, mask=hsv)

        #cv2.imshow("hsvPoints", hsvPoints)

        # Find contours of the points in the plane
        _,contours,_ = cv2.findContours(cv2.cvtColor(hsvPoints,cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find largest contour by taking max area
        c = max(contours, key = cv2.contourArea)
        # Draw contours on to the left image
        
        cv2.drawContours(blankContour,[c], -1, (13,118,223), -1 )


        # Create convex hull of the largest contour
        hull = cv2.convexHull(c)
        cv2.drawContours(imgL, [hull], -1, (0,0,255), 2)
        cv2.drawContours(blankConvexHull, [hull], -1, (13,118,223), -1)

        # Take the subtraction of the filled in convex hull and the largest contour
        blankConvexHull = blankConvexHull - blankContour

        cv2.imshow("points", blank)
        
        # To show the obstacle detection using convex hull and contour take a weighted sum
        # This will allow transparent showing of obstacles
        finalObs = cv2.addWeighted(imgL, 1, blankConvexHull, 0.5, 0)

        # Get the points for the normal arrow, pass it a constant point on the image
        normPoints = drawNormal([ransacRes[2][0][0], ransacRes[2][1][0], ransacRes[2][2][0]], [516,366], holeHist)

        # Draw arrow
        cv2.arrowedLine(finalObs, (int(normPoints[0][0]), int(normPoints[0][1])), (int(normPoints[1][0]), int(normPoints[1][1])), (0, 255, 0), 2, 8, 0);

        cv2.imshow('finalObs',finalObs)
        cv2.imshow('right image',imgR)

        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparity_scaled)
            cv2.imwrite("left.png", imgL)
            cv2.imwrite("right.png", imgR)
        elif (key == ord('c')):     # crop
            crop_disparity = not(crop_disparity)
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback)
        else:
            #print("-- files skipped (perhaps one is missing or not PNG)")
            print()

        print(filename_left)
        print(filename_right + " : road surface normal (" + str(ransacRes[2][0][0]) + ", " + str(ransacRes[2][1][0]) + ", " + str(ransacRes[2][2][0]) + ")")


# close all windows

cv2.destroyAllWindows()

#####################################################################

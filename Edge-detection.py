"""
@author: tenkapo
"""

import cv2 as cv



#===================== load image and convert to grayscale  ==============================
img = cv.imread('/home/tenkapo/Downloads/logo_om.png')  #read the image

# Display original image
cv.imshow('Original', img)
cv.waitKey(0)
 
# Convert to graycsale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Blur the image for better edge detection
#img_blur = cv.GaussianBlur(img_gray, (3,3), 0) 
 
#===================== Sobel operator  ==============================
sobelx = cv.Sobel(src=img_gray, ddepth=cv.CV_64F, dx=1, dy=0, ksize=1) # Sobel Edge Detection on the X axis with kernel size 1x1
sobely = cv.Sobel(src=img_gray, ddepth=cv.CV_64F, dx=0, dy=1, ksize=1) # Sobel Edge Detection on the Y axis with kernel size 1x1
sobelxy = cv.Sobel(src=img_gray, ddepth=cv.CV_64F, dx=1, dy=1, ksize=1) # Combined X and Y Sobel Edge Detection


# Display Sobel Edge Detection Images
cv.imshow('Sobel X', sobelx)
cv.waitKey(0)
cv.imshow('Sobel Y', sobely)
cv.waitKey(0)
cv.imshow('Sobel X Y using Sobel() function', sobelxy)
cv.waitKey(0)
 
#===================== Laplacian operator  ==============================
lapl = cv.Laplacian(img_gray, ddepth=cv.CV_64F, ksize=1) # laplace edge detection with kernel size 1x1

# Display laplace Edge Detection Images
cv.imshow('laplace', lapl)
cv.waitKey(0)

#===================== canny operator  ==============================
edges = cv.Canny(image=img_gray, threshold1=100, threshold2=200) # Canny Edge Detection with 100 as min and 200 as max for hyteresis thresholding

# Display Canny Edge Detection Image
cv.imshow('Canny Edge Detection', edges)
cv.waitKey(0)
 
cv.destroyAllWindows()

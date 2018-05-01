import numpy as np
import cv2

def grayscale(img):
    """convert to gray image"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Blur"""
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img,low_threshold,high_threshold)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

image = cv2.imread('input.jpg')
gray_image = grayscale(image)
blur_image = gaussian_blur(gray_image, 5)
canny_image = canny(blur_image, 50, 150)

imshape = image.shape
lower_left = [1,imshape[0]]
lower_right = [imshape[1],imshape[0]]
top_left = [1,imshape[0]/2]
top_right = [imshape[1],imshape[0]/2]
vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
roi_image = region_of_interest(canny_image, vertices)

rho = 4
theta = np.pi/180
hough_threshold = 10
minLineLength = 10
maxLineGap = 200
lines = cv2.HoughLinesP(roi_image,rho,theta,hough_threshold,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('gray.jpg', gray_image)
cv2.imwrite('blur.jpg', blur_image)
cv2.imwrite('canny.jpg', canny_image)
cv2.imwrite('roi.jpg', roi_image)
cv2.imwrite('output.jpg', image)

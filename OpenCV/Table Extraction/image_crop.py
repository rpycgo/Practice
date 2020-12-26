# reference: https://medium.com/coinmonks/a-box-detection-algorithm-for-any-image-containing-boxes-756c15d7ed26

import cv2
import numpy as np


# read the image
img = cv2.imread('d:/extraction/table.jpeg', 0)
# threshold the image
(threshold, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
# invert the image
img_bin = 255 - img_bin
cv2.imwrite('d:/extraction/image_bin.jpg', img_bin)


# define a kernel length
kernel_length = np.array(img).shape[1] // 50
# vertical kernel of (1 x kernel_length), whichwill detect all the vertical lines from the image
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
# horizontal kernel of (kernel_length x 1), which will help to detect all the horizontal line from the image.
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
# kernel of (3 x 3) 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))


# morphologial operation to detect vertical lines from an image
img_temp1 = cv2.erode(img_bin, vertical_kernel, iterations = 3)
vertical_line_image = cv2.dilate(img_temp1, vertical_kernel, iterations = 3)
cv2.imwrite('d:/extraction/vertial_lines.jpg', vertical_line_image)

img_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations = 3)
horizontal_line_image = cv2.dilate(img_temp2, horizontal_kernel, iterations = 3)
cv2.imwrite('d:/extraction/horizontal_lines.jpg', horizontal_line_image)


# weighting parameters, this will decide the quantity of an image to be added to make a new image
alpha = 0.5
beta = 1.0 - alpha

# this function helps to add two image with specific weight parameter to get a third image as summation of two image
img_final_bin = cv2.addWeighted(vertical_line_image, alpha, horizontal_line_image, beta, 0.0)
img_final_bin = cv2.erode(~img_final_bin, kernel, iterations = 2)
(threshold, img_final_bin) = cv2.threshold(img_final_bin, 122, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
cv2.imwrite('d:/extraction/img_final_bin.jpg', img_final_bin)


# find contours for image, which will detect al lthe boxes
contours, img2 = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# loop over all the contours
idx = 0
for c in contours[::-1]:
    x, y, w, h = cv2.boundingRect(c)
    if (w > 80 and h > 20) and (w > 3 * h):
        idx += 1
        new_img = img[y: y + h, x: x + w]
        cv2.imwrite(''.join(['d:/extraction/', str(idx), '.jpg']), new_img)

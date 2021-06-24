# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:56:48 2021

@author: MJH
"""

import numpy as np 
import cv2 as cv
from PIL import Image
from pdf2image import convert_from_path


def PILtoarray(image):
    return np.array(img.getdata(), np.uint).reshape(img.size[1], img.size[0], 3)





def convert_pdf_to_image(file):
    
    return convert_from_path(file, poppler_path = r"C:\Program Files (x86)\poppler-21.03.0\Library\bin", dpi = 600)
    



def get_table(image):
    
    #image = cv.imread(image)
    image = np.array(image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize = 3)
    cv.imshow('image', image)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength = 500, maxLineGap = 100)
    
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
        cv.imshow('image', image)
        
        
 
# threshold the image
(threshold, img_bin) = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
img_bin = 255 - img_bin



# define a kernel length
kernel_length = np.array(image).shape[1] // 50
# vertical kernel of (1 x kernel_length), whichwill detect all the vertical lines from the image
vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_length))
# horizontal kernel of (kernel_length x 1), which will help to detect all the horizontal line from the image.
horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_length, 1))
# kernel of (3 x 3) 
kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))



# morphologial operation to detect vertical lines from an image
img_temp1 = cv2.erode(img_bin, vertical_kernel, iterations = 3)
vertical_line_image = cv2.dilate(img_temp1, vertical_kernel, iterations = 3)
cv2.imwrite('c:/etc/test_img/vertial_lines.jpg', vertical_line_image)

img_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations = 3)
horizontal_line_image = cv2.dilate(img_temp2, horizontal_kernel, iterations = 3)
cv2.imwrite('c:/etc/test_img/horizontal_lines.jpg', horizontal_line_image)



alpha = 0.5
beta = 1.0 - alpha


img_temp1 = cv2.erode(edges, vertical_kernel, iterations = 3)
vertical_line_image = cv2.dilate(img_temp1, vertical_kernel, iterations = 3)
img_temp2 = cv2.erode(edges, horizontal_kernel, iterations = 3)
horizontal_line_image = cv2.dilate(img_temp2, horizontal_kernel, iterations = 3)

img_final_bin = cv2.addWeighted(vertical_line_image, alpha, horizontal_line_image, beta, 0.0)
img_final_bin = cv2.erode(~img_final_bin, kernel, iterations = 2)
(threshold, img_final_bin) = cv2.threshold(img_final_bin, 122, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
cv2.imwrite('c:/etc/test_img/img_final_bin2.jpg', img_final_bin)


contours, img2 = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
idx = 0
for c in contours[::-1]:
    x, y, w, h = cv2.boundingRect(c)
    if (w > 80 and h > 20) and (w > 3 * h):
        idx += 1
        new_img = image[y: y + h, x: x + w]
        cv2.imwrite(''.join(['c:/etc/test_img/', str(idx), '.jpg']), new_img)
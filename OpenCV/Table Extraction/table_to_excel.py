# reference: https://towardsdatascience.com/a-table-detection-cell-recognition-and-text-extraction-algorithm-to-convert-tables-to-excel-files-902edcf289ec

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# read the image
image = cv2.imread('d:/extraction/table.png', 0)
# threshold the image
(threshold, image_binary) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
# invert the image
image_binary = 255 - image_binary
# save
cv2.imwrite('d:/extraction/image_binary.png', image_binary)
# plotting
plotting = plt.imshow(image_binary, cmap = 'gray')
plt.show()


# define a kernel length
kernel_length = np.array(image).shape[1] // 100
# vertical kernel of (1 x kernel_length), whichwill detect all the vertical lines from the image
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
# horizontal kernel of (kernel_length x 1), which will help to detect all the horizontal line from the image.
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
# kernel of (2 x 2) 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))


# morphologial operation to detect vertical lines from an image
vertical_image = cv2.erode(image_binary, vertical_kernel, iterations = 3)
vertical_line_image = cv2.dilate(vertical_image, vertical_kernel, iterations = 3)
cv2.imwrite('d:/extraction/vertial_lines.png', vertical_line_image)

plt.imshow(vertical_line_image, cmap = 'gray')
plt.show()


horizontal_image = cv2.erode(image_binary, horizontal_kernel, iterations = 3)
horizontal_line_image = cv2.dilate(horizontal_image, horizontal_kernel, iterations = 3)
cv2.imwrite('d:/extraction/horizontal_lines.png', horizontal_line_image)

plt.imshow(horizontal_line_image, cmap = 'gray')
plt.show()



# weighting parameters, this will decide the quantity of an image to be added to make a new image
alpha = 0.5
beta = 1.0 - alpha

# this function helps to add two image with specific weight parameter to get a third image as summation of two image
add_image = cv2.addWeighted(vertical_line_image, alpha, horizontal_line_image, beta, 0.0)
add_image = cv2.erode(~add_image, kernel, iterations = 2)
(threshold, add_image) = cv2.threshold(add_image, 122, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
cv2.imwrite('d:/extraction/add_image.png', add_image)

bitxor = cv2.bitwise_xor(image, add_image)
bitnot = cv2.bitwise_not(bitxor)

plt.imshow(add_image, cmap = 'gray')
plt.show()



# find contours for image, which will detect al lthe boxes
contours, boundingBoxes = cv2.findContours(add_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Creating a list of heights for all detected boxes
heights = [coordinates[3] for coordinates in boundingBoxes[0]]
#Get mean of heights
mean = np.mean(heights)


# Create list box to store all boxes in
box = []
# Get position (x,y), width and height for every contour and show the contour on image
for contour in contours[::-1]:
    x, y, w, h = cv2.boundingRect(contour)
    if (w < 700) and (h < 250):
        image = cv2.rectangle(image, (x,y) , (x + w, y + h), (0, 255, 0), 2)
        box.append([x, y, w, h])
        
plt.imshow(image, cmap = 'gray')
plt.show()
cv2.imwrite('d:/extraction/countour.png', image)


# Creating two lists to define row and column in which cell is located
row = []
column = []
j = 0

# Sorting the boxes to their respective row and column
column.append(box[0]) # i == 0
previous = box[0]     # i == 0
for i in range(1, len(box)):    
    if (box[i][1] <= previous[1] + mean / 2):
        column.append(box[i])
        previous = box[i]
        if (i == len(box) - 1):
            row.append(column)
    else:
        row.append(column)
        column = []
        previous = box[i]
        column.append(box[i])
print(column)
print(row)


# calculating maximum number of cells
countcol = max(map(lambda x: len(x), row))
        
        
#Retrieving the center of each column
center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
center = np.array(center)
center.sort()


#Regarding the distance to the columns center, the boxes are arranged in respective order
finalboxes = []
for i in range(len(row)):
    lis = []
    for k in range(countcol):
        lis.append([])
    for j in range(len(row[i])):
        diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
        minimum = min(diff)
        indexing = list(diff).index(minimum)
        lis[indexing].append(row[i][j])
    finalboxes.append(lis)
    
    
# from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
outer = []
for i in range(len(finalboxes)):
    for j in range(len(finalboxes[i])):
        inner = ''
        if (len(finalboxes[i][j]) == 0):
            outer.append(' ')
        else:
            for k in range(len(finalboxes[i][j])):
                y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], finalboxes[i][j][k][3]
                finalimg = bitnot[x: x + h, y: y + w]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2,   cv2.BORDER_CONSTANT, value = [255, 255])
                resizing = cv2.resize(border, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
                dilation = cv2.dilate(resizing, kernel, iterations = 1)
                erosion = cv2.erode(dilation, kernel, iterations = 1)

                
                out = pytesseract.image_to_string(erosion)
                if (len(out) == 0):
                    out = pytesseract.image_to_string(erosion, lang = 'English', config = '-psm 10')
                inner = inner + ' ' + out
            outer.append(inner)
            
            
# Creating a dataframe of the generated OCR list
arr = np.array(outer)
dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
print(dataframe)
data = dataframe.style.set_properties(align = "left")
#Converting it in a excel-file
data.to_excel('d:/extraction/output.xlsx')

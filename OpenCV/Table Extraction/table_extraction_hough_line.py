import cv2 as cv
import numpy as np




def getFile(file):
  
    image = cv.imread(cv.samples.findFile(file))
  
      return image
  
  
def showImage(function):
    def wrapper(image):
        image = function(image)
        cv.imshow('image', image)
        cv.waitKey(0)
        # cv.destroyWindow('image')


@showImage
def showGrayImage(image):
  
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  
      return gray_image
  
  
@showImage
def showCannyImage(gray_image):
  
    canny_image = cv.Canny(gray_image, 50, 150)
  
      return canny_image
  
  
  
  
if __name__ == '__main__':
    file = 'd:/source.png'
    image = getFile(file)
    image_copy = np.copy(image)
  
    showGrayImage(image)
    showCannyImage(image)

import cv2 as cv
import numpy as np




def showImage(function):
    def wrapper(self, image):
        image = function(self, image)
        cv.imshow('image', image)
        cv.waitKey(0)
        # cv.destroyWindow('image')
    
    return wrapper

  
class Image:
          
    @showImage
    def showGrayImage(self, image):
      
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        return gray_image
  
    
    @showImage
    def showCannyImage(self, gray_image):
      
        canny_image = cv.Canny(gray_image, 50, 150)
      
        return canny_image
  
  
  
  
if __name__ == '__main__':
    file = 'd:/source.png'
    image = getFile(file)
    image_copy = np.copy(image)
  
    showGrayImage(image)
    showCannyImage(image)

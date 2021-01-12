import cv2 as cv
import numpy as np




def getFile(file):
  
    image = cv.imread(cv.samples.findFile(file))
  
    return image


def showImage(function):
    def wrapper(self):
        image = function(self)
        cv.imshow('image', image)
        cv.waitKey(0)
        # cv.destroyWindow('image')
    
    return wrapper

  
class Image:
    
    def __init__(self, image):
        self.image = image
        
        
    def getGrayImage(self):
        
        gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        
        return gray_image
    
    
    def getCannyImage(self):
        
        gray_image = self.getGrayImage()
        canny_image = cv.Canny(gray_image, 50, 150)
      
        return canny_image
        
      
    @showImage
    def showGrayImage(self):
      
        gray_image = self.getGrayImage()
        
        return gray_image
        
  
    
    @showImage
    def showCannyImage(self):
      
        canny_image = self.getCannyImage()
      
        return canny_image
  
  
  
  
if __name__ == '__main__':
    file = 'd:/source.png'
    image = getFile(file)
    image_copy = np.copy(image)
  
    showimage = Image(image)
    showimage.showGrayImage()
    showimage.showCannyImage()
    
    RHO = 1
    THETA = np.pi/180
    THRESHOLD = 50
    MIN_LINE_LENGTH = 350
    MAX_LINE_GAP = 6
    hough_lines = cv.HoughLinesP(
        image = showimage.getCannyImage, 
        rho = ROH, 
        theta = THETA, 
        threshold = THRESHOLD, 
        None, 
        minLinLength = MIN_LINE_LENGTH, 
        maxLineGap = MAX_LINE_GAP
        )

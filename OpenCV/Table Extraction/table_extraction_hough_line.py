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

  
def isVertical(line):
  
    return line[0] == line[2]
  

def isHorizontal(line):
  
    return line[1] == line[3] 
  

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
      
      
class Cell:
    
    def __init__(self, hough_lines, image):
        self.__image = image
        self.hough_lines = hough_lines
        self.vertical_lines = []
        self.horizontal_lines = []
        
        self._getVerticalAndHorizontal()
    
    
    @property
    def image(self):
        return self.__image
    
    
    @image.setter
    def image(self, image):
        
        self.__image = image
    
    
    
    def _getVerticalAndHorizontal(self):
        
        if self.hough_lines is not None:
            for i in range(len(self.hough_lines)):
                line = self.hough_lines[i][0]
                
                if ( isVertical(line) ):
                    self.vertical_lines.append(line)
                if ( isHorizontal(line) ):
                    self.horizontal_lines.append(line)
    
    
    @showImage                    
    def enterLine(self):
        
        for _, vertical_and_horizontal_lines in enumerate(zip(self.vertical_lines, self.horizontal_lines)):
            vertical_line, horizontal_line = vertical_and_horizontal_lines
            # vertical
            cv.line(
                self.__image,                          # image
                (vertical_line[0], vertical_line[1]),  # pt1
                (vertical_line[2], vertical_line[3]),  # pt2
                (0, 255, 0),                           # color
                3,                                     # thickness
                cv.LINE_AA                             # lineType
                )
            # horizontal
            cv.line(
                self.__image,                              # image
                (horizontal_line[0], horizontal_line[1]),  # pt1
                (horizontal_line[2], horizontal_line[3]),  # pt2
                (0, 255, 0),                               # color
                3,                                         # thickness
                cv.LINE_AA                                 # lineType
                )
        
        return self.__image
      
      
       
  
if __name__ == '__main__':
    file = 'd:/source.png'
    image = getFile(file)
    image_copy = np.copy(image)
  
    showimage = Image(image)
    showimage.showGrayImage()
    showimage.showCannyImage()
    
    # get hough line
    hough_lines = cv.HoughLinesP(
        showimage.getCannyImage(),   # image
        1,                           # rho
        np.pi / 180,                 # theta
        50,                          # threshold
        None,
        350,                         # minLineLength
        6                            # maxLineGap
        )
    
    # cell
    cell = Cell(hough_lines, image_copy)
    cell.enterLine()

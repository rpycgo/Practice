# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:50:56 2021

@author: MJH
"""


import glob

from PIL import Image
from tqdm import tqdm









class data_loader:
    
    def __init__(self, path):

        self.path = path
        
        self._get_image_lists()
    
    
    def _get_image_lists(self):

        self.image_lists = glob.glob(self.path + '/*png')
        
        
    def get_image(self):

        image = []
        for image_file in tqdm(self.image_lists):
            temp = Image.open(image_file)
            keep = temp.copy()
            image.append(keep)
            temp.close()
            
        return image
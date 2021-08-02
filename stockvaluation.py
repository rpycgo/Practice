# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 19:05:11 2020

@author: MJH
"""


import pandas as pd
import requests
import datetime
import re
from bs4 import BeautifulSoup
from tqdm import tqdm



def parsing(url):
    
    response = requests.get(url)
    parsedResponse = BeautifulSoup(response.text, 'lxml')
   
    return parsedResponse
    

def padding(corp_code):
        
    code_length = len(corp_code)
    
    if code_length < 6:
        return '0' * (6 - code_length) + corp_code
    else:
        return corp_code
    
    

class valuation:
    
    def __init__(self):
        self.URL = 'https://finance.naver.com/item/main.nhn?code='
        self.corp_list = pd.read_csv('D:/Chrome Download/corp_list.csv', encoding = 'cp949')        
        self.corpROELists = pd.DataFrame(columns = ['corp_name', 'corp_code', 'min_ROE', 'ROE', 'net_profit',
                                                    'fair_price_min', 'market_capitalization', 'fair_price_max', 'time'])
        
        self.corp_list['종목코드'] = self.corp_list['종목코드'].apply(lambda x: str(x))        
        self.corp_list['종목코드'] = self.corp_list['종목코드'].apply(lambda x: padding(x))
        
        self.now = datetime.datetime.now()
        self.now = str(self.now.year) + '-' + str(self.now.month) + '-' + str(self.now.day)
    
    def insertROE(self, ROE):
        self.ROE = int(ROE)
    
    
    def getValues(self, index):
        
        url = self.URL + self.corp_list['종목코드'][index]        
        response = requests.get(url)
        parsedResponse = BeautifulSoup(response.text, 'lxml')
    
            
        if parsedResponse.findAll('td', {'class' : 't_line cell_strong'}) == []:
            return
        
        expectedColumn = parsedResponse.findAll('td', {'class' : 't_line cell_strong'})        
        try:
            net_profit = int(re.findall('\d+,?\d{0,3}', str(expectedColumn[1]))[0].replace(',', ''))
        except Exception as e:
            print(e)
            
        try:
            ROE = float(re.findall('\d+\.?\d{0,2}', str(expectedColumn[5]))[0])
        except:
            ROE = float(re.findall('\d+\.?\d{0,2}', str(parsedResponse.findAll('tr', {'class' : 'line_end'})[1].findAll('td')[2]))[0])
        
        capitalization_tag = parsedResponse.findAll('div', {'class' : 'first'})[0].find('td')
        capitalization = capitalization_tag.text.replace('\n', '').replace('\t', '').replace(',', '').replace('억원', '')
        if '조' in capitalization:
            capitalization = capitalization.split('조')
            capitalization = int(capitalization[0]) * 10000 + int(capitalization[1])
        else:
            capitalization = int(capitalization)
                
        data = {
            'corp_name' : self.corp_list['회사명'][index],
            'corp_code' : self.corp_list['종목코드'][index],
            'min_ROE' : self.ROE,
            'ROE' : ROE,
            'net_profit' : net_profit,
            'fair_price_min' : self.ROE * net_profit,
            'market_capitalization' : capitalization,
            'fair_price_max' : ROE * net_profit,
            'time' : self.now}
            
        self.corpRoeLists = self.corpRoeLists.append(data = pd.DataFrame(data = data), ignore_index = True)


val = valuation()
val.insertROE(10)
[val.getValues(i) for i in tqdm(range(val.corp_list.shape[0]))]
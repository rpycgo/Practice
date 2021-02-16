import requests

krx_url = 'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd'

headers = {
'POST': '/comm/bldAttendant/getJsonData.cmd HTTP/1.1',
'Host': 'data.krx.co.kr',
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:85.0) Gecko/20100101 Firefox/85.0',
'Accept': 'application/json, text/javascript, */*; q=0.01',
'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
'Accept-Encoding': 'gzip, deflate',
'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
'X-Requested-With': 'XMLHttpRequest',
'Content-Length': '98',
'Origin': 'http://data.krx.co.kr',
'DNT': '1',
'Connection': 'keep-alive',
'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020101&bld[%E2%80%A6]01&mktId=ALL&trdDd=20210216&share=1&money=1&csvxls_isNo=false',
'Cookie': '__smVisitorID=gYxpfbsCGXk; JSESSIONID=mxYWI7uTIisDloLL9L9pv7Oia9T1bhBojRnmWZ0aPCjWH8TkJVpHCxC1mcS6skX4.bWRjX2RvbWFpbi9tZGNvd2FwMi1tZGNhcHAwMQ=='
}


data = {
        'bld':'dbms/MDC/STAT/standard/MDCSTAT01501',
        'mktId':'ALL',
        'trdDd':'20210216',
        'share':'1',
        'money':'1',
        'csvxls_isNo':'false'
        }

response = requests.get(krx_url, data, headers = headers)

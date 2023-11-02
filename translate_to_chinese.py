import pandas as pd
import requests
import random
from hashlib import md5

def translate_to_chinese(query):

    # Set your own appid/appkey.
    appid = '20230926001830344'
    appkey = 'aWzx9777j2KM6b0xHVhC'

    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    from_lang = 'auto'
    to_lang =  'zh'

    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path
    chinese_result = ''
    # Generate salt and sign
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()
    try:
        salt = random.randint(32768, 65536)
        sign = make_md5(appid + query + str(salt) + appkey)

        # Build request
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

        # Send request
        r = requests.post(url, params=payload, headers=headers)
        result = r.json()

        if len(result['trans_result'])>0:
            for i in result['trans_result']:
                chinese_result+= i['dst']
    except:
        pass
    return chinese_result

if __name__=='__main__':

    data=pd.read_excel('E:\AutoFinance\\meigu.xlsx')
    data['china']=data['stock_name'].map(lambda x: translate_to_chinese(x))
    data.to_excel('E:\AutoFinance\\meigufanyi.xlsx',index=False)

    # chinese_result = translate_to_chinese('AbCellera Biologics Inc. Common Shares')
    # print(chinese_result)

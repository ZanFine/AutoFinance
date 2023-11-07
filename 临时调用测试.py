from data_obtain import data_obtain
import os
import pandas as pd
import math
current_dir = os.getcwd()
# 配置文件
config = pd.read_excel(f'{current_dir}/config.xlsx')
config.fillna('None',inplace=True)

for index,row in config.iterrows():
    config=row
    do = data_obtain.data_obtain(row)
    do.download()
    do.data_clean()




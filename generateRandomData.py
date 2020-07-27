# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:03:40 2020

@author: YuJeong
"""


import csv, random    

csv.register_dialect(
    'mydialect',
    delimiter = ',',
    quotechar = '"',
    doublequote = True,
    skipinitialspace = True,
    lineterminator = '\r\n',
    quoting = csv.QUOTE_MINIMAL)

with open("LargeData.csv",'r') as f:
    f1 = list(csv.reader(f,delimiter=","))
    
    
randName = []
randInst = []
randPrice = []
randDataName = []
randDataType = []
randDate = []
processType = ['암호화', '익명화']
agreeType = ['Yes', 'No']
for i in range(10000):
    randName.append(f1[i][0])
    randInst.append(f1[i][1])
    randDataName.append(f1[i][2])
    randPrice.append(round(int(f1[i][3]), -3))
    randDataType.append(f1[i][4])
    randDate.append(f1[i][7])
    
dataIndex = ['소유자', '소유자id', '소유자type', '데이터', '값', '파일', '생성기관', '행동', '날짜', '기타정보', '행위자', '	행위자id	', '행위자type', '시작기간','종료기간	' ,'가격',	'동의여부']
 
genData = [dataIndex]    
procData = []
provData = []
for i in range(3333):
    genData.append([randName[i], randPrice[i], '개인', randDataType[i], randDataName[i], '', randInst[i], '생성', randDate[i]])
    procData.append([randName[i+3333], randPrice[i+3333], '개인', randDataType[i+3333], randDataName[i+3333], '', randInst[i+3333], '가공', randDate[i+3333], processType[i%2]])
    provData.append([randName[i+6666], randPrice[i+6666], '개인', randDataType[i+6666], randDataName[i+6666], '', randInst[i+6666], '제공', randDate[i+6666], '', randInst[i], randPrice[i], '기관', randDate[i], randDate[i+3333], randPrice[i], agreeType[i%2]])
    
randData = genData + provData + procData    
random.shuffle(randData)  

f = open('output.csv', 'w', encoding='euc-kr', newline='')
wr = csv.writer(f)
for i in range(10000):
    wr.writerow(randData[i])
    
f.close()



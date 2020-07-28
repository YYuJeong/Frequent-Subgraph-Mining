# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:50:09 2020

@author: YuJeong
"""

import csv, sys, time
import itertools
from itertools import product
from itertools import combinations
from itertools import groupby
start_time = time.time()


from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "wowhi223"))

def search_personNode(tx):
    personNodes = tx.run("Match (p:Person) where p.p_type = '기관' return DISTINCT p.name")

    return personNodes
 
def search_perNode(tx):
    personNodes = tx.run("Match (p:Person) where p.p_type = '개인' return DISTINCT p.name")
    return personNodes

def search_dataNode(tx):
    dataNodes = tx.run("Match (d:Data) return d.name, d.value, d.file_path")
    
    return dataNodes 

def get_allGraphs(tx, name):
    allGraphs = tx.run("MATCH p = ({name : $name})-[*]-(connected) "
                       "WHERE length(nodes(p)) = length(apoc.coll.toSet(nodes(p)))"
                       "RETURN p", name = name).values()
    '''
    ag = tx.run("""MATCH (p:Person {name: "강민석"}) 
                    CALL apoc.path.subgraphAll(p, {
                        maxLevel: 2
                    })
                    YIELD nodes, relationships
                    RETURN nodes, relationships;""")
    '''
    
    # separate path to [node, node]
    prov = [] 
    for g in allGraphs:
        if len(g[0]) == 2:
            prov.append(list(g[0].nodes))

    pr2dic = []
    for pr in prov:
        p2dic = []
        p2dic = [[pr[0], pr[1]], [pr[1], pr[2]]]
        pr2dic.append(p2dic)
        

        
    # direction of node : [node, node] means node -> node  
    processFlag = 0
    for pr in pr2dic:
        pr[0].reverse()
        if 'Data' in pr[1][1].labels: #행동 뒤 데이터가 오는 경우 방향 바꿔야함
            pr[1].reverse() 
            if pr[1][1].get('name') == '가공' and processFlag == 1:
                pr[1].reverse()
                processFlag = 0
            if pr[1][1].get('name') == '가공':
                processFlag = 1

    # flatten graph list
    gList = []
    for p in pr2dic:
        gList.append(p[0])
        gList.append(p[1])
            

    # encoding graph node to dictionary
    graph2Dic = []
    for gl in gList:
        node2Dic = []        
        for g in gl:
            if 'Person' in g.labels:
                node2Dic.append(allDict[g.get('name')])
            elif 'Data'in g.labels:
                if g.get('value') == '':
                    dataTuple = (g.get('name'), g.get('file_path'))
                else:
                    dataTuple = (g.get('name'), g.get('value'))
                node2Dic.append(allDict[dataTuple])
            elif 'Activity' in g.labels:
                node2Dic.append(allDict[g.get('name')])
        graph2Dic.append(node2Dic)

    return graph2Dic
  

with driver.session() as session:
     
     # All personNodes to dict 
     personNodes = session.read_transaction(search_personNode)
     perNodes = session.read_transaction(search_perNode)
     records = []
     for personNode in personNodes:
         records.append(personNode["p.name"])
     personDict = {k: v for v, k in enumerate(records)}

     '''
     records = []
     for personNode in perNodes:
         records.append('Person')
     perDict = {k: v for v, k in enumerate(records)}
     '''
     perDict = {'개인': 0}
     dataDict = {'데이터': 0}
     #dataNodes to dict
     dataNodes = session.read_transaction(search_dataNode)
     records = []
     datas = []
     for dataNode in dataNodes:
         datas.append(dataNode["d.name"])
         if dataNode["d.value"] == '' :
             datas.append(dataNode["d.file_path"])
         else:    
             datas.append(dataNode["d.value"])
         records.append(datas)
         datas = []
     dataDict = {tuple(k): (v+len(personDict)) for v, k in enumerate(records)}
     
     #activityNodes to dict
     activityNodes = ['생성', '가공', '제공']
     actDict = {k: (v+len(personDict)+len(dataDict)) for v, k in enumerate(activityNodes)}
        
     allDict = {**personDict, **dataDict, **actDict,}
     print(allDict)
     
     #get all graphs 
     ''' 
     allGraph2Dic : Neo4j의 모든 이력 그래프들이 딕셔너리로 표현되어 저장
     len(allGraph2Dic) : Neo4j에 저장된 이력 수
     allGraph2Dic[i] : i-번째 이력그래프가 [[node1, node2],... , [nodeN-1, nodeN]] 형태로 저장됨
     [node1, node2]: node1과 node2가 node1 -> node2 방향으로 연결
     '''
     allGraph2Dic = []
     for key in perDict:
         allGraph2Dic.append(session.read_transaction(get_allGraphs, key))
 
         
driver.close()

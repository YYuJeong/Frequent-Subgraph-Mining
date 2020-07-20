# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:50:09 2020

@author: YuJeong
"""

import csv, sys, time
start_time = time.time()


from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "wowhi223"))

def search_personNode(tx):
    personNodes = tx.run("Match (p:Person) where p.p_type = '개인' return DISTINCT p.name")

    return personNodes

def search_dataNode(tx):
    dataNodes = tx.run("Match (d:Data) return d.name, d.value, d.file_path")
    
    return dataNodes 

def get_allGraphs(tx, name):
    allGraphs = tx.run("MATCH ({name : $name})-[*]-(connected) "
                       "RETURN DISTINCT connected", name = name)
    graph_tmp = []
    for graph in allGraphs:
        print(graph.value())
        graph_tmp.append(graph.value())

    return graph_tmp
  
with driver.session() as session:
     
     #personNodes to dict 
     personNodes = session.read_transaction(search_personNode)
     records = []
     for personNode in personNodes:
         records.append(personNode["p.name"])
     personDict = {k: v for v, k in enumerate(records)}
     
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
     nodes = []
     for key in personDict:
         nodes.append(session.read_transaction(get_allGraphs, key))
     
     
         
         
driver.close()

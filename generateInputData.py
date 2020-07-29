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
    dataNodes = tx.run("Match (d:Data) return  DISTINCT d.name")
    
    return dataNodes 

def get_allGraphs(tx, name, allDict, edgeDict):
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
    nodes = []
    for g in allGraphs:
        #print(g)
        if len(g[0]) == 2:
            prov.append([g[0].relationships[0].nodes[0], g[0].relationships[0].nodes[1], g[0].relationships[0].type])
            prov.append([g[0].relationships[1].nodes[0], g[0].relationships[1].nodes[1], g[0].relationships[1].type])
            nodes.append(g[0].relationships[0].nodes[0])
            nodes.append(g[0].relationships[0].nodes[1])
            nodes.append(g[0].relationships[1].nodes[0])
            nodes.append(g[0].relationships[1].nodes[1])

    nodes = list(set(nodes))
    
    allnode2Dic = []
    for n in nodes:
        if 'Person' in n.labels:
            if n.get('p_type') == '기관':
                allnode2Dic.append(allDict[n.get('name')])  
            else:
                allnode2Dic.append(allDict['개인'])
        elif 'Data'in n.labels:
            allnode2Dic.append(allDict[n.get('name')])
        elif 'Activity' in n.labels:
            allnode2Dic.append(allDict[n.get('name')])
            
    allnode2Dic = list(set(allnode2Dic))        
            
    # encoding graph node to dictionary
    graph2Dic = []
    for gl in prov:
        node2Dic = []        
        for g in gl:
            if type(g) != str:
                if 'Person' in g.labels:
                    if g.get('p_type') == '기관':
                        node2Dic.append(allDict[g.get('name')])  
                    else:
                        node2Dic.append(allDict['개인'])
                elif 'Data'in g.labels:

                    node2Dic.append(allDict[g.get('name')])
                elif 'Activity' in g.labels:
                    node2Dic.append(allDict[g.get('name')])
            else:
                node2Dic.append(edgeDict[g])
        
        graph2Dic.append(node2Dic)

    return allnode2Dic , graph2Dic

def generateInput():
    with driver.session() as session:
         
         # All personNodes to dict 
         personNodes = session.read_transaction(search_personNode)
         perNodes = session.read_transaction(search_perNode)
    
         records = []
         for personNode in perNodes:
             records.append(personNode["p.name"])
         personDict = {k: v for v, k in enumerate(records)}
    
         perDict = {'개인': 0}
         records = []
        
         for personNode in personNodes:
             records.append(personNode["p.name"])
         instDict = {k: v + len(perDict) for v, k in enumerate(records)}
       
    
         #dataDict = {'데이터': 0}
         #dataNodes to dict
         dataNodes = session.read_transaction(search_dataNode)
         records = []
         for dataNode in dataNodes:
             records.append(dataNode["d.name"])

         dataDict = {k: (v+len(perDict)+len(instDict)) for v, k in enumerate(records)}
    
         
         #activityNodes to dict
         activityNodes = ['생성', '가공', '제공']
         actDict = {k: (v+len(perDict)+len(dataDict)+len(instDict)) for v, k in enumerate(activityNodes)}
         
         #edge labels to dict
         edgeLabels = ['Act', 'Generate', 'Send', 'Receive']
         edgeDict = {k: v for v, k in enumerate(edgeLabels)}
    
         allDict = {**perDict, **instDict,**dataDict, **actDict}
         print(allDict)
         
         #get all graphs 
         ''' 
         allGraph2Dic : Neo4j의 모든 이력 그래프들이 딕셔너리로 표현되어 저장
         len(allGraph2Dic) : Neo4j에 저장된 이력 수
         allGraph2Dic[i] : i-번째 이력그래프가 [[node1, node2],... , [nodeN-1, nodeN]] 형태로 저장됨
         [node1, node2]: node1과 node2가 node1 -> node2 방향으로 연결
         '''
        
         
         allGraph2Dic = []
         for key in personDict:
             
             allGraph2Dic.append(list(session.read_transaction(get_allGraphs, key, allDict, edgeDict)))
         
      
    driver.close()
    return allGraph2Dic

def numberingIndex(graph2Dic):

    ver2Dict = {k: v for v, k in enumerate(graph2Dic[0])}
    
    del graph2Dic[0]
    
    for n in graph2Dic[0]:
        n[0] = ver2Dict[n[0]]
        n[1] = ver2Dict[n[1]]
    
    return ver2Dict, graph2Dic[0]
    

if __name__ == '__main__':
    allGraph2Dic = generateInput()
    
    #dictionary {node2Dic : index}
    gspanInput = []
    for graph2Dic in allGraph2Dic:
        gspanInput.append(list(numberingIndex(graph2Dic)))
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
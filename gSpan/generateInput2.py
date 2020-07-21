# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 01:21:25 2020

@author: DILab
"""
import networkx as nx
import sys

from neo4j import GraphDatabase
from numpy import array
from numpy import argmax
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

sys.stdout=open('./graphdata/output.txt','w')

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "wowhi223"))
"""
def export_query():
    # neo4j.conf에서 apoc.import.file.enabled=true 로 변경
    input_query = "CALL apoc.export.graphml.all(\"graph.csv\",{})"
    with driver.session() as session:
        results = session.run(input_query, parameters={})
        return results
    
results = export_query()

"""
def run_query(input_query):
    """
    - input_query를 전달받아서 실행하고 그 결과를 출력하는 함수입니다.
    """
    # 세션을 열어줍니다.
    with driver.session() as session: 
        # 쿼리를 실행하고 그 결과를 results에 넣어줍니다.
        results = session.run(
            input_query,
            parameters={}
        )
        return results

# 다음과 같이 쿼리하여, 노드, 엣지, 노드를 모두 가져온다.
# 그래도 전체를 다 가져오는 경우 부하가 많이 걸리므로, 100로 제한하여 가져와 본다.
input_query = """
MATCH (n1)-[e]->(n2)
RETURN n1, e, n2
"""

results = run_query(input_query)
# result => neo4j.BoltStatementResult object
"""
print(results)
print(type(results))
print("=="*30)

"""


# 긁어온 쿼리를 다음의 방향성이 있는 그래프에 넣어준다.
DG = nx.DiGraph()

for i, path in enumerate(results):
    # 앞서, 쿼리에서 변수명을 n1, n2, e, 로 가져왔으므로 각 값에 할당된 것을 변수에 추가로 넣어준다.
    n1, n2, e = path['n1'], path['n2'], path['e']
    # 그리고, 보통 노드의 경우는 id, labels, properties 로 나누어 정보가 저장되어 있다.
    # 이를 가져오기 편하게, dictionary로 변경한다. 
    n1_dict = {
        'id': path['n1'].id, 
        'labels':path['n1'].labels, 
        'properties':dict(path['n1'])
    }
    n2_dict = {
        'id': path['n2'].id, 
        'labels':path['n2'].labels, 
        'properties':dict(path['n2'])
    }
    # 마찬가지로, edge의 경우도 아래와 같이 정보를 저장한다.
    e_dict = {
        'id':path['e'].id, 
        'type':path['e'].type, 
        'properties':dict(path['e'])
    }
    # print(e_dict)
    # 해당 노드를 넣은 적이 없으면 넣는다.
    if n1_dict['id'] not in DG:
        DG.add_nodes_from([
            (n1_dict['id'], n1_dict)
        ])
    # 해당 노드를 넣은 적이 없으면 넣는다.
    if n2_dict['id'] not in DG:
        DG.add_nodes_from([
            (n2_dict['id'], n2_dict)
        ])
    # edge를 넣어준다. 노드의 경우 중복으로 들어갈 수 있으므로 중복을 체크해서 넣어주지만, 
    # edge의 경우 중복을 체크하지 않아도 문제없다.
    DG.add_edges_from([
        (n1_dict['id'], n2_dict['id'], e_dict)
    ])
"""
print("=="*30)
print("==: network is generated")
"""

"""
for n in DG.nodes(data=True):
    print(n[1]['id'], n[1]['labels'], n[1]['properties'])
    print("--"*30)

for e in DG.edges(data=True):
    print(e)
""" 
#DataName=[]
#DataFilePath=[]
#DataValue=[]
#DataOrigin=[]

#ActivityDate=[]
#ActivityName=[]
#ActivityDetail=[]

#PersonName=[]
#PersonP_Type=[]
#PersonPid=[]
NodeLabel=[]
NodeLabel.append([])
NodeLabel.append([])
EdgeLabel=[]
FromIdx=[]
ToIdx=[]
From=[]
To=[]
Name=[]
NodeLabelEncoded=[]
EdgeLabelEncoded=[]

for n in DG.nodes(data=True):

	if 'Person' in n[1]['labels']:
		Name.append(n[1]['properties']['name'])
		
count=0



for name in Name:
    
    del NodeLabel[:]
    NodeLabel.append([])
    NodeLabel.append([])
    del EdgeLabel[:]
    del FromIdx[:]
    del ToIdx[:]
    del From[:]
    del To[:]

    n1_dict.clear()
    n2_dict.clear()
    e_dict.clear()
    
    input_query = "match (n1:Person{name:\""+name+"\"})match (n1)-[e1]-(n2)-[e2]-(n3)-[e3]-(n4) return n1,n2,n3,n4,e1,e2,e3 limit 25"
    results = run_query(input_query)
    DG1 = nx.DiGraph()

    for i, path in enumerate(results):
        
    # 앞서, 쿼리에서 변수명을 n1, n2, e, 로 가져왔으므로 각 값에 할당된 것을 변수에 추가로 넣어준다.
        n1, n2, n3, n4, e1, e2, e3 = path['n1'], path['n2'], path['n3'], path['n4'], path['e1'], path['e2'], path['e3']
    # 그리고, 보통 노드의 경우는 id, labels, properties 로 나누어 정보가 저장되어 있다.
    # 이를 가져오기 편하게, dictionary로 변경한다. 
        
        n1_dict = {
                'id': path['n1'].id, 
                'labels':path['n1'].labels, 
                'properties':dict(path['n1'])
                }
        n2_dict = {
                'id': path['n2'].id, 
                'labels':path['n2'].labels, 
                'properties':dict(path['n2'])
                }
        n3_dict = {
                'id': path['n3'].id, 
                'labels':path['n3'].labels, 
                'properties':dict(path['n3'])
                }
        n4_dict = {
                'id': path['n4'].id, 
                'labels':path['n4'].labels, 
                'properties':dict(path['n4'])
                }
    # 마찬가지로, edge의 경우도 아래와 같이 정보를 저장한다.
        e1_dict = {
                'id':path['e1'].id, 
                'type':path['e1'].type, 
                'properties':dict(path['e1'])
                }
        e2_dict = {
                'id':path['e2'].id, 
                'type':path['e2'].type, 
                'properties':dict(path['e2'])
                }
        e3_dict = {
                'id':path['e3'].id, 
                'type':path['e3'].type, 
                'properties':dict(path['e3'])
                }
    # print(e_dict)
    # 해당 노드를 넣은 적이 없으면 넣는다.
        if n1_dict['id'] not in DG1:
           DG1.add_nodes_from([
                   (n1_dict['id'], n1_dict)
                   ])
    # 해당 노드를 넣은 적이 없으면 넣는다.
        if n2_dict['id'] not in DG1:
            DG1.add_nodes_from([
                    (n2_dict['id'], n2_dict)
                    ])
        if n3_dict['id'] not in DG1:
            DG1.add_nodes_from([
                    (n3_dict['id'], n3_dict)
                    ])
        if n4_dict['id'] not in DG1:
            DG1.add_nodes_from([
                    (n4_dict['id'], n4_dict)
                    ])
    # edge를 넣어준다. 노드의 경우 중복으로 들어갈 수 있으므로 중복을 체크해서 넣어주지만, 
    # edge의 경우 중복을 체크하지 않아도 문제없다.
        DG1.add_edges_from([
                (n1_dict['id'], n2_dict['id'], e1_dict),(n2_dict['id'], n3_dict['id'], e2_dict),(n3_dict['id'], n4_dict['id'], e3_dict)
                ])
    """
        DG1.add_edges_from([
                (n2_dict['id'], n3_dict['id'], e2_dict)
                ])
        DG1.add_edges_from([
                (n3_dict['id'], n4_dict['id'], e3_dict)
                ])

    print("=="*30)

    """
    
    for n in DG1.nodes(data=True):
        
        if 'Data' in n[1]['labels']:
            NodeLabel[0].append(n[1]['properties']['name'])        
            NodeLabel[1].append(n[1]['id'])
		#DataOrigin.append(n[1]['properties']['origin'])
        if 'Activity' in n[1]['labels']:
            NodeLabel[0].append(n[1]['properties']['name'])        
            NodeLabel[1].append(n[1]['id'])
		#ActivityDetail.append(n[1]['properties']['detail'])
        if 'Person' in n[1]['labels']:
            NodeLabel[0].append('Person')
            NodeLabel[1].append(n[1]['id'])

    for e in DG1.edges(data=True):
        FromIdx.append(NodeLabel[1].index(e[0]))
        ToIdx.append(NodeLabel[1].index(e[1]))
        EdgeLabel.append(e[2]['type'])
        
    NodeLen = len(NodeLabel[0])
    EdgeLen = len(EdgeLabel)
    
    NodeLabelArray = array(NodeLabel[0])
    EdgeLabelArray = array(EdgeLabel)
    
    #From.append(NodeLabel[0][FromIdx])
    #To.append(NodeLabel[0][ToIdx])
                

		

    NodeLabel_encoder = LabelEncoder()
    EdgeLabel_encoder = LabelEncoder()


    NodeLabelEncoded = NodeLabel_encoder.fit_transform(NodeLabelArray)
    EdgeLabelEncoded = EdgeLabel_encoder.fit_transform(EdgeLabelArray)

    for i in range(EdgeLen):
        From.append(NodeLabelEncoded[FromIdx[i]])
        To.append(NodeLabelEncoded[ToIdx[i]])

    print("t # "+str(count))
    for i in range(NodeLen):
        print("v " + str(i) + " " + str(NodeLabelEncoded[i]))
    for i in range(EdgeLen):
        print("e " + " " + str(FromIdx[i]) + " " + str(ToIdx[i]) + " " + str(EdgeLabelEncoded[i]))
    count+=1
    DG1.clear()
print("t # -1")    



 
"""
DataNum = len(DataName)
ActivityNum = len(ActivityName)
PersonNum = len(PersonName)
"""


"""
DataNameArray = array(DataName)
DataFilePathArray = array(DataFilePath)
DataValueArray = array(DataValue)
DataOriginArray = array(DataOrigin)

ActivityNameArray = array(ActivityName)
ActivityDateArray = array(ActivityDate)
ActivityDetailArray = array(ActivityDetail)

PersonNameArray = array(PersonName)
PersonP_TypeArray = array(PersonP_Type)
PersonPidArray = array(PersonPid)
"""
"""
DataNameEncoded = label_encoder.fit_transform(DataNameArray)
DataFilePathEncoded = label_encoder.fit_transform(DataFilePathArray)
DataValueEncoded = label_encoder.fit_transform(DataValueArray)
DataOriginEncoded = label_encoder.fit_transform(DataOriginArray)

ActivityNameEncoded = label_encoder.fit_transform(ActivityNameArray)
ActivityDateEncoded = label_encoder.fit_transform(ActivityDateArray)
ActivityDetailEncoded = label_encoder.fit_transform(ActivityDetailArray)

PersonNameEncoded = label_encoder.fit_transform(PersonNameArray)
PersonP_TypeEncoded = label_encoder.fit_transform(PersonP_TypeArray)
PersonPidEncoded = label_encoder.fit_transform(PersonPidArray)
"""


   



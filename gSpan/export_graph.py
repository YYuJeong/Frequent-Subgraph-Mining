# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 01:21:25 2020

@author: DILab
"""
import networkx as nx

from neo4j import GraphDatabase
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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
LIMIT 100
"""

results = run_query(input_query)
# result => neo4j.BoltStatementResult object
print(results)
print(type(results))
print("=="*30)

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
print("=="*30)
print("==: network is generated")



for n in DG.nodes(data=True):
    print(n[1]['id'], n[1]['labels'], n[1]['properties'])
    print("--"*30)
    
#for e in DG.edges(data=True):
#    print(e)

DataName=[]
DataFilePath=[]
DataValue=[]
DataOrigin=[]

ActivityDate=[]
ActivityName=[]
ActivityDetail=[]

PersonName=[]
PersonP_Type=[]
PersonPid=[]

DataNodeLabel=[]
ActivityNodeLabel=[]
PersonNodeLabel=[]

 

for n in DG.nodes(data=True):

	if 'Data' in n[1]['labels']:
		DataName.append(n[1]['properties']['name'])
		DataFilePath.append(n[1]['properties']['file_path'])
		DataValue.append(n[1]['properties']['value'])
		DataOrigin.append(n[1]['properties']['origin'])

	if 'Activity' in n[1]['labels']:
		ActivityName.append(n[1]['properties']['name'])
		ActivityDate.append(n[1]['properties']['date'])
		ActivityDetail.append(n[1]['properties']['detail'])

	if 'Person' in n[1]['labels']:
		PersonName.append(n[1]['properties']['name'])
		PersonP_Type.append(n[1]['properties']['p_type'])
		PersonPid.append(n[1]['properties']['pid'])

 

DataNum = len(DataName)
ActivityNum = len(ActivityName)
PersonNum = len(PersonName)


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


label_encoder = LabelEncoder()

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


Datamuchjari = len(str(DataName))
Activitymuchjari = len(str(ActivityName))
Personmuchjari = len(str(PersonName))

 

 


"""
for i in range(len(DataName)):
    DataNodeLabel.append(str(DataNameEncoded[i]).zfill(Datamuchjari)+str(DataFilePathEncoded[i]).zfill(Datamuchjari)+str(DataValue[i]).zfill(Datamuchjari)+str(DataOriginEncoded[i]).zfill(Datamuchjari)) 
    print(DataNodeLabel[i])
    
for i in range(len(ActivityName)):
    ActivityNodeLabel.append(str(ActivityNameEncoded[i]).zfill(Activitymuchjari)+str(ActivityDateEncoded[i]).zfill(Activitymuchjari)+str(ActivityDetailEncoded[i]).zfill(Activitymuchjari))
    print(ActivityNodeLabel[i])

for i in range(len(PersonName)): 
    PersonNodeLabel.append(str(PersonNameEncoded[i]).zfill(Personmuchjari)+str(PersonP_TypeEncoded[i]).zfill(Personmuchjari) + str(PersonPidEncoded[i]).zfill(Personmuchjari))
    print(PersonNodeLabel[i])
"""
print("data")
for i in range(len(DataName)):
    DataNodeLabel.append(str(DataNameEncoded[i])+"0"+str(DataFilePathEncoded[i])+"0"+str(DataValueEncoded[i])+"0"+str(DataOriginEncoded[i])) 
    print(DataNodeLabel[i])
    
print("activity")
for i in range(len(ActivityName)):
    ActivityNodeLabel.append(str(ActivityNameEncoded[i])+"0"+str(ActivityDateEncoded[i])+"0"+str(ActivityDetailEncoded[i]))
    print(ActivityNodeLabel[i])

print("person")
for i in range(len(PersonName)): 
    PersonNodeLabel.append(str(PersonNameEncoded[i])+"0"+str(PersonP_TypeEncoded[i])+"0" + str(PersonPidEncoded[i]))
    print(PersonNodeLabel[i])
    

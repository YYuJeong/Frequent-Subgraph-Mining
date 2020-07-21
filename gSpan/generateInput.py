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


for n in DG.nodes(data=True):
    
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
		Name.append(n[1]['properties']['name'])        
	print(n)

        


for e in DG.edges(data=True):
    FromIdx.append(NodeLabel[1].index(e[0]))
    ToIdx.append(NodeLabel[1].index(e[1]))
    
    #From.append(NodeLabel[0][FromIdx])
    #To.append(NodeLabel[0][ToIdx])
    EdgeLabel.append(e[2]['type'])

for name in Name:
    input_query = "match (n1:Person{name:\""+name+"\"})match (n1)-[*1..3]-(n2) return n1,n2 limit 25"
    results = run_query(input_query)


		
 
"""
DataNum = len(DataName)
ActivityNum = len(ActivityName)
PersonNum = len(PersonName)
"""

NodeLen = len(NodeLabel)
EdgeLen = len(EdgeLabel)

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

NodeLabelArray = array(NodeLabel[0])
EdgeLabelArray = array(EdgeLabel)

NodeLabel_encoder = LabelEncoder()
EdgeLabel_encoder = LabelEncoder()
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
NodeLabelEncoded = NodeLabel_encoder.fit_transform(NodeLabelArray)
EdgeLabelEncoded = EdgeLabel_encoder.fit_transform(EdgeLabelArray)


for i in range(EdgeLen):
    From.append(NodeLabelEncoded[FromIdx[i]])
    To.append(NodeLabelEncoded[ToIdx[i]])


print(NodeLabelEncoded)
print(EdgeLabelEncoded)
print(EdgeLen)
print(From)
print(To)





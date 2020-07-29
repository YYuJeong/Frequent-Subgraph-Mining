"""The main program that runs gSpan."""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

#from config import parser
from gspan import gSpan

#from gspan import allDict
#from gspan import ver2Dict
#from gspan import edgeDict
#from gspan import instDict
#from gspan import dataDict

from gspan import *

def main(FLAGS=None):
    """Run gSpan."""
    '''
    if FLAGS is None:
        FLAGS, _ = parser.parse_known_args(args=sys.argv[1:])

    if not os.path.exists(FLAGS.database_file_name):
        print('{} does not exist.'.format(FLAGS.database_file_name))
        sys.exit()
    '''    
    gs = gSpan(
        #database_file_name=FLAGS.database_file_name,
        database_file_name='../gSpan-master/graphdata/neo4j.txt',
        
        #min_support=FLAGS.min_support,
        min_support=2,
        
#        min_num_vertices=FLAGS.lower_bound_of_num_vertices,
#        max_num_vertices=FLAGS.upper_bound_of_num_vertices,

#        max_ngraphs=FLAGS.num_graphs,
        is_undirected=True,
#        verbose=FLAGS.verbose,
#        visualize=FLAGS.plot,
#        where=FLAGS.where
    )

    gs.run()
    gs.time_stats()
    return gs


if __name__ == '__main__':
    gs = main()


    print(edgeDict)
    print(allDict)
    print(ver2Dict)
    print(instDict)
    # Create an empty list 
    row_list =[] 
      
    # Iterate over each row 
    for index, rows in gs._report_df.iterrows(): 
        # Create list for the current row 
        my_list =[rows.support, rows.vertex, rows.link, rows.num_vert] 
          
        # append the list to the final list 
        row_list.append(my_list) 
        
    #Generate ouputTable
    edgeInfo = row_list[38]  #example
   
    #Extract row's information
    support = edgeInfo[0]
    nodes = edgeInfo[1]
    edges = edgeInfo[2]
    numVer = edgeInfo[3]
   
    #Create table output
   
    #Create cypher output
    
    #decode dict to node
    dic2graph = []
    for edge in edges:
        #print(edge)
        dic2node = []
        dic2node.append(list(allDict.keys())[list(allDict.values()).index(edge[0])])
        dic2node.append(list(allDict.keys())[list(allDict.values()).index(edge[1])])
        dic2node.append(list(edgeDict.keys())[list(edgeDict.values()).index(edge[2])])          
        #(dic2node)
        dic2graph.append(dic2node)
    
    #look up original node
    fsmResult = []
    for nodes in dic2graph:
        dic2node = []
        #print(nodes)
        for node in nodes:
            if instDict.get(node) != None:
                label = '기관'
            elif dataDict.get(node) != None:
                label = 'Data'
            elif actDict.get(node) != None:
                label = 'Activity'
            elif edgeDict.get(node) != None:
                label = 'Edge'
            else:
                label = '개인'
            dic2node.append((node, label))
        #print(dic2node)
        fsmResult.append(dic2node)

    #create cypher by activity type
    for result in fsmResult:
        print(result)
        node1 = result[0]
        node2 = result[1]
        edge = result[2]
        cypher = ''
        if node2[0] == '생성':
            if edge[0] == 'Generate':
                cypher = ("CREATE (d:Data), (ac:Activity) "
                         "SET d = {name: " + "'" + node1[0] +"'"+ "},"
                         "   ac = {name: " + "'" + node2[0] +"'" +  "}"
                         "CREATE (ac) <- [g:Generate] - (d)")
            elif edge[0] == 'Act':
                cypher = ("CREATE (p:Person), (ac:Activity) "
                         "SET p = {name: " + "'" + node1[0] + "'" +  "},"
                         "   ac = {name: " + "'" + node2[0] + "'" +  "}"
                         "CREATE (ac) - [a:Act] -> (p)")
        elif node2[0] == '가공':
            if edge[0] == 'Generate':
                cypher = ("CREATE (d:Data), (ac:Activity) "
                         "SET d = {name: " + "'" + node1[0] + "'" +  "},"
                         "   ac = {name: " + "'" + node2[0] + "'" +  "}"
                         "CREATE (ac) <- [g:Generate] - (d)")
            elif edge[0] == 'Act':
                cypher = ("CREATE (p:Person), (ac:Activity) "
                         "SET p = {name: " + "'" + node1[0] + "'" + "},"
                         "   ac = {name: " + "'" + node2[0] + "'" + "}"
                         "CREATE (ac) - [a:Act] -> (p)")
        elif node2[0] == '제공':
            if edge[0] == 'Generate':
                cypher = ("CREATE (d:Data), (ac:Activity) "
                         "SET d = {name: " + "'" + node1[0] + "'" + "},"
                         "   ac = {name: " + "'" + node2[0] + "'" + "}"
                         "CREATE (ac) <- [g:Generate] - (d)")
            elif edge[0] == 'Send':
                cypher = ("CREATE (p:Person), (ac:Activity) "
                         "SET p = {name: " + "'" + node1[0] + "'" + "},"
                         "   ac = {name: " + "'" + node2[0] + "'" + "}"
                         "CREATE (ac) - [s:Send] -> (p)")
            elif edge[0] == 'Receive':
                cypher = ("CREATE (p:Person), (ac:Activity) "
                         "SET p = {name: " + "'" + node1[0] + "'" + "},"
                         "   ac = {name: " + "'" + node2[0] + "'" + "}"
                         "CREATE (ac) - [r:Receive] -> (p)")
        print(cypher)
            
        
        
        
        
        
        
        
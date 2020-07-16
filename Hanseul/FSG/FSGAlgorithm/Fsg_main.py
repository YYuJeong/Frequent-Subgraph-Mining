import sys
import itertools
from pprint import pprint

trxes = list()

# Before Optimization
# 작동 안하는 추상화 코드

class Node():
    m_iData = 0
    m_iEdge = 0
    m_iLabel = 0
    m_iWeight = 0
    m_iDegree = 0

    def __init__(self, _iData, _iEdge, _iLabel, _iWeight, _iDegree):
        self.m_iData = _iData
        self.m_iEdge = _iEdge
        self.m_iLabel = _iLabel
        self.m_iWeight = _iWeight
        self.m_iDegree = _iDegree

class Edge():
    m_LeftNode
    m_RightNode

    def __init__(self):
        self.m_LeftNode = Node()
        self.m_RightNode = Node()

# 껍데기 실행함수
def fsg(D,sigma):
    # None
    return

def fsg_gen(frequentSet):
    candidate = list() # C(K+1) <- 0
    for i in range(frequentSet.count()): # for each paif of G(k), i <=j (cl (G Ki) <= cl (G kj))
        if i%2 != 0 : 
            continue
        itemSet1 = frequentSet[i]
        itemSet2 = frequentSet[i+1]

        hCoreSet = coreDetection(itemSet1, itemSet2)

        for core in hCoreSet :
            TentCandiSet = fsg_join(itemSet1, itemset2, hCoreSet)
            for graph in TentCandiSet:
                flag = true
                for edge in graph :
                    hEdgeCore = graph - edge
                    if hEdgeCore != frequentSet:
                        flag = false
                        break
                if flag :
                    candidate += graph
    return candidate

def fsg_join(subgraphG1, subgraphG2, subgraphH1):
    # e1 - the edge appears only in g1, not in h1
    # e2 - the edge appears only in g2, not in h1
    # M - all automorphisms of h1
    candidate = list()

    # for each automorphism M
    # candidate += (edge1, edge2, core) contained set
    return candidate

def coreDetection(CurGraph, NextGraph):
    coreSet = list()
    if CurGraph == NextGraph :
        return CurGraph

    rootNode = Node()
    
    for edge in CurGraph :
        # 가중치를 비교하여 노드의 차수를 변경한다
        # 가중치의 절대비교값 계산
        iSubsWeight = abs(edge.m_LeftNode.m_iWeight - edge.m_RightNode.m_iWeight)

        if edge.m_LeftNode.m_iWeight < edge.m_RightNode.m_iWeight :
            edge.m_LeftNode.m_iDegree -= iSubsWeight
            edge.m_RightNode.m_iDegree += iSubsWeight
        elif egde.m_LeftNode.m_iWeight >= edge.m_RightNode.m_iWeight :
            edge.m_LeftNode.m_iDegree += iSubsWeight
            edge.m_RightNode.m_iDegree -= iSubsWeight

    for node in CurGraph:
        if node.m_iWeight >= 0 :
            coreSet.append(node)
    return coreSet


def apply_association_rule(length, frequent_set):
    for item_set, freq in frequent_set.items():
        frequent_set_len = length
        
        ## 예를 들어 길이가 4인 key가 있다면 (3,1) (2,2) (1,3) 이렇게 다 조합을 만들어준다
        while frequent_set_len > 1:
            comb = list(itertools.combinations(item_set, frequent_set_len-1))
            for item in comb:
                item = set(item)
                
                ## 차집합을 이용해서 반대편 조합을 저장해놓는다
                counterpart = set(item_set) - set(item) 
                
                support = freq / len(trxes) * 100
                
                cnt_item = 0
                for trx in trxes:
                    if set(trx) >= item:
                        cnt_item = cnt_item + 1
                confidence = freq / cnt_item * 100 
                
                ## 채점 하기 위해 요소들을 int로 바꿔줌
                item = set(map(int, item))
                counterpart = set(map(int, counterpart))

                line = str(item) + '\t' + str(counterpart) + '\t' + str('%.2f' % round(support, 2)) + '\t' + str('%.2f' % round(confidence, 2)) + '\n'
                save_result(line)
               
            frequent_set_len = frequent_set_len - 1

def prune(length, previous_frequent_set, candidate):
    global trxes
    frequent_set = dict() 

    ## 길이가 2일 때는 previous_frequent_set의 candidate 길이가 1이니 리스트 안의 리스트로 넣어준다
    if length == 2:
        temp = list() 
        for item in previous_frequent_set:
            ## 리스트 요소를 그냥 append하면 '14'의 경우 '1''4'로 짤려온다 
            temp.append(list([item,])) 
        previous_frequent_set = temp
    else:
    ## 길이가 3 이상의 candidate를 만들 때는 각 transaction들을 set로 감싸준다
        previous_frequent_set = change_element_to_set(previous_frequent_set)
    
    ## Downward closure property
    for item_set in candidate: 
        cnt = 0
        for item in list(itertools.combinations(item_set, length - 1)):
            if length == 2:
                item = list(item) ## 비교를 하기 위해 뽑은 조합을 리스트로 변경
            else:
                item = set(item)

            ## 하나라도 없으면 break
            if item not in previous_frequent_set:
                break
            cnt = cnt + 1
        
        ## 모든 combination이 있다면 frequent set에 넣는다
        if cnt == length:
            ## 다시 딕셔너리 키로 사용하기 위해 tuple로 변환해서 넣어준다
            frequent_set[tuple(item_set)] = 0 
    
    ## k+1 frequent set을 DB Scan을 통해 count한다
    for key in frequent_set.keys():
        for trx in trxes:
            if set(key) <= set(trx): 
                frequent_set[key] = frequent_set[key] + 1
    
    ## 마지막으로 minimum suppport로 가지치기
    return filter_by_min_sup(frequent_set)

def change_element_to_set(container):
    return_list = list()
    for item in container:
        return_list.append(set(item))
    return return_list

def filter_by_min_sup(candidate):
    global trxes
    ## 매번 min_sup을 비교할 때마다 value/len(trxes) 하는 것보다 min_sup를 count로 바꿔주면 효율적
    min_sup_cnt = min_sup * len(trxes) 
    ## for문 와중에 딕셔너리 사이즈가 변경 되면 안 되기에 삭제를 하지 못함 -> dict 안의 sub-dict을 뽑아냄
    frequent_set = {key: candidate[key] for key in candidate.keys() if candidate[key] >= min_sup_cnt}
    
    ## min_sup을 만족하는 candidate가 하나라도 없으면 exit
    if len(frequent_set) < 1: 
        exit() 
    else:
        return frequent_set

# FSG는 1,2에 대한 서브그래프를 초기에 생성한다
def generate_first_frequent_set():
    global trxes
    item_set = dict()
    for trx in trxes:
        for item in trx:
            if item not in item_set.keys(): 
                item_set[item] = 1
            else:
                item_set[item] = item_set[item] + 1 
    filterItemSet = dict()
    filterItemSet = filter_by_min_sup(item_set)

    previous_frequent_set = list(filterItemSet.keys())
    ## 이 키들을 가지고만 self-join
    candidate = self_join(1, previous_frequent_set)

    ## self join으로 뽑힌 candidate를 pruning 한다
    candidate = prune(1, previous_frequent_set, candidate)

    ## prune이 끝난 candidate를 마지막으로 association rule에 적용시킨다
    apply_association_rule(1, candidate)
    return candidate

def load_data(): 
    global trxes
    ## sys.argv[]로 인수 넣어주면 자동으로 ' ' 인식함
    with open(sys.argv[2], 'r') as f: 
        input_data = f.read().split('\n')
        trx_id = 1
        for trx in input_data: 
            trx = trx.split('\t')
            trxes.append(trx) 
            print(str(trx_id) + ': ', trx) # 주석처리 되어있던 줄
            trx_id += 1

# min_sup이 클수록 탐색속도가 빠르고 결과값이 적게 나온다

# 이 메인 자체가 fsg 알고리즘의 메인이 된다. 따라서 상기에 있는 D, Sigma를 매개변수로 하는 함수는 필요없다.
if __name__ == '__main__':
    ## arguments 리스트로 저장
    argv = sys.argv 
    min_sup = float(sys.argv[1])/100
    output = sys.argv[3]

    ## 데이터 로딩
    load_data() 
    ## frequent_set들을 담을 리스트를 만듬
    frequentCompleteset = ['',]
    frequentCompleteset.append(generate_first_frequent_set())
    
    length = 3
    bRun = True

    while bRun:
        ## k-frequentset의 키들만 뽑아낸다
        previous_frequent_set = list(frequentCompleteset[length - 1].keys())

        ## 이 키를 가지고 fsg-gen 실행
        candidate = fsg_gen(previous_frequent_set)
        
        frequent_set = dict()
        previous_frequent_set = change_element_to_set(previous_frequent_set)

        for item_set in candidate: 
            cnt = 0
            for item in list(itertools.combinations(item_set, length - 1)):
                item = set(item)

                ## 하나라도 없으면 break
                if item not in previous_frequent_set:
                    break
                cnt = cnt + 1
            
            ## 모든 combination이 있다면 frequent set에 넣는다
            if cnt == length:
                ## 다시 딕셔너리 키로 사용하기 위해 tuple로 변환해서 넣어준다
                frequent_set[tuple(item_set)] = 0 
    
        ## k+1 frequent set을 DB Scan을 통해 count한다
        for key in frequent_set.keys():
            for trx in trxes:
                if set(key) <= set(trx): 
                    frequent_set[key] = frequent_set[key] + 1
        
        ## 마지막으로 minimum suppport로 가지치기
        candidate = filter_by_min_sup(frequent_set)

        ## 더 이상 후보를 generate 못하면 exit
        if candidate == -1 | len(candidate) == 0: 
            bRun = false
        else:
        ## frequest_set의 리스트에 추가하고 다음 candidate를 위해 길이를 1 증가
            frequentCompleteset.append(candidate)
            length = length + 1
import sys, getopt
import itertools
from pprint import pprint


'''
데이터 불러오기
input.txt 파일의 데이터를 불러와서 하나의 transaction으로 구분해서 list에 넣어 저장
'''
trxes = list()
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
    
'''
1-frequentset generate
처음으로 DB 돌면서 count하면서 dict에 저장
'''
def generate_first_frequent_set():
    global trxes
    item_set = dict()
    for trx in trxes:
        for item in trx:
            if item not in item_set.keys(): 
                item_set[item] = 1
            else:
                item_set[item] = item_set[item] + 1 

    return filter_by_min_sup(item_set)
   
'''
Minimum support 기준에 만족하지 않는 데이터 가지치기
- 인자로는 self-join과 pruning이 끝난 candidate를 넣어준다
'''
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
        

'''
prune 과정에서 하나의 item과 transaction들을 비교할 때 
set 자료형이 유용해서 리스트 안의 리스트 요소들을 모두 set으로 바꿔줌 
- 인자로는 이중 리스트 자료형이 들어감
'''
def change_element_to_set(container):
    return_list = list()
    for item in container:
        return_list.append(set(item))
    return return_list

'''
Self-Joining: k-frequentset을 통해 k+1-candidate를 만드는 과정
itertools의 combinations 함수를 통해 조합을 함
- 인자로는 k-frequentset과 k+1 길이를 나타내는 length가 들어감 
'''
def self_join(length, previous_frequent_set):
    identical_items = list()
    ## 1-frequentset일 때는 바로 combination 가능
    if length == 2: 
        ## combination은 튜플을 반환하기 때문에 list로 바꿔준다
        return change_element_to_set(list(itertools.combinations(previous_frequent_set, length)))
    
    else:
        ## combination 하기 전에 identical한 길이가 1인 후보들을 뽑는다
        for item_set in previous_frequent_set: 
            for item in item_set:
                if item not in identical_items: 
                    identical_items.append(item)
        ## 길이가 1인 후보들을 가지고 combination을 하고 리스트로 감싼다
        candidate_tuple_inside = list(itertools.combinations(identical_items, length))
        candidate = change_element_to_set(candidate_tuple_inside)

        return candidate

'''
Pruning: Self-join으로 만들어진 K+1-candidate와 이전 k-frequentset이랑 비교하여 
downward closure property로 가지치기 -> 그 후, min Support로 가지치기
'''
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

'''
frequent set을 대상으로 association rule을 적용한다
- support count: A와 B가 전체 trx에서 같이 있을 확률
- confidence count: A->B: A가 있을 때 B가 있을 확률
'''
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
            
            
'''
결과를 output.txt로 저장한다
'''
def save_result(line):
    with open(sys.argv[3], 'a') as f:    
            f.write(line)


if __name__ == '__main__':
    print("Number of arguments: ", len(sys.argv), 'arguments.')
    print("Argument List: ", sys.argv, "\n\n")

    ## arguments 리스트로 저장
    argv = sys.argv 
    min_sup = float(sys.argv[1])/100
    output = sys.argv[3]

    ## 데이터 로딩
    load_data() 
    ## frequent_set들을 담을 리스트를 만듬
    frequent_set = ['',]
    frequent_set.append(generate_first_frequent_set())
    
    length = 2

    while True:
        ## k-frequentset의 키들만 뽑아낸다
        previous_frequent_set = list(frequent_set[length - 1].keys())
        ## 이 키들을 가지고만 self-join
        candidate = self_join(length, previous_frequent_set)
        
        ## 더 이상 candidate 만들어지지 않을 때, 
        if len(candidate) == 0: 
            exit()
        
        ## self join으로 뽑힌 candidate를 pruning 한다
        candidate = prune(length, previous_frequent_set, candidate)

        ## prune이 끝난 candidate를 마지막으로 association rule에 적용시킨다
        apply_association_rule(length, candidate)
        ## 더 이상 후보를 generate 못하면 exit
        if candidate == -1: 
            exit()
        else:
        ## frequest_set의 리스트에 추가하고 다음 candidate를 위해 길이를 1 증가
            frequent_set.append(candidate)
            length = length + 1
        
            


import sys
import os
import argparse

import openpyxl as xl


def maxMemory(batch):   #Byte
    if batch==4 :
        return ( 2762739712 / (2**20) * 1000)

    elif batch==8 :
        return ( 3490525184 / (2**20) * 1000)

    elif batch==16 :
        return ( 4938428928 / (2**20) * 1000)

    elif batch==32 :
        return ( 7842853888 / (2**20) * 1000)


def endEpochTime(batch):    #ms
    if batch==4:
        return 323.3348935842514
    elif batch==8:
        return 308.8732838630676
    elif batch==16:
        return 241.9559121131897
    elif batch==32:
        return 226.55188739299774


def getMemory(batch):
    filename = f"batch_{batch}_mem.txt"
    f = open(filename, "r")
    f_list = f.readlines()

    mem_list = [0,]

    for i, l in enumerate(f_list) :
        mem_list.append(int(float(l.strip())*1000))

    return mem_list


def getTime(batch):
    filename = f"batch_{batch}_time.txt"
    f = open(filename, "r")
    f_list = f.readlines()

    time_list = [0.0,]

    for i, l in enumerate(f_list) :
        time_list.append(float(l.strip()))

    return time_list

# for max_time
def dp(target_mem, W, wt, val, n, factor):  # W: peak mem - budget - max_layer_mem , wt: memory_list, val: time_list, n: layer_num
    
    global K
    global layers
    global mem
    global exceed
    global target_n
    global target_W

    K = [[0.0 for x in range(W+1)] for x in range(n+1)]  # DP를 위한 2차원 리스트 초기화
    layers = [ [ [] for x in range(W+1) ] for x in range(n+1) ]

    # make new list for memory and as hit the budget break the loop
    mem = [[0.0 for x in range(W+1)] for x in range(n+1)]
    exceed = [[0 for x in range(W+1)] for x in range(n+1)]

    target_n =n
    target_W =W    
    max_time =0

    for i in range(n+1):
        
        for w in range(W+1):  # 각 칸을 돌면서

            if i==0 or w==0:  # 0번째 행/열은 0으로 세팅
                K[i][w] = 0
            elif wt[i-1] <= w:  # 점화식을 그대로 프로그램으로
                if exceed[i-1][w-wt[i-1]] == 0 :
                    if (factor * val[i-1]) + K[i-1][w-wt[i-1]] > K[i-1][w]:
                        K[i][w]  = (factor * val[i-1]) + K[i-1][w-wt[i-1]]
                        mem[i][w] = wt[i-1] + mem[i-1][w-wt[i-1]]
                        layers[i][w].extend( layers[i-1][w-wt[i-1]] )
                        layers[i][w].append( i-1 ) # layer_num

                    else:
                        K[i][w] = K[i-1][w]
                        mem[i][w] = mem[i-1][w]
                      
                        layers[i][w].extend( layers[i-1][w] )


                elif exceed[i-1][w-wt[i-1]] == 1 :
                    if K[i][w-1] > K[i-1][w]:
                        K[i][w] = K[i][w-1]
                        mem[i][w] = mem[i][w-1]

                        layers[i][w].extend( layers[i][w-1])

                    else:
                        K[i][w] = K[i-1][w]
                        mem[i][w] = mem[i-1][w]
                            #exceed[i][w] = exceed[i-1][w]
                        layers[i][w].extend( layers[i-1][w] )
                
            else:
                K[i][w] = K[i-1][w]
                mem[i][w] = mem[i-1][w]
                #exceed[i][w] = exceed[i-1][w]

                layers[i][w].extend( layers[i-1][w] )

            if mem[i][w] >= target_mem:
                exceed[i][w] = 1
                
                if max_time<K[i][w]:
                    target_n = i
                    target_W = w
                    max_time = K[i][w]            

#
#    max_time = 0

#    for i in range(n+1):
#        for w in range(int(target_mem), W+1):
#            if exceed[i][w]==1 and max_time < K[i][w]:
#                target_n = i
#                target_W = w
#                max_time = K[i][w]
                
    
    print("n, W : ", target_n, target_W) 
    print("target_mem, mem : " ,target_mem, mem[target_n][target_W])
    print("layers : ",  layers[target_n][target_W])
    
    return K[target_n][target_W]

if __name__ == "__main__":
    
    # memory budget from 1GB to 5GB
    # batch size 8, 16, 32, 64

    
    given_budget = int(sys.argv[1])
   
    batch = [ 4, 8, 16, 32 ]

    for budget in range(given_budget,given_budget+1):
        print (f"Memory Budget = {budget}GiB")
        
        # GiB to Byte
        budget = budget * (2**30) / (2**20) *1000
        
        for i, batch_size in enumerate(batch):
            
            if i == 0:
                fac = 8
            elif i == 1:
                fac = 4
            elif i == 2:
                fac = 2
            else:
                fac = 1
            
                       
            peak_mem = maxMemory(batch_size)
            end_time = endEpochTime(batch_size)
            
            print(peak_mem, budget)
            if peak_mem <= budget:
                continue
       
            mem_list = getMemory(batch_size)
            time_list = getTime(batch_size)

            max_layer_mem = max(mem_list)

            # how to manage target_mem???
            target_mem = peak_mem - budget
            range_mem = int(peak_mem - budget + max_layer_mem)
            
            
            result_time = dp (target_mem, range_mem, mem_list, time_list, len(time_list), fac)
            
            max_time = end_time + result_time

            print("peak_mem : ", peak_mem)
            print("target_mem : ", target_mem)
            print("cal_mem : ", mem[target_n][target_W])


            print("max mem : ", (peak_mem - mem[target_n][target_W])/1000 / (2**10))
            print("max time : ", max_time)

    
    #wb.save(f'simulator_results_{sys.argv[1]}GB.xlsx')
    

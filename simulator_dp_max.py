import sys
import os
import argparse

import openpyxl as xl
from configparser import ConfigParser, ExtendedInterpolation

def maxMemory(batch):   #Byte
    if batch==8 :
        return ( 3055843840 / (2**20) *1000 )

    elif batch==16 :
        return ( 4070816768 / (2**20) *1000)

    elif batch==32 :
        return ( 6108790272 / (2**20) *1000)

    elif batch==64 :
        return ( 10188734976 / (2**20) *1000)


def endEpochTime(batch):    #ms
    if batch==8:
        return 543.2973835
    elif batch==16:
        return 413.875944
    elif batch==32:
        return 384.257897
    elif batch==64:
        return 372.033123


def getMemory(model, batch):
    filename = f"{model}_{batch}_layer_parsed_memory.txt"
    f = open(filename, "r")
    f_list = f.readlines()

    mem_list = [0,]

    for i, l in enumerate(f_list) :
        mem_list.append(int(float(l.strip())*1000))

    return mem_list


def getTime(model, batch):
    filename = f"{model}_{batch}_layer_avg_time.txt"
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

                
    
    print("layers : ",  layers[target_n][target_W])
    
    return K[target_n][target_W]

if __name__ == "__main__":
    
    # memory budget from 1GB to 5GB
    # batch size 8, 16, 32, 64

    
    given_budget = int(sys.argv[1])
   
    #batch = [ 8, 16, 32, 64 ]

    for budget in range(given_budget,given_budget+1):
        print (f"Memory Budget = {budget}GiB")
        
        # GiB to Byte
        budget = budget * (2**30) / (2**20) *1000
        
        for i in range(1,5):
            parser = ConfigParser(interpolation=ExtendedInterpolation())
            parser.read(f'conf_{i}.ini')

            model = parser.get('setting','model')
            dataset = parser.get('setting','dataset')
            batch_size = parser.get('setting','batch_size')
            peak_mem = parser.get('memory','peak_memory')
            end_time = parser.get('time','epoch_time')
            

            
            if i == 1:
                fac = 8
            elif i == 2:
                fac = 4
            elif i == 3:
                fac = 2
            else:
                fac = 1
            
            # B to MB
            peak_mem = int(peak_mem) / (2**20) * 1000
            # epoch time
            end_time = float(end_time)
            
                       
            print("\nbatch size : ", batch_size) 
            print("peak_mem, budget : ",peak_mem, budget)

            if peak_mem <= budget:
                print("skip batch : ", batch_size)
                continue
       
            mem_list = getMemory(model, batch_size)
            time_list = getTime(model, batch_size)

            if peak_mem - sum(mem_list) > budget:
                print("skip batch : ", batch_size)
                continue


            max_layer_mem = max(mem_list)

            # how to manage target_mem???
            target_mem = peak_mem - budget
            range_mem = int(peak_mem - budget + max_layer_mem)
            
            
            result_time = dp (target_mem, range_mem, mem_list, time_list, len(time_list), fac)
            
            max_time = end_time + result_time

            print("max mem : ", (peak_mem - mem[target_n][target_W])/1000 / (2**10))
            print("max time : ", max_time)

    
    #wb.save(f'simulator_results_{sys.argv[1]}GB.xlsx')
    

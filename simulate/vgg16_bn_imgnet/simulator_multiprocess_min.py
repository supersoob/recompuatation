import sys
import os
import argparse
from multiprocessing import Pool
import openpyxl as xl


min_time = 10000000000
#max_time = 0
min_time_lst = [0,0,0,0]
#max_time_lst = [0,0,0,0]
avg_time = 0
avg_mem = 0
cnt = 0


def maxMemory(batch):   #Byte
    if batch==4 :
        return 2762739712

    elif batch==8 :
        return 3490525184

    elif batch==16 :
        return 4938428928

    elif batch==32 :
        return 7842853888


def endEpochTime(batch):    #ms
    if batch==4:
        return (323.3348935842514 / 8)
    elif batch==8:
        return (308.8732838630676 / 4)
    elif batch==16:
        return (241.9559121131897 / 2)
    elif batch==32:
        return 226.55188739299774



def getMemory(batch):
    filename = f"batch_{batch}_mem.txt"
    f = open(filename, "r")
    f_list = f.readlines()

    mem_list = []

    for i, l in enumerate(f_list) :
        mem_list.append(float(l.strip()) * (2**20))

    return mem_list


def getTime(batch):
    filename = f"batch_{batch}_time.txt"
    f = open(filename, "r")
    f_list = f.readlines()

    time_list = []

    for i, l in enumerate(f_list) :
        time_list.append(float(l.strip()))

    return time_list


def greedy(budget, batch, num, cur_mem, cur_time, memdict, timedict, check_lst):
    global min_time
    #global max_time
    global min_time_lst
    #global max_time_lst
    global avg_time
    global avg_mem
    global cnt
    
    #print(num)
    cur_mem = cur_mem - memdict[num]
    cur_time = cur_time + timedict[num]

    if cur_time > min_time:
        return 0

    #print(num,cur_mem,cur_time)
    
    if cur_mem <= budget:
        comb=[]

        if batch == 8:
            ite = 8
        elif batch == 16:
            ite = 4
        elif batch == 32:
            ite = 2
        elif batch == 64:
            ite = 1

        
        for i in range(len(check_lst)):
            if check_lst [i] == 1:
                comb.append(i+1)

        if min_time > cur_time:
            min_time = cur_time
            min_time_lst = [batch, cur_mem / (2**30), cur_time*ite, comb]

       # if max_time < cur_time:
       #     max_time = cur_time
       #     max_time_lst = [batch, cur_mem / (2**30), cur_time*ite, comb]

        avg_time += (cur_time*ite)
        avg_mem += cur_mem / (2**30)
        cnt = cnt + 1

        #print(cur_mem, cur_time, comb)        


        """
        print("\nresidual in comb. : ", comb)

        print("new memory usage : ", cur_mem)
        print("new end-to-end time :", cur_time)
        """

    else:
        for i in range(num+1, 44):
            check_lst [i] = 1
            greedy(budget, batch, i, cur_mem, cur_time, memdict, timedict, check_lst)
            check_lst [i] = 0

def get_max_min(budget, batch_size, num, new_mem, new_time, memdict, timedict, check_lst):
    
    global min_time
    #global max_time
    global min_time_lst
    #global max_time_lst
    global avg_time
    global avg_mem
    global cnt


    #try:
        #print ("exist")
    #    wb = xl.load_workbook(f'simulator_results_{sys.argv[1]}GB.xlsx')
   # except:
        #print("new")
    #    wb = xl.Workbook()


    #sheet = wb.active
    #sheet.append(["budget","batch","min_time","min_mem","min_res","max_time","max_mem","max_res","avg_time","avg_memory"])

    check_lst[num] = 1
    greedy(budget, batch_size, num, new_mem, new_time, memdict, timedict, check_lst)
    check_lst[num] = 0

    if cnt>0:
        avg_time = avg_time / cnt
        avg_mem = avg_mem / cnt    

    #sheet.append([ budget, batch_size, min_time_lst[2], min_time_lst[1], str(min_time_lst[3]), max_time_lst[2], max_time_lst[1], str(max_time_lst[3]), avg_time, avg_mem ])

    #print("\nbudget : ", budget)
    #print("cur_batch : ", batch_size)
    print("minTime : ", min_time_lst)
    #print("maxTime : ", max_time_lst)
    #print("avgTime : ", avg_time)
    #print("avgMem : ", avg_mem)
    #print("cnt : ", cnt)
    
    #wb.save(f'simulator_results_{sys.argv[1]}GB.xlsx')
    

if __name__ == "__main__":
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()

    # GB to B
    budget = args.budget * (2**30)
    batch_size = args.batch

    """
    # memory budget from 1GB to 5GB
    # batch size 256, 512, 1024, 2048

    
    
    given_budget = int(sys.argv[1])
    batch = [ 4, 8, 16, 32 ]

    print (f"Memory Budget = {given_budget}GiB")
    
    # GiB to Byte
    budget = given_budget * (2**30)
    
    for batch_size in batch:    
        
        peak_mem = maxMemory(batch_size)
        end_time = endEpochTime(batch_size)
        if peak_mem <= budget:
            continue

        #print("batch_size = ",batch_size)
        
        memdict = getMemory(batch_size)
        timedict = getTime(batch_size)

        new_mem = peak_mem
        new_time = end_time

        check_lst = [ 0 for _ in range(0,44) ]


        mem_sum = 0
        init_idx = len(memdict)
        for i in reversed(range(44)):
            mem_sum += memdict[i]
            if (mem_sum >= (new_mem - budget)):
                init_idx = i
                break

        with Pool(processes=init_idx) as pool:
            pool.starmap(get_max_min, [(budget, batch_size, num, new_mem, new_time, memdict, timedict, check_lst) for num in range(init_idx)] )

        
        min_time = 10000000000
        #max_time = 0
        min_time_lst = [0,0,0,0]
        #max_time_lst = [0,0,0,0]
        avg_time = 0
        avg_mem = 0
        cnt = 0
                
        """
        for num, mem in enumerate(memdict):
            #check_lst[num] = 1
            greedy(budget, batch_size, num, new_mem, new_time, memdict, timedict, check_lst)
            #check_lst[num] = 0
        """
    

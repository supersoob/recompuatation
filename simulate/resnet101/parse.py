import os
import sys
import openpyxl as xl

f = open(f"simulate_{sys.argv[1]}GB.txt","r")

lines = f.readlines()

wb = xl.Workbook()

#sheet = wb.active

budget = 0
cur_batch = 0

minTime = 1000000.0
minMem = 10000000000.0

maxTime = 0
maxMem = 0

avgMem = 0
avgTime =0

minMemList = [0,0,0,0]
minTimeList = [0,0,0,0]

maxMemList = [0,0,0,0]
maxTimeList = [0,0,0,0]
cnt = 0


for l in lines:
    if "batch" in l:

        if cnt>0:        
            avgTime = avgTime / cnt
            avgMem = avgMem / cnt

        print("budget : ", budget)
        print("cur_batch : ",cur_batch)
        print("minTime : ",minTimeList)
        print("maxTime : ",maxTimeList)
        #print("minMem : ", minMemList)
        #print("maxMem : ", maxMemList)
        print("avgTime : ", avgTime)
        print("avgMem : ", avgMem)
        print("cnt : ", cnt)

        
        minTime = 1000000.0
        minMem = 10000000000.0
        maxTime = 0
        maxMem = 0
        avgTime = 0
        avgMem = 0
        cnt = 0
        cur_batch = int(l.split(" = ")[1])
        
        minMemList = [0,0,0,0]
        minTimeList = [0,0,0,0]
        maxMemList = [0,0,0,0]
        maxTimeList = [0,0,0,0]


    elif "Budget" in l:
        budget = int(l.split(" = ")[1].split("G")[0])


    else:
        if "done" in l:
            continue

        cnt = cnt + 1
        mem = float(l.split("[")[0].split(" ")[0])
        time = float(l.split("[")[0].split(" ")[1])
        comb = "["+l.split("[")[1]

        avgTime += (time)
        avgMem += (mem / (2**30))

        if minMem > mem:
            minMem = mem
            minMemList = [cur_batch, mem / (2**30),time,comb.strip()]
        
        if minTime > time:
            minTime = time
            minTimeList = [cur_batch,mem / (2**30) ,time,comb.strip()]
            
        if maxMem < mem:
            maxMem = mem
            maxMemList = [cur_batch, mem / (2**30), time, comb.strip()]

        if maxTime < time:
            maxTime = time
            maxTimeList = [cur_batch,mem / (2**30),time,comb.strip()]


avgTime = avgTime / cnt
avgMem = avgMem / cnt

print("budget : ", budget)
print("cur_batch : ",cur_batch)
print("minTime : ",minTimeList)
print("maxTime : ",maxTimeList)
#print("minMem : ", minMemList)
#print("maxMem : ", maxMemList)
print("avgTime : ", avgTime)
print("avgMem : ", avgMem)
print("cnt : ", cnt)



"""
for l in lines:
    
    if "batch" in l:
        batch_size = l.split(" = ")[1]
        sheet = wb.create_sheet()
        sheet.title = f"{budget}GB_{batch_size}_batch"
        sheet.append(["memory","time","comb"])

    
    elif "Budget" in l:
        budget = l.split(" = ")[1].split("G")[0]
    
    else:
        #print(l)
        mem = l.split("[")[0].split(" ")[0]
        time = l.split("[")[0].split(" ")[1]
        comb = "["+l.split("[")[1]
        #print(comb)

        ap = [mem,time,comb]

        sheet.append(ap)
"""
#wb.save(f"simulate_{sys.argv[1]}.xlsx")

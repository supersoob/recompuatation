import os
import sys

if __name__=="__main__":
    tfname = f"{sys.argv[1]}_layer_avg_time.txt"
    mfname = f"{sys.argv[1]}_layer_parsed_memory.txt"
    dfname = f"{sys.argv[1]}_data_generation_rate.txt"

    timef = open(tfname, "r").readlines()
    memoryf = open(mfname, "r").readlines()

    leng = len(timef)

    dgenf = open(dfname,"w")

    for i in range(leng):
        time = float(timef[i].strip())
        mem = float(memoryf[i].strip())

        dgen = "%.10f\n" % (mem / time)
        dgenf.write(dgen)
    
    dgenf.close()

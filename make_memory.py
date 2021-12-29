import openpyxl
import os
import sys

cnt = 0
layer_cnt = 0
layers = [ 0 for _ in range(31)]

if __name__ == "__main__":
    filename = sys.argv[1]
    f = open(filename, "r")
    lines = f.readlines()

    for i, l in enumerate(lines):
        if "Epoch" in l:
            break
        if "Current" in l:
            continue
        print(l.strip())
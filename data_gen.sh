#!/bin/bash

model="inception"
dataset="imgnet"
batch_size=(4 8 16 32)
epoch_time_size=32
cnt=0


for batch in ${batch_size[@]}; do
    (( cnt = "${cnt}" + 1 ))
    
    python train.py -bs ${batch} -bn ${cnt} -m ${model} -v memory &> ${model}_${batch}_layer_memory.txt
    python make_memory.py ${model}_${batch}_layer_memory.txt &> ${model}_${batch}_layer_parsed_memory.txt
    rm ${model}_${batch}_layer_memory.txt
    python train.py -bs ${batch} -bn ${cnt} -m ${model} -v time &> ${model}_${batch}_layer_time.txt
    python make_time.py ${model}_${batch}_layer_time.txt &> ${model}_${batch}_layer_avg_time.txt
    rm ${model}_${batch}_layer_time.txt
    python make_data_gen.py ${model}_${batch}
    python train_peakmem.py -bs ${batch} -bn ${cnt} -m ${model}
    python train_epochtime.py -bs ${batch} -bn ${cnt} -m ${model} -et ${epoch_time_size}
done

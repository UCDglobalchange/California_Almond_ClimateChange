#!/bin/bash
home_path=

model_name_list=('ACCESS-CM2' 'EC-Earth3' 'EC-Earth3-Veg' 'GFDL-ESM4' 'INM-CM5-0' 'MPI-ESM1-2-HR' 'MRI-ESM2-0' 'CNRM-ESM2-1')
for model in "${model_name_list[@]}"
do
    sed -i "s/model=.*/model='${model}'/g" $home_path/LOCA_create_nc.py
    sleep 30s
    python $home_path/LOCA_create_nc.py
    sleep 2m
done


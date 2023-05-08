#!/bin/bash
home_path=''

model_name_list=('bcc-csm1-1' 'bcc-csm1-1-m' 'BNU-ESM' 'CanESM2' 'CSIRO-Mk3-6-0' 'GFDL-ESM2G' 'GFDL-ESM2M' 'inmcm4' 'IPSL-CM5A-LR' 'IPSL-CM5A-MR' 'CNRM-CM5' 'HadGEM2-CC365' 'HadGEM2-ES365' 'IPSL-CM5B-LR' 'MIROC5' 'MIROC-ESM' 'MIROC-ESM-CHEM')
for model in "${model_name_list[@]}"
do
    sed -i "s/model=.*/model='${model}'/g" $home_path/MACA_create_nc.py
    sleep 30s
    python $home_path/MACA_create_nc.py
    sleep 2m
done


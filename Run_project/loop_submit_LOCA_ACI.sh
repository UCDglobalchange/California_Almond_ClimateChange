#!/bin/bash
home_path=
model_name_list=('ACCESS-CM2' 'EC-Earth3' 'EC-Earth3-Veg' 'GFDL-ESM4' 'INM-CM5-0' 'MPI-ESM1-2-HR' 'MRI-ESM2-0' 'CNRM-ESM2-1')
for model in "${model_name_list[@]}"
do 
    mkdir $home_path/intermediate_data/LOCA_ACI/$model/
    sed -i "s/-J.*/-J '${model}'/g" $home_path/submit_LOCA_ACI.sh
    sbatch $home_path/submit_LOCA_ACI.sh $model
    sleep 5s
done

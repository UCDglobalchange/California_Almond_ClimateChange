#!/bin/bash
home_path=/home/shqwu/California_Almond_ClimateChange-main/Run_project
model_name_list=('MRI-CGCM3' 'bcc-csm1-1' 'bcc-csm1-1-m' 'BNU-ESM' 'CanESM2' 'CSIRO-Mk3-6-0' 'GFDL-ESM2G' 'GFDL-ESM2M' 'inmcm4' 'IPSL-CM5A-LR' 'IPSL-CM5A-MR' 'CNRM-CM5' 'HadGEM2-CC365' 'HadGEM2-ES365' 'IPSL-CM5B-LR' 'MIROC5' 'MIROC-ESM' 'MIROC-ESM-CHEM')

for model in "${model_name_list[@]}"
do 
    mkdir $home_path/intermediate_data/MACA_ACI/$model/
    sed -i "s/-J.*/-J '${model}'/g" $home_path/submit_MACA_ACI.sh
    sbatch submit_MACA_ACI.sh $model
    sleep 5s
done

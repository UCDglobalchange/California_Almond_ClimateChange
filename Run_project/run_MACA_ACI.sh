#!/bin/bash
data_ID='11_19'
model_name_list=('MRI-CGCM3' 'bcc-csm1-1' 'bcc-csm1-1-m' 'BNU-ESM' 'CanESM2' 'CSIRO-Mk3-6-0' 'GFDL-ESM2G' 'GFDL-ESM2M' 'inmcm4' 'IPSL-CM5A-LR' 'IPSL-CM5A-MR' 'CNRM-CM5' 'HadGEM2-CC365' 'HadGEM2-ES365' 'IPSL-CM5B-LR' 'MIROC5' 'MIROC-ESM' 'MIROC-ESM-CHEM')
#model_list = ('MRI-CGCM3_r1i1p1', 'bcc-csm1-1_r1i1p1','bcc-csm1-1-m_r1i1p1','BNU-ESM_r1i1p1','CanESM2_r1i1p1', 'CSIRO-Mk3-6-0_r1i1p1', 'GFDL-ESM2G_r1i1p1', 'GFDL-ESM2M_r1i1p1', 'inmcm4_r1i1p1', 'IPSL-CM5A-LR_r1i1p1', 'IPSL-CM5A-MR_r1i1p1', 'CNRM-CM5_r1i1p1', 'HadGEM2-CC365_r1i1p1','HadGEM2-ES365_r1i1p1','IPSL-CM5B-LR_r1i1p1', 'MIROC5_r1i1p1', 'MIROC-ESM_r1i1p1', 'MIROC-ESM-CHEM_r1i1p1' )
for model in "${model_name_list[@]}"
do
    mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID/MACA_ACI/$model/
    sed -i "s/model_name=.*/model_name='${model}'/g" /home/shqwu/Almond_code_git/MACA_ACI.py
    sed -i "s/-J.*/-J '${model}'/g" /home/shqwu/Almond_code_git/submit_MACA_ACI.sh
    sleep 30s
    sbatch /home/shqwu/Almond_code_git/submit_MACA_ACI.sh
    sleep 1m
done


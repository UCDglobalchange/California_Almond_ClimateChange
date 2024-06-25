#!/bin/bash

home_path=''

mkdir $home_path/intermediate_data
mkdir $home_path/input_data
mkdir $home_path/input_data/GridMet
mkdir $home_path/input_data/MACA
mkdir $home_path/input_data/LOCA
mkdir $home_path/input_data/reference_cropland
mkdir $home_path/output_data
mkdir $home_path/intermediate_data/Gridmet_ACI
mkdir $home_path/intermediate_data/Gridmet_nc
mkdir $home_path/intermediate_data/Gridmet_csv
mkdir $home_path/intermediate_data/MACA_ACI
mkdir $home_path/intermediate_data/LOCA_ACI
mkdir $home_path/intermediate_data/lasso_model
mkdir $home_path/intermediate_data/MACA_nc
mkdir $home_path/intermediate_data/MACA_csv
mkdir $home_path/intermediate_data/LOCA_nc
mkdir $home_path/intermediate_data/LOCA_csv
mkdir $home_path/output_data/projection
mkdir $home_path/output_data/projection/MACA
mkdir $home_path/output_data/projection/LOCA
mkdir $home_path/output_data/aci_contribution
mkdir $home_path/output_data/aci_contribution/MACA
mkdir $home_path/output_data/aci_contribution/LOCA
mkdir $home_path/output_data/plots
mkdir $home_path/output_data/plots/MACA
mkdir $home_path/output_data/plots/LOCA
mkdir $home_path/slurm_log

sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/MACA_ACI.py
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/LOCA_ACI.py
sed -i "s,home_path=.*,home_path=$home_path,g" $home_path/loop_submit_MACA_ACI.sh
sed -i "s,home_path=.*,home_path=$home_path,g" $home_path/loop_submit_LOCA_ACI.sh

sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/run_MACA_create_nc.sh
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/run_LOCA_create_nc.sh

sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/mask_cropland_Gridmet.py
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/mask_cropland_MACA.py
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/mask_cropland_LOCA.py

sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/Almond_lasso.py
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/Gridmet_ACI.py
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/Gridmet_create_csv.py
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/Gridmet_create_nc.py
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/MACA_create_csv.py
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/MACA_create_nc.py
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/MACA_projection.py
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/LOCA_create_csv.py
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/LOCA_create_nc.py
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/LOCA_projection.py
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/aci_contribution.py
sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/plot.py


sed -i "s,-D.*,-D $home_path,g" $home_path/submit_Almond_lasso.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/submit_Gridmet_ACI.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/submit_MACA_ACI.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/submit_MACA_projection.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/submit_LOCA_ACI.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/submit_LOCA_projection.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/submit_aci_contribution.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/submit_mask_Gridmet.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/submit_mask_MACA.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/submit_mask_LOCA.sh

sed -i "s,-e.*,-e $home_path/slurm_log/sterror_%j.txt,g" $home_path/submit_Almond_lasso.sh
sed -i "s,-e.*,-e $home_path/slurm_log/sterror_%j.txt,g" $home_path/submit_Gridmet_ACI.sh
sed -i "s,-e.*,-e $home_path/slurm_log/sterror_%j.txt,g" $home_path/submit_MACA_ACI.sh
sed -i "s,-e.*,-e $home_path/slurm_log/sterror_%j.txt,g" $home_path/submit_LOCA_projection.sh
sed -i "s,-e.*,-e $home_path/slurm_log/sterror_%j.txt,g" $home_path/submit_LOCA_ACI.sh
sed -i "s,-e.*,-e $home_path/slurm_log/sterror_%j.txt,g" $home_path/submit_MACA_projection.sh
sed -i "s,-e.*,-e $home_path/slurm_log/sterror_%j.txt,g" $home_path/submit_aci_contribution.sh
sed -i "s,-e.*,-e $home_path/slurm_log/sterror_%j.txt,g" $home_path/submit_mask_Gridmet.sh
sed -i "s,-e.*,-e $home_path/slurm_log/sterror_%j.txt,g" $home_path/submit_mask_MACA.sh
sed -i "s,-e.*,-e $home_path/slurm_log/sterror_%j.txt,g" $home_path/submit_mask_LOCA.sh

sed -i "s,-o.*,-o $home_path/slurm_log/stdoutput_%j.txt,g" $home_path/submit_Almond_lasso.sh
sed -i "s,-o.*,-o $home_path/slurm_log/stdoutput_%j.txt,g" $home_path/submit_Gridmet_ACI.sh
sed -i "s,-o.*,-o $home_path/slurm_log/stdoutput_%j.txt,g" $home_path/submit_MACA_ACI.sh
sed -i "s,-o.*,-o $home_path/slurm_log/stdoutput_%j.txt,g" $home_path/submit_MACA_projection.sh
sed -i "s,-o.*,-o $home_path/slurm_log/stdoutput_%j.txt,g" $home_path/submit_LOCA_ACI.sh
sed -i "s,-o.*,-o $home_path/slurm_log/stdoutput_%j.txt,g" $home_path/submit_LOCA_projection.sh
sed -i "s,-o.*,-o $home_path/slurm_log/stdoutput_%j.txt,g" $home_path/submit_aci_contribution.sh
sed -i "s,-o.*,-o $home_path/slurm_log/stdoutput_%j.txt,g" $home_path/submit_mask_Gridmet.sh
sed -i "s,-o.*,-o $home_path/slurm_log/stdoutput_%j.txt,g" $home_path/submit_mask_MACA.sh
sed -i "s,-o.*,-o $home_path/slurm_log/stdoutput_%j.txt,g" $home_path/submit_mask_LOCA.sh



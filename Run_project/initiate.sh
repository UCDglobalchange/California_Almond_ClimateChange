#!/bin/bash

home_path=''

mkdir $home_path/intermediate_data
mkdir $home_path/input_data
mkdir $home_path/input_data/GridMet
mkdir $home_path/input_data/MACA
mkdir $home_path/input_data/MACA/reference_cropland
mkdir $home_path/output_data
mkdir $home_path/intermediate_data/Gridmet_ACI
mkdir $home_path/intermediate_data/Gridmet_nc
mkdir $home_path/intermediate_data/Gridmet_csv
mkdir $home_path/intermediate_data/MACA_ACI
mkdir $home_path/intermediate_data/lasso_model
mkdir $home_path/intermediate_data/MACA_nc
mkdir $home_path/intermediate_data/MACA_csv
mkdir $home_path/output_data/projection
mkdir $home_path/output_data/aci_contribution
mkdir $home_path/output_data/plots

sed -i "s,home_path=.*,home_path='$home_path',g" $home_path/MACA_ACI.py
sed -i "s,home_path=.*,home_path=$home_path,g" $home_path/run_MACA_ACI.sh
sed -i "s,home_path=.*,home_path=$home_path,g" $home_path/run_MACA_create_nc.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/submit_Almond_lasso.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/submit_Gridmet_ACI.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/submit_MACA_ACI.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/submit_MACA_projection.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/submit_aci_contribution.sh
sed -i "s,-D.*,-D $home_path,g" $home_path/mask_cropland_MACA.py
sed -i "s,-D.*,-D $home_path,g" $home_path/mask_cropland_Gridmet.py

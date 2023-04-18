#!/bin/bash

data_ID='11_19'

mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID
mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID/Gridmet_ACI
mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID/Gridmet_nc
mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID/Gridmet_csv
mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID/MACA_ACI
mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID/lasso_model
mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID/MACA_nc
mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID/MACA_csv
mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID/MACA_csv/tech_2010
mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID/MACA_csv/to_2020
mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID/projection
mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID/projection/to_2020
mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID/projection/tech_2010
mkdir /home/shqwu/Almond_code_git/saved_data/$data_ID/aci_contribution


sed -i "s/data_ID=.*/data_ID='$data_ID'/g" /home/shqwu/Almond_code_git/Gridmet_ACI.py
sed -i "s/data_ID=.*/data_ID='$data_ID'/g" /home/shqwu/Almond_code_git/Gridmet_create_nc.py
sed -i "s/data_ID=.*/data_ID='$data_ID'/g" /home/shqwu/Almond_code_git/Gridmet_create_csv.py
sed -i "s/data_ID=.*/data_ID='$data_ID'/g" /home/shqwu/Almond_code_git/MACA_ACI.py
sed -i "s/data_ID=.*/data_ID='$data_ID'/g" /home/shqwu/Almond_code_git/MACA_create_csv.py
sed -i "s/data_ID=.*/data_ID='$data_ID'/g" /home/shqwu/Almond_code_git/MACA_create_nc.py
sed -i "s/data_ID=.*/data_ID='$data_ID'/g" /home/shqwu/Almond_code_git/Almond_lasso.py
sed -i "s/data_ID=.*/data_ID='$data_ID'/g" /home/shqwu/Almond_code_git/MACA_projection.py
sed -i "s/data_ID=.*/data_ID='$data_ID'/g" /home/shqwu/Almond_code_git/run_MACA_ACI.sh
sed -i "s/data_ID=.*/data_ID='$data_ID'/g" /home/shqwu/Almond_code_git/aci_contribution.py

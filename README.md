# Climate change impact on almond yield in California

This project aims to build the statistical relationship between histroical Agro-climatic Index (ACIs) calculated from gridMET and county-level almond yields in California via Lasso regression, and project future almond yield based on MACA climate datasets. 

## Data availability
**gridMET:** High-resolution (~4km, 1/24th degree) daily climate data covering contiguous U.S. from 1979 to the present, 
which can be downloaded at: https://www.northwestknowledge.net/metdata/data/

**MACAv2-METDATA:** High-resolution (~4km, 1/24th degree) projected climate data downscaled from GCMs by Multivariate Adaptive Constructed Analogs (MACA)    statistical method and trained by gridMET, covering contiguous U.S. for historical period (1950-2005) and future periods (2006-2099)   under RCP4.5 and RCP8.5, which can be downloaded at https://climate.northwestknowledge.net/MACA/data_portal.php

**California almond harvest data:** Annual county-level agricultural production data provided by the USDA's National Agricultural Statistics Services from 1980 to 2011, which can be downloaded at https://www.nass.usda.gov/Statistics_by_State/California/Publications/AgComm/index.php

**Cropland Data Layer:** geospatial data of Cropland Data Layer(CDL) across contiguous U.S. provided by the USDA, which can be exported from the CropScape web app (https://nassgeodata.gmu.edu/CropScape/)


## Data processing
**Almond cropland mask:** The spatial resolution of CDL (30m)is different from those of climate datasets (4km). To filter out gridcells without almond croplands from gridMET and MACA, we used ArcGIS Pro to aggregrate and re-coordinate CDL raster data to match the resolution and geospatial coordinate of climate datasets. The CDL of almonds in California is available from 2007 to 2022 and the netCDF files produced from ArcGIS Pro are available in folder ***Almond_cropland_nc***

**Calculte ACIs:** Python codes to calculate ACIs from gridMET and MACA are available in folder ***Calculate_ACI***

## Run LASSO regression ##
We used the function *sklearn.linear_model.LassoCV*  from Python package —— scikit-learn. We calibrated the function by providing a list of alphas (penalty parameters), passing intergers (1-1000) to random state for reproducible outputs, and passing fit_intercept to *False*. The code to run LASSO regression is ***Almond_lasso.py***.

## Yield projection ##
After obtaining statistical relationship between gridMET-ACIs and historical county-level yields, we run ***MACA_projection.py*** to project yield based on MACA climate datasets(historical:1950-2005; future:2006-2099 under RCP4.5 and RCP8.5) and compute cropland area-weighted California statewide almond yield. 

## Run the project ##
Due to the enormous size of climate datasets and huge computation power requirement, we recommend to run the project on supercomputing cluster. The folder ***Run_project*** contains python and bash scripts which can be run sequentially to reproduce the results and generate plots in the Run_project/saved_data/. Please read the **Running_instruction.txt**, which can be found in the folder, at first.



# Climate change impact on almond yield in California

This project aims to build the statistical relationship between histroical Agro-climatic Index (ACIs) calculated from gridMET and county-level almond yields in California via Lasso regression, and project future almond yield based on MACA climate datasets. 

## Data availability
**gridMET:** High-resolution (~4km, 1/24th degree) daily climate data covering contiguous U.S. from 1979 to the present, 
which can be downloaded at: https://www.northwestknowledge.net/metdata/data/

**MACAv2-METDATA:** High-resolution (~4km, 1/24th degree) projected climate data downscaled from GCMs by Multivariate Adaptive Constructed Analogs (MACA)    statistical method and trained by gridMET, covering contiguous U.S. for historical period (1950-2005) and future periods (2006-2099)   under RCP4.5 and RCP8.5, which can be downloaded at https://climate.northwestknowledge.net/MACA/data_portal.php

**California almond harvest data:** Annual county-level agricultural production data provided by the USDA's National Agricultural Statistics Services from 1980 to 2011, which can be downloaded at https://www.nass.usda.gov/Statistics_by_State/California/Publications/AgComm/index.php

**Cropland Data Layer:** geospatial data of Cropland Data Layer(CDL) across contiguous U.S. provided by the USDA, which can be exported from the CropScape web app (https://nassgeodata.gmu.edu/CropScape/)

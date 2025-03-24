# Data for Snow Drought to Hydrologic Drought Progression

This directory contains SWE (Snow Water Equivalent)  time series data for all CAMELS basins.

## Basin SWE Data Files

Each CSV file in this directory is named with the basin ID (e.g., `1013500.csv`) and contains the following columns:

- `date`: Date in YYYY-MM-DD format
- `daymet_swe`: Daily SWE value (mm) from Daymet dataset
- `gldas_swe`: Daily SWE value (mm) from GLDAS dataset
- `snodas_swe`: Daily SWE value (mm) from SNODAS dataset
- `uaz_swe`: Daily SWE value (mm) from UAZ dataset
  
## Data Sources

The SWE data in this repository is derived from the following sources:

1. **UAZ**: University of Arizona Daily 4km Gridded SWE (1981-2021)
   - Source: https://doi.org/10.5067/0GGPB220EX6A

2. **Daymet**: Daily 1km gridded SWE (1980-present)
   - Source: https://doi.org/10.3334/ORNLDAAC/2129

3. **SNODAS**: Snow Data Assimilation System (2003-present)
   - Source: https://doi.org/10.7265/N5TB14TC
     
4. **GLDAS**: Global Land Data Assimilation System
   - Source: https://ldas.gsfc.nasa.gov/gldas
     
All data has been preprocessed to calculate basin-averaged SWE values for each CAMELS basin.

## Coverage
- Temporal coverage varies by dataset:
  - Daymet: 1980-2023
  - GLDAS: 1948-2023
  - SNODAS: 2003-2023
  - UAZ: 1981-2021

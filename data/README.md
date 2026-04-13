# Example data (BioTIME)

This folder contains **example datasets** used in the reserBUGS tutorials.

These data are derived from the **BioTIME database**, a global collection of biodiversity time series.

---

## About BioTIME

BioTIME is a curated database of ecological time series describing changes in species abundances through time across marine, terrestrial, and freshwater systems.

More information:
<https://doi.org/10.1111/geb.70003>

---

## Example dataset

This repository includes a small sample dataset:

```text
data/sample/series_4_RESERBUGS_EXAMPLES.csv
```

### Description

This file contains a **processed subset of annual insect abundance time series** extracted from BioTIME.

It is formatted specifically for use with reserBUGS and is used in:

- `Example_biotime.ipynb`

The dataset allows users to run forecasting examples without downloading the full BioTIME database.

---

## Column description

- **STUDY_ID**  
  Identifier of the original study in BioTIME.

- **valid_name**  
  Scientific name of the species.

- **LATITUDE**  
  Latitude (decimal degrees) of the sampling location.

- **LONGITUDE**  
  Longitude (decimal degrees) of the sampling location.

- **ABUNDANCE**  
  Observed abundance (count) of the species.  
  This is the variable used for forecasting.

- **YEAR**  
  Year of observation.

---

## Time series structure

Each time series is defined by a unique combination of:

- `STUDY_ID`
- `valid_name`
- `LATITUDE` / `LONGITUDE`

Within each time series:

- observations are ordered by `YEAR`
- `ABUNDANCE` is the variable to be predicted

---

## Access to full data

The full BioTIME dataset is **not included** in this repository.

It can be accessed through official sources (e.g. Zenodo and project repositories).

---

## Notes

- This is a **small subset for demonstration purposes only**
- It does not represent the full BioTIME dataset
- Abundance values are non-negative integers
- Missing years may occur in time series

---

## Citation

If you use BioTIME data, please cite:

Dornelas, M. et al. (2025).  
*BioTIME 2.0: Expanding and Improving a Database of Biodiversity Time Series.*  
Global Ecology and Biogeography, 34(5), e70003.  
<https://doi.org/10.1111/geb.70003>

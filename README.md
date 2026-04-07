# reserBUGS

![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

reserBUGS is a forecasting tool developed within the ANTENNA research project to support ecological forecasting of pollinator populations, especially insects.

It provides a structured pipeline to predict species abundance from time series data using environmental (abiotic) variables derived from satellite data.

---

## About the ANTENNA Project

This tool has been developed as part of the ANTENNA project (Task 3.2), which focuses on iterative near-term ecological forecasting.

The goal is to:

- Generate short-term forecasts of pollinator populations
- Continuously update predictions as new data become available
- Identify key environmental drivers affecting biodiversity

More information: https://pollinators-antenna.eu/

---

## Authors and Institutions

**Institutions:**

- Universidad Politécnica de Madrid ([UPM](https://www.upm.es/))
- Estación Biológica de Doñana ([EBD-CSIC](https://www.ebd.csic.es/))

**Contributors:**

- [Miguel Ángel Muñoz](https://orcid.org/0000-0001-6114-4460)
- [Alfonso Allen-Perkins](https://orcid.org/0000-0003-3547-2190)
- [Juan Manuel Pastor](https://orcid.org/0000-0002-1067-3642)
- [Javier Galeano](https://orcid.org/0000-0003-0706-4317)
- [Julia G. de Aledo](https://orcid.org/0000-0001-9065-9316)
- [Ignasi Bartomeus](https://orcid.org/0000-0001-7893-4389)

---

## What does reserBUGS do?

reserBUGS allows you to:

- Forecast insect abundance from time series data
- Combine biological data with environmental variables
- Automatically retrieve satellite-derived data (climate, vegetation)
- Evaluate forecast performance

The tool is based on reservoir computing, a modelling approach well suited for ecological time series with delayed and nonlinear effects.

---

## Key Features

- Works with count data (e.g. insect abundance)
- Supports annual, monthly, or daily time series
- Flexible inclusion of lagged variables
- Automatic retrieval of environmental data from:
  - Copernicus (ERA5 / ERA5-Land)
  - MODIS (NDVI)
- Probabilistic forecasting with uncertainty estimation
- Built-in forecast evaluation metrics:
  - Type S error (direction)
  - Type M error (magnitude)
  - CRPS (Continuous Ranked Probability Score)
  - DSS (Dawid–Sebastiani Score)
  - Interval Score

---

## Who is this for?

- Ecologists working with time series data  
- Researchers in biodiversity forecasting  
- Data scientists interested in ecological modelling  
- Students learning forecasting and environmental data analysis

---

## Quick Start (BioTIME-style workflow)

This example follows the same workflow as the `Example_biotime.ipynb` notebook:

- load a time series dataset
- split the series chronologically into training and testing sets
- fit a reservoir computing model
- generate forecasts
- evaluate forecast performance

> **Note:** This example assumes that `reserBUGS` is installed (e.g. `pip install -e .`, see below). If you are working directly from the repository without installation, you may need to add `src/` to your Python path.

```python
import pandas as pd
import numpy as np

from reserbugs.reservoir_computing import ReservoirComputing
from reserbugs.evaluation import (
    type_s_error,
    type_m_error,
    scoring_rules,
)

# ---------------------------------------------------------------------
# Load your time series data
# ---------------------------------------------------------------------
# Expected:
# - one target column (e.g. insect counts)
# - one or more numeric predictor columns
# - optionally, a time column for indexing or plotting
#
# Example columns:
# time | count | temperature | precipitation

data = pd.read_csv("your_timeseries.csv")

X = data[["temperature", "precipitation"]].values
y = data["count"].values

# ---------------------------------------------------------------------
# Train / test split (chronological)
# ---------------------------------------------------------------------
split_idx = int(0.8 * len(y))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ---------------------------------------------------------------------
# Fit reservoir computing model
# ---------------------------------------------------------------------
rc = ReservoirComputing()
rc.fit(X_train, y_train.values)

# ---------------------------------------------------------------------
# Generate forecast paths
# ---------------------------------------------------------------------
preds, stats = rc.sample_paths(
    X_train=X_train,
    Y_train=y_train.values,
    X_test=X_test,
    Y_init=y_train.values[-1:],
    n_lags=1,
    N=100,
)

# ---------------------------------------------------------------------
# Type S error
# ---------------------------------------------------------------------
last_y = float(y_train.iloc[-1])

y_gt_ext = np.concatenate(([last_y], y_test.values))
preds_ext = np.concatenate(
    (np.full((preds.shape[0], 1), last_y), preds),
    axis=1,
)

error_rate, ci_low, ci_high, x_out, n_valid = type_s_error(
    true_value=y_gt_ext,
    estimate=preds_ext,
    baseline="diff",
    return_ci=True,
)

# ---------------------------------------------------------------------
# Type M error
# ---------------------------------------------------------------------
steps, per_step_errs, means = type_m_error(
    estimate=preds,
    true_value=y_test.values,
    threshold=0.1,
    base=10,
)

# ---------------------------------------------------------------------
# Probabilistic scoring rules
# ---------------------------------------------------------------------
scores = scoring_rules(
    true_value=y_test.values[:preds.shape[1]],
    estimate=preds,
    alpha=0.05,
)

print(scores.head())
```

### Notes

- For a full workflow (including visualization and data preparation), see:
`notebooks/Example_biotime.ipynb`
- The example above retrieves monthly environmental data and aggregates it to annual values.
- For daily data retrieval, see:  
  `notebooks/Example_daily_data_retriever.ipynb`
- If you use more than one lag, adapt the example as follows (e.g. 3 lags):

```python
rc.fit(X_train, y_train, n_lags=3)
preds = rc.predict(X_test, Y_init=y_train[-3:], n_lags=3)
```

---

## Repository Structure

```
reserBUGS/
├── pyproject.toml
├── src/
│   └── reserbugs/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── copernicus.py
│       │   └── modis.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── error_evaluation.py
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── visualizations.py
│       └── reservoir_computing/
│           ├── __init__.py
│           └── reservoir_computing.py
│
├── notebooks/
│   ├── Example_biotime.ipynb
│   ├── Example_NDVI.ipynb
│   └── Example_daily_data_retriever.ipynb
│
├── data/
│   ├── sample/
│   │   └── series_4_RESERBUGS_EXAMPLES.csv
│   └── README.md
│
├── outputs/
│   ├── abiotic_data/
│   ├── figures/
│   └── predictions/
│
├── tests/
│   ├── conftest.py
│   ├── test_error_evaluation.py
│   └── test_reservoir_computing.py
│
├── LICENSE
├── environment.yml
└── README.md
```

---

## Installation

Follow these steps to install the package.

### 1. Clone the repository

``` bash
git clone https://github.com/your-username/reserBUGS.git
cd reserBUGS
```

------------------------------------------------------------------------

### 2. Install the package (recommended)

Install in editable mode:

``` bash
pip install -e .
```

This allows you to import:

``` python
from reserbugs.data import CopernicusDataRetriever
```

------------------------------------------------------------------------

### 3. (Optional) Create a Conda environment

If you prefer using Conda:

``` bash
conda env create -f environment.yml
conda activate reserBUGS
```

The required Python packages are listed in `environment.yml`.

------------------------------------------------------------------------

> **Note**\
> External data access (Copernicus, MODIS) is optional and not required
> for the Quick Start example.

------------------------------------------------------------------------

## Platform Support

- **Windows**: fully supported in the current workflow
- **macOS**: expected to behave similarly to Windows, but the NDVI workflow has not yet been fully validated
- **Linux**: some functions currently require platform-specific modifications

------------------------------------------------------------------------

## External Data Setup (optional)

This section is only needed if you want to retrieve environmental
variables.

reserBUGS supports:

-   Copernicus Climate Data Store (ERA5)
-   NASA Earthdata (MODIS)
  
> **Note**\
> External datasets (Copernicus, NASA) are subject to their own licences and attribution requirements (see **Data Sources and Licensing** below).

------------------------------------------------------------------------

## Copernicus API (ERA5 Climate Data)

### 1. Register

https://cds.climate.copernicus.eu/

------------------------------------------------------------------------

### 2. Create credentials file

#### Linux / macOS

``` bash
~/.cdsapirc
```

#### Windows

``` text
C:\Users\<Username>\.cdsapirc
```

------------------------------------------------------------------------

### 3. Add credentials

``` text
url: https://cds.climate.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```

------------------------------------------------------------------------

### Minimal Copernicus Test

``` python
import cdsapi

cds = cdsapi.Client()

cds.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        "variable": "2m_temperature",
        "product_type": ["monthly_averaged_reanalysis"],
        "year": "1997",
        "month": "01",
        "time": ["00:00"],
        "format": "grib"
    },
    "download.grib"
)
```

**Expected result:**

-   A file named `download.grib` is created in your working directory\
-   The file contains ERA5 climate data (e.g. temperature)\
-   No authentication prompt appears (credentials are read from
    `.cdsapirc`)\
-   The request completes without errors

------------------------------------------------------------------------

## MODIS (NDVI) via NASA Earthdata (AppEEARS)

### 1. Create account

https://urs.earthdata.nasa.gov

### 2. Login to AppEEARS

https://appeears.earthdatacloud.nasa.gov

### 3. Enable access

Ensure access to:

-   **LP DAAC (Land Processes Distributed Active Archive Center)**

------------------------------------------------------------------------

### Authentication in Python

``` python
from pathlib import Path

config = ModisRetrieverConfig(
    local_path=Path("data/modis"),
    auth_strategy="interactive",
)
```

------------------------------------------------------------------------

### Minimal MODIS Workflow

This example retrieves MODIS NDVI values for a site. It first downloads Copernicus climate data to create the required `climate_data` table.

> **Note:** This example assumes that `reserBUGS` is installed (e.g. `pip install -e .`).

```python
from pathlib import Path

from reserbugs.data import CopernicusDataRetriever, ModisDataRetriever, ModisRetrieverConfig

values_dict = {
    "AmazonForest": {
        "latitude": -3.5,
        "longitude": -62.0,
        "min_year": 2020,
        "max_year": 2021,
    }
}

# ---------------------------------------------------------------------
# Step 1: Retrieve Copernicus climate data
# ---------------------------------------------------------------------
values_dict = CopernicusDataRetriever(values_dict).retrieve_data()

# ---------------------------------------------------------------------
# Step 2: Configure MODIS retrieval
# ---------------------------------------------------------------------
config = ModisRetrieverConfig(
    local_path=Path("data/modis"),
    auth_strategy="interactive",
    cleanup_downloads=True,
    cleanup_only_hdf=False,
)

# ---------------------------------------------------------------------
# Step 3: Retrieve MODIS NDVI values
# ---------------------------------------------------------------------
modis = ModisDataRetriever(
    values_dict,
    config=config,
    copy_input=True,
)

values_with_ndvi = modis.retrieve_data()

# ---------------------------------------------------------------------
# Step 4: Inspect the downloaded NDVI information
# ---------------------------------------------------------------------
for site in values_with_ndvi:
    df = values_with_ndvi[site]["climate_data"]
    print(f"\\n{site}")
    print(df[["valid_time", "NDVI"]].head(12))
    print("min NDVI:", df["NDVI"].min())
    print("max NDVI:", df["NDVI"].max())
```

**Expected result:**

- Earthdata authentication is requested (depending on `auth_strategy`)
- Each site's `climate_data` table is updated with an `NDVI` column
- A preview of `valid_time` and `NDVI` values is printed

---

### Notes

- External data retrieval from MODIS is demonstrated in:  
  `notebooks/Example_NDVI.ipynb`
- First-time MODIS access may require browser authentication.

---

## Data Sources and Licensing

reserBUGS provides utilities to retrieve environmental data from external services. These data are **not part of this package** and are subject to their own licenses and terms of use.

### Copernicus Climate Data Store (ERA5)

Climate data retrieved via the Copernicus Climate Data Store (CDS) are provided by the Copernicus Climate Change Service (C3S).

Use of these data is subject to the Copernicus licence, which allows free use, distribution, and modification, including for commercial purposes, provided that proper attribution is given.

When publishing or redistributing results based on these data, users should include:

> Generated using Copernicus Climate Change Service information [YEAR]

If the data have been modified or processed:

> Contains modified Copernicus Climate Change Service information [YEAR]

Additionally, users should include the following disclaimer:

> Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data contained herein.

More information:
https://cds.climate.copernicus.eu/licences

---

### NASA Earthdata (MODIS NDVI – MOD13A3)

This package retrieves NDVI data from the MODIS product:

- **MOD13A3 – Monthly Vegetation Indices (NDVI), 1 km resolution**
- Collection: 061
- Provider: NASA LP DAAC (Land Processes Distributed Active Archive Center)

MODIS data are distributed by NASA Earthdata and are generally free and open to use. However, users are responsible for properly citing the dataset in any scientific or public use of results derived from these data.

Recommended citation:

> Didan, K. (2015). MOD13A3 MODIS/Terra Vegetation Indices Monthly L3 Global 1km SIN Grid V006. NASA EOSDIS Land Processes DAAC. https://doi.org/10.5067/MODIS/MOD13A3.006

Users are encouraged to consult the official dataset documentation for additional citation guidelines and product details.

---

### User Responsibility

By using reserBUGS to download or process external data, you agree to comply with the corresponding data providers’ terms and conditions.

reserBUGS does not redistribute Copernicus or NASA datasets by default.

---

## Citation

If you use **reserBUGS**, please cite the software:

> reserBUGS contributors (2026). *reserBUGS: Reservoir computing for ecological forecasting*.  
> Zenodo. https://doi.org/XXXX

A formal publication describing the method is currently in preparation and will be added here once available.

---

### Example datasets

If you use BioTIME data, please cite:

Dornelas, M. et al. (2025). **BioTIME 2.0: Expanding and Improving a Database of Biodiversity Time Series.** *Global Ecology and Biogeography*, 34(5), e70003.  
https://doi.org/10.1111/geb.70003

---

### Data sources

If your work relies on environmental data retrieved via reserBUGS,
please also cite the corresponding data providers, such as:

- Copernicus Climate Change Service (ERA5)
- NASA LP DAAC (MODIS MOD13A3)

Refer to the **Data Sources and Licensing** section for recommended citations.

---

## Acknowledgements

This work was developed as part of the ANTENNA project, funded by the Biodiversa+ and European Commission joint call (2022–2023, BiodivMon programme), with support from the Agencia Estatal de Investigación (AEI, Spain).

---

## License

This project is licensed under the BSD 3-Clause License.

See the [LICENSE](LICENSE) file for details.

---

## Contact

Open an issue on GitHub.

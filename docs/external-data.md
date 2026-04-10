# External Data

reserBUGS can retrieve environmental variables from external data providers.
These datasets are not redistributed with the package and remain subject to
their own licences and terms of use.

## Copernicus Climate Data Store

reserBUGS supports Copernicus Climate Data Store workflows for ERA5 climate
data.

Create a Copernicus account at:

<https://cds.climate.copernicus.eu/>

Then create a credentials file.

Linux and macOS:

```bash
~/.cdsapirc
```

Windows:

```text
C:\Users\<Username>\.cdsapirc
```

The file should contain:

```text
url: https://cds.climate.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```

## MODIS NDVI Through NASA Earthdata

reserBUGS retrieves NDVI data from MODIS product MOD13A3:

- Monthly Vegetation Indices, 1 km resolution.
- Collection 061.
- Provider: NASA LP DAAC.

Create an Earthdata account at:

<https://urs.earthdata.nasa.gov>

Then log in to AppEEARS:

<https://appeears.earthdatacloud.nasa.gov>

Make sure access to LP DAAC data is enabled.

## Minimal MODIS Configuration

```python
from pathlib import Path

from reserbugs.data import (
    CopernicusDataRetriever,
    ModisDataRetriever,
    ModisRetrieverConfig,
)

values_dict = {
    "AmazonForest": {
        "latitude": -3.5,
        "longitude": -62.0,
        "min_year": 2020,
        "max_year": 2021,
    }
}

values_dict = CopernicusDataRetriever(values_dict).retrieve_data()

config = ModisRetrieverConfig(
    local_path=Path("data/modis"),
    auth_strategy="interactive",
    cleanup_downloads=True,
    cleanup_only_hdf=False,
)

modis = ModisDataRetriever(
    values_dict,
    config=config,
    copy_input=True,
)

values_with_ndvi = modis.retrieve_data()
```

## Attribution

When publishing or redistributing results based on Copernicus data, include the
required Copernicus attribution and disclaimer.

When publishing or redistributing results based on MODIS data, cite the NASA LP
DAAC dataset used by the workflow.

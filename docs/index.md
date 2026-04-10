# reserBUGS

reserBUGS is a Python package for ecological forecasting with reservoir computing.
It supports workflows that combine biological time series with environmental
variables derived from remote sensing products.

The package was developed within the ANTENNA research project to support
near-term forecasting of pollinator populations, especially insects.

## What It Does

- Forecasts abundance from ecological time series.
- Combines biological observations with environmental predictors.
- Retrieves climate and vegetation data from Copernicus and MODIS workflows.
- Evaluates forecast performance with direction, magnitude, and probabilistic
  scoring metrics.
- Provides notebooks for complete end-to-end examples.

## Main Components

- `reserbugs.reservoir_computing`: reservoir computing models for forecasting.
- `reserbugs.data`: Copernicus and MODIS data retrieval utilities.
- `reserbugs.evaluation`: forecast evaluation metrics.
- `reserbugs.visualization`: Plotly-based visualization helpers.

## Project

reserBUGS is developed by contributors from Universidad Politecnica de Madrid
and Estacion Biologica de Donana.

More information about ANTENNA: <https://pollinators-antenna.eu/>

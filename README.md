# ReserBUGS computing

## Project Overview

Pollinator prediction is crucial because it allows us to anticipate changes in biodiversity and in the productivity of agricultural and natural ecosystems. 
Their monitoring and prediction help identify threats such as climate change, pesticide use, or habitat loss, facilitating decision-making to conserve them and ensure the ecological health. 

Throughout this project, various methods have been implemented to serve as baseline models, but we have focused on the implementation of the Reservoir Computing method, adding a Bayesian sequential perspective to its final layer.

## Environment setup and configuration
### Conda environment
The python packages required for using this tool are listed in the `environment.yml` file. This file can directly generate the conda environment using the following line:

```console
cd /path/to/environment.yml
conda env create -f environment.yml
```
Once the environment has been installed, it can be activated using the following line:

```console
conda activate reserBUGS
```
### Copernicus API configuration
This library includes the option to automatically download the climate data required for prediction from the Copernicus API. To use the Copernicus API, the following steps must be followed:

1. Register an account on the Copernicus Climate Data Store (CDS). You can get the account in the [following url](https://accounts.ecmwf.int/auth/realms/ecmwf/protocol/openid-connect/auth?client_id=cds&scope=openid%20email&response_type=code&redirect_uri=https%3A%2F%2Fcds.climate.copernicus.eu%2Fapi%2Fauth%2Fcallback%2Fkeycloak&state=ZbonXeh0UBrHdffFQdCkVYIhqUfE3EJIW3DH-KWaOh8&code_challenge=-sN_BNsJTjt90yWR38DQVg49SIONkYJWNUQ3Rd17rdY&code_challenge_method=S256).

2. Log in to your account and go to your user profile.

3. Locate your API key, which consists of your user ID and an API key string.

4. Create or edit the `.cdsapirc` file in your home directory (`~/.cdsapirc` for Linux and Mac users, usually `C:\Users\<Username>\.cdsapirc` in Windows environments). It should contain the following lines:

    ```console
    url: https://cds.climate.copernicus.eu/api
    key: <PERSONAL-ACCESS-TOKEN>
    ```
5. Save the file and ensure it is accessible from your Python environment.
6. Test the connection by running a simple data request using the cdsapi Python package.
   ```python
   $ python
   >>> import cdsapi
   >>> cds = cdsapi.Client()
   >>> cds.retrieve('reanalysis-era5-single-levels-monthly-means', {
           "variable": "2m_temperature",
           "product_type" : ['monthly_averaged_reanalysis'],
           "year": "1997",
           "month": "01",
           "time": ["00:00"],
           "format": "grib"
       }, 'download.grib')
   ```
7. Once executed, if this is your first time using the API, you must follow the instructions and accept the terms, as indicated in the console message

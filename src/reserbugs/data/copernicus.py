"""
Copernicus Climate Data Store (CDS) retrieval utilities.

This module provides tools to download and preprocess abiotic climate variables
from the Copernicus Climate Data Store (CDS), specifically ERA5 reanalysis data.

The main entry point is the `CopernicusDataRetriever` class, which:
- retrieves monthly or daily climate data for multiple sites
- extracts variables for given latitude/longitude coordinates
- merges and aligns datasets into pandas DataFrames
- stores results in a user-provided site dictionary structure

Supported features include:
- ERA5 monthly means and daily (sub-daily) data retrieval
- automatic handling of spatial resolution (~0.25° grid)
- merging of multiple variables into a single dataset
- retry logic for robust data download
- optional debugging utilities for dataset alignment

Typical usage
-------------
>>> retriever = CopernicusDataRetriever(values_dict)
>>> updated_dict = retriever.retrieve_data(time_scale="monthly")

Notes
-----
- Data are retrieved via the `cdsapi` client and require valid CDS credentials.
- Temporary files are created during download and automatically cleaned up.
- The module assumes that each site entry in `values_dict` contains:
  'latitude', 'longitude', 'min_year', and 'max_year'.
"""

import pandas as pd
import xarray as xr
import tempfile
import cdsapi
import zipfile
import os
import time
import calendar

class CopernicusDataRetriever:
    
    """
    Retrieve ERA5-based climate variables from the Copernicus Climate Data Store.

    The class downloads monthly or daily abiotic variables for each site in
    `values_dict` and stores the resulting data under each site's
    ``climate_data`` key.

    Parameters
    ----------
    values_dict : dict
        Dictionary of site metadata. Each site entry is expected to contain
        latitude, longitude, min_year, and max_year.
    """
    
    def __init__(self, values_dict: dict):
        self.values_dict = values_dict 
        self.c = cdsapi.Client()

    def retrieve_data(self, time_scale: str = 'monthly', max_retries: int = 5) -> dict:
        
        """
        Retrieve climate data from the Copernicus Climate Data Store (CDS)
        for all sites defined in `values_dict`.
        
        One CDS request is performed per site. Results are stored in-place under
        the key `'climate_data'`.
        
        Parameters
        ----------
        time_scale : str, optional
            Temporal resolution of the data ("monthly" or "daily").
            Default is "monthly".
        max_retries : int, optional
            Number of retry attempts in case of request failure.
        
        Returns
        -------
        dict
            Updated `values_dict` including retrieved climate data.

        Notes
        -----
        - Results are written in-place into `self.values_dict`.
        - One request is performed per site and per time scale.
        - Retries are handled with a fixed delay of 5 seconds between attempts.

        Examples
        --------
        >>> retriever = CopernicusDataRetriever(values_dict)
        >>> data = retriever.retrieve_data(time_scale="monthly")
        >>> data["site_a"]["climate_data"].head()
        """

        for site, site_info in self.values_dict.items(): #for site in self.values_dict.keys():
            
            print(f'Retrieving data for {site}')
            min_year = site_info["min_year"]
            max_year = site_info["max_year"]
            latitude = site_info["latitude"]
            longitude = site_info["longitude"]
            
            # make request:
            if time_scale == 'monthly':
                for attempt in range(max_retries):
                    try:
                        climate_data = self.retrieve_abiotic_data_monthly(
                            years=list(f'{y}' for y  in range(min_year, max_year+1)), 
                            months=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 
                            lat=latitude, 
                            lon=longitude,
                        )
                        break
                    except Exception as e:
                        print(e)
                        print(f"Attempt {attempt + 1} failed. Retrying after 5 seconds...")
                        time.sleep(5)
                else:
                    print(f' --> Failed to retrieve monthly data for site {site}. Skipping...')
                    climate_data = None
            elif time_scale == "daily":
                
                for attempt in range(max_retries):
                    try:
                        
                        years = [str(y) for y in range(min_year, max_year + 1)]
                        months = [f"{m:02d}" for m in range(1, 13)]
                        days = self.generate_valid_days(years, months)
                        
                        climate_data = self.retrieve_abiotic_data_daily(
                            years=years,
                            months=months,
                            days=days,
                            lat=latitude,
                            lon=longitude,
                        )
                        break
                    except Exception as e:
                        print(e)
                        print(f"Attempt {attempt + 1} failed. Retrying after 5 seconds...")
                        time.sleep(5)
                else:
                    print(f' --> Failed to retrieve daily data for site {site}. Skipping...')
                    climate_data = None
    
            else:
                raise ValueError('time_scale must be either "monthly" or "daily"')
            
            # add data to dictionary:
            self.values_dict[site]['climate_data'] = climate_data
        
        return self.values_dict


    def retrieve_abiotic_data_monthly(
        self,
        years: list,
        months: list,
        lat: float,
        lon: float,
        variables: list | None = None
    ) -> pd.DataFrame:
        
        """
        Retrieve monthly abiotic variables from the Copernicus Climate Data Store (CDS)
        for a given location and time period.
    
        This method downloads ERA5 monthly mean data for the selected variables and
        extracts the grid cell corresponding to the requested latitude and longitude.
        The retrieved datasets are merged and returned as a single pandas DataFrame.
    
        Parameters
        ----------
        years : list of str
            Years to retrieve, formatted as four-digit strings.
            Example: ['2019'] or ['2018', '2019'].
        months : list of str
            Months to retrieve, formatted as two-digit strings.
            Example: ['06'] or ['05', '06'].
        lat : float
            Latitude of the target location.
        lon : float
            Longitude of the target location.
        variables : list of str, optional
            List of CDS variable names to retrieve. By default, the function
            downloads wind components, temperature, precipitation, soil type,
            soil water content, and leaf area index variables.
    
        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the retrieved monthly abiotic data.
            The DataFrame includes the time coordinate (`valid_time`), spatial
            coordinates (`latitude`, `longitude`), and the requested variables.
    
        Notes
        -----
        - Data are retrieved from the ERA5 monthly means dataset:
          `reanalysis-era5-single-levels-monthly-means`.
        - The spatial request is defined as a small area around the input
          coordinates to match the ERA5 spatial resolution (0.25°).
        - Returned datasets are merged and time-aligned internally before being
          converted to a DataFrame.
        """
        if variables is None:
            variables = [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
                "total_precipitation",
                "soil_type",
                "volumetric_soil_water_layer_1",
                "volumetric_soil_water_layer_2",
                "volumetric_soil_water_layer_3",
                "volumetric_soil_water_layer_4",
                "leaf_area_index_high_vegetation",
                "leaf_area_index_low_vegetation",
            ]
        
        # Create a CDS API client
        c = self.c

        dataset = 'reanalysis-era5-single-levels-monthly-means'
        request = {
            'product_type' : ['monthly_averaged_reanalysis'],
            'variable': variables,
            'year': years,
            'month': months,
            'time': ['00:00'],
            # Area of ~0.25 degrees (ERA5 resolution)
            'area': [lat+0.125, lon-0.125, lat-0.125, lon+0.125],
            'data_format': 'netcdf',
            'download_format': 'zip'
        }

        df = self.merge_df_monthly(c, request, dataset)

        return df



    def debug_datasets(self, datasets):
        """
        Print diagnostic information about a list of xarray datasets prior to merging.
    
        This utility inspects the temporal coordinate (`valid_time`) of each dataset
        and reports key properties such as length, range, ordering, and uniqueness.
        It also compares the time coordinates across datasets to detect potential
        misalignment issues (e.g., differing timestamps like 00:00 vs 06:00).
    
        Parameters
        ----------
        datasets : list of xarray.Dataset
            List of datasets to be inspected. Each dataset is expected to contain
            a `valid_time` coordinate representing the temporal dimension.
    
        Notes
        -----
        - This function is intended for debugging and validation purposes only.
        - It does not modify the input datasets.
        - If all datasets contain a `valid_time` coordinate, the function performs
          pairwise comparisons against the first dataset to identify differences
          in time indices.
        - Particularly useful for detecting subtle alignment issues before
          calling `xarray.merge`.
    
        Output
        ------
        Prints to stdout:
        - Dataset index and variable names
        - Length, first and last timestamps
        - Whether timestamps are sorted and unique
        - Full list of timestamps (for detailed inspection)
        - Differences in time coordinates between datasets
    
        Examples
        --------
        >>> self.debug_datasets(datasets)
        --- DEBUG: datasets before merge ---
        Dataset 0
          len: 156
          first: 2003-01-01 00:00:00
          last: 2015-12-01 00:00:00
          sorted: True
          unique: True
    
        Compare dataset 0 vs dataset 1
          Exact match: True
          Only in dataset 0: []
          Only in dataset 1: []
        """
        print("\n--- DEBUG: datasets before merge ---")
        for i, ds in enumerate(datasets):
            print(f"\nDataset {i}")
            print("Variables:", list(ds.data_vars))
    
            if "valid_time" not in ds.coords:
                print("  No valid_time coordinate found")
                continue
    
            times = pd.to_datetime(ds["valid_time"].values)
            idx = pd.Index(times)
    
            print("  len:", len(idx))
            print("  first:", idx.min())
            print("  last:", idx.max())
            print("  sorted:", idx.is_monotonic_increasing)
            print("  unique:", idx.is_unique)
            print("  values:", list(idx))
    
        if len(datasets) > 1 and all("valid_time" in ds.coords for ds in datasets):
            ref = pd.Index(pd.to_datetime(datasets[0]["valid_time"].values))
            for i, ds in enumerate(datasets[1:], start=1):
                idx = pd.Index(pd.to_datetime(ds["valid_time"].values))
                print(f"\nCompare dataset 0 vs dataset {i}")
                print("  Exact match:", ref.equals(idx))
                print("  Only in dataset 0:", list(ref.difference(idx)))
                print(f"  Only in dataset {i}:", list(idx.difference(ref)))
    

    def merge_df_monthly(self, c: cdsapi.Client, request: dict, dataset: str) -> pd.DataFrame:
        """
        Retrieve, extract, merge, and post-process climate data from the Copernicus
        Climate Data Store (CDS) into a single pandas DataFrame.
    
        This method:
        1. Downloads a ZIP archive from the CDS using the provided request.
        2. Extracts NetCDF files from the archive.
        3. Loads each NetCDF file as an xarray Dataset.
        4. Normalizes the `valid_time` coordinate to monthly timestamps to ensure
           consistent alignment across datasets.
        5. Merges all datasets using `xarray.merge`.
        6. Converts the merged dataset to a pandas DataFrame.
        7. Aggregates values by date and spatial coordinates.
        8. Cleans up temporary files.
    
        Parameters
        ----------
        c : cdsapi.Client
            CDS API client used to retrieve data.
        request : dict
            Dictionary specifying the data request, including variables, spatial
            extent, and time range.
        dataset : str
            Name of the CDS dataset to retrieve.
    
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the merged and aggregated climate data with
            columns for time (`valid_time`), latitude, longitude, and requested variables.
    
        Notes
        -----
        - The `valid_time` coordinate is normalized to monthly timestamps using
          `to_period("M").to_timestamp()` to avoid misalignment issues (e.g.,
          00:00 vs 06:00 timestamps across variables).
        - Datasets are merged using `join="outer"` and `compat="no_conflicts"`
          to preserve all available data while avoiding merge conflicts.
        - Temporary files are created during download and extraction and are
          removed after processing.
        - The function groups data by date (day-level resolution) and spatial
          coordinates, summing numeric variables.
    
        Raises
        ------
        Exception
            Propagates any exceptions raised during data retrieval, file handling,
            or dataset processing.
    
        Examples
        --------
        >>> df = retriever.merge_df(client, request, "reanalysis-era5-land-monthly-means")
        >>> df.head()
        """
        # Create a temporary file to store the downloaded ZIP archive
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            temp_file_path = tmp.name  # Path to temporary ZIP file
    
        # Download data from CDS
        c.retrieve(dataset, request).download(temp_file_path)
        
        # Create a temporary directory to extract the ZIP contents
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
    
            # Open and manually combine NetCDF files
            datasets = []
            for file in os.listdir(temp_dir):
                if file.endswith(".nc"):
                    path = os.path.join(temp_dir, file)
                    ds = xr.open_dataset(path, engine='netcdf4')
                    datasets.append(ds)
            
            # Normalize time coordinates to ensure alignment across datasets
            normalized = []
    
            for ds in datasets:
                ds = ds.copy()
                times = pd.to_datetime(ds["valid_time"].values)
                times = times.to_period("M").to_timestamp()
                ds = ds.assign_coords(valid_time=times)
                normalized.append(ds)
            
            # Debug normalized datasets (optional)
            # self.debug_datasets(normalized)
            
            # Merge datasets along shared coordinates
            combined = xr.merge(
                normalized,
                join="outer",
                compat="no_conflicts"
            )
            
            # Convert merged dataset to pandas DataFrame
            df = combined.to_dataframe().reset_index()
    
            # Close all opened datasets to free resources
            for ds in datasets:
                ds.close()
    
        # Aggregate values for identical dates and spatial coordinates
        df = df.groupby(
            ['valid_time', 'latitude', 'longitude'] #[df['valid_time'].dt.date, 'latitude', 'longitude']
        ).sum(numeric_only=True).reset_index()
    
        # Drop unnecessary column if present
        df = df.drop(columns=['number'], errors='ignore')
    
        # Remove temporary ZIP file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
        return df


    @staticmethod
    def generate_valid_days(years: list[str], months: list[str]) -> list[str]:
        """
        Generate a list of valid day strings for the given years and months.
        
        This function computes the number of days in each (year, month) combination
        using the standard calendar and returns the union of all valid days formatted
        as two-digit strings (e.g., "01", "02", ..., "31").
        
        Parameters
        ----------
        years : list of str
            List of years formatted as four-digit strings (e.g., ["2020", "2021"]).
        months : list of str
            List of months formatted as two-digit strings (e.g., ["01", "02"]).
        
        Returns
        -------
        list of str
            Sorted list of unique valid day strings (two-digit format) across all
            provided years and months.
        
        Notes
        -----
        - Leap years are handled automatically (e.g., February may have 28 or 29 days).
        - The returned list represents the union of valid days across all input
          (year, month) combinations, not exact date combinations.
        - This approach is suitable for CDS API requests, where year, month, and day
          are specified independently.
        
        Examples
        --------
        >>> generate_valid_days(["2020"], ["02"])
        ['01', '02', ..., '29']
        
        >>> generate_valid_days(["2021"], ["02"])
        ['01', '02', ..., '28']
        """
        days = set()
    
        for year in years:
            for month in months:
                n_days = calendar.monthrange(int(year), int(month))[1]
                for d in range(1, n_days + 1):
                    days.add(f"{d:02d}")
    
        return sorted(days)

    def merge_df_daily(self, c: cdsapi.Client, request: dict, dataset: str) -> pd.DataFrame:
        """
        Retrieve, extract, merge, and post-process daily or sub-daily climate data
        from the Copernicus Climate Data Store (CDS) into a single pandas DataFrame.
    
        This method:
        1. Downloads a ZIP archive from the CDS using the provided request.
        2. Extracts NetCDF files from the archive.
        3. Loads each NetCDF file as an xarray Dataset.
        4. Ensures the `valid_time` coordinate is properly parsed as datetime.
        5. Merges all datasets using `xarray.merge` without altering temporal resolution.
        6. Converts the merged dataset to a pandas DataFrame.
        7. Cleans up temporary files.
    
        Parameters
        ----------
        c : cdsapi.Client
            CDS API client used to retrieve data.
        request : dict
            Dictionary specifying the data request, including variables, spatial
            extent, time range, and temporal resolution.
        dataset : str
            Name of the CDS dataset to retrieve.
    
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the merged climate data with columns for time
            (`valid_time`), latitude, longitude, and requested variables.
    
        Notes
        -----
        - The `valid_time` coordinate is preserved at its original resolution
          (e.g., hourly or sub-daily timestamps such as 00:00, 06:00, 12:00, 18:00).
        - No temporal aggregation is performed in this function.
        - Datasets are merged using `join="outer"` and `compat="no_conflicts"`
          to preserve all available data while avoiding merge conflicts.
        - Temporary files are created during download and extraction and are
          removed after processing.
    
        Raises
        ------
        Exception
            Propagates any exceptions raised during data retrieval, file handling,
            or dataset processing.
    
        Examples
        --------
        >>> df = retriever.merge_df_daily(client, request, "reanalysis-era5-single-levels")
        >>> df.head()
        """
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            temp_file_path = tmp.name
    
        c.retrieve(dataset, request).download(temp_file_path)
    
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(temp_file_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)
    
                datasets = []
                for file in os.listdir(temp_dir):
                    if file.endswith(".nc"):
                        path = os.path.join(temp_dir, file)
                        ds = xr.open_dataset(path, engine="netcdf4")
                        datasets.append(ds)
    
                normalized = []
                for ds in datasets:
                    ds = ds.copy()
                    times = pd.to_datetime(ds["valid_time"].values)
                    ds = ds.assign_coords(valid_time=times)
                    normalized.append(ds)
    
                combined = xr.merge(
                    normalized,
                    join="outer",
                    compat="no_conflicts",
                )
    
                df = combined.to_dataframe().reset_index()
    
                for ds in datasets:
                    ds.close()
    
            df = df.drop(columns=["number"], errors="ignore")
            return df
    
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    

    def retrieve_abiotic_data_daily(
        self,
        years: list,
        months: list,
        days: list,
        lat: float,
        lon: float,
        variables: list | None = None
    ) -> pd.DataFrame:
        """
        Retrieve daily abiotic variables from the Copernicus Climate Data Store (CDS)
        for a given location and time period.
    
        This method downloads ERA5 reanalysis data at multiple time steps per day
        (00:00, 06:00, 12:00, 18:00) for the selected variables. The data are extracted
        for a small spatial area around the specified latitude and longitude, merged,
        and returned as a pandas DataFrame.
    
        Parameters
        ----------
        years : list of str
            Years to retrieve, formatted as four-digit strings.
            Example: ['2019'] or ['2018', '2019'].
        months : list of str
            Months to retrieve, formatted as two-digit strings.
            Example: ['06'] or ['05', '06'].
        days : list of str
            Days to retrieve, formatted as two-digit strings.
            Example: ['13'] or ['12', '13'].
        lat : float
            Latitude of the target location.
            Example: -34.603722
        lon : float
            Longitude of the target location.
            Example: -58.381592
        variables : list of str, optional
            List of CDS variable names to retrieve.
            See: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
    
        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the retrieved abiotic data, including
            time (`valid_time`), spatial coordinates (`latitude`, `longitude`),
            and the requested variables.
    
        Notes
        -----
        - Data are retrieved from the ERA5 dataset:
          `reanalysis-era5-single-levels`.
        - Four time steps per day are retrieved (00:00, 06:00, 12:00, 18:00).
        - The spatial request is defined as a small bounding box around the input
          coordinates to match ERA5 resolution (~0.25°).
        - Data are merged and processed internally using `merge_df_daily`.
        """
        if variables is None:
            variables = [
                "2m_temperature",
                "total_precipitation",
                "soil_type",
                "low_vegetation_cover",
            ]
        
        # CDS API client
        c = self.c
    
        dataset = 'reanalysis-era5-single-levels'
    
        request = {
            'product_type': ['reanalysis'],
    
            # Each variable corresponds to a different dataset to be downloaded
            'variable': variables,
            'year': years,
            'month': months,
            'day': days,
    
            # Time steps within each day (ERA5 temporal resolution)
            'time': ['00:00', '06:00', '12:00', '18:00'],
    
            # Output format (NetCDF files inside a ZIP archive)
            'data_format': 'netcdf',
            'download_format': 'zip',
    
            # Spatial bounding box (North, West, South, East)
            # Slightly expanded (~0.25°) to match ERA5 grid resolution
            'area': [lat + 0.125, lon - 0.125, lat - 0.125, lon + 0.125],
        }
    
        df = self.merge_df_daily(c, request, dataset)
    
        return df

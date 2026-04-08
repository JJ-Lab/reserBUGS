"""
MODIS data retrieval utilities based on NASA Earthdata.

This module provides tools to download and process MODIS satellite products,
specifically monthly NDVI data (e.g., MOD13A3), for a set of geographic sites.

The main entry point is the `ModisDataRetriever` class, which:
- authenticates with NASA Earthdata
- searches and downloads MODIS granules for each site and month
- extracts NDVI values from HDF files at given coordinates
- writes results into each site's `climate_data` DataFrame

Key features include:
- configurable retrieval via `ModisRetrieverConfig`
- safe file handling and optional cleanup of downloads
- lazy loading of optional dependencies (e.g., `pyhdf`)
- validation of downloaded files and tile indexing
- support for cloud-hosted datasets via `earthaccess`

Typical usage
-------------
>>> retriever = ModisDataRetriever(values_dict)
>>> updated_dict = retriever.retrieve_data()

Configuration example
---------------------
>>> config = ModisRetrieverConfig(local_path=Path("data/modis"))
>>> retriever = ModisDataRetriever(values_dict, config=config)

Notes
-----
- Requires authentication with NASA Earthdata (e.g., via `.netrc`).
- MODIS data are provided in HDF format and require `pyhdf` for processing.
- The module assumes that each site entry in `values_dict` contains:
  'latitude', 'longitude', 'min_year', and 'max_year'.
- NDVI values are extracted from MODIS sinusoidal grid tiles using coordinate
  transformation.
"""


from __future__ import annotations

import copy
import logging
import re
import tempfile
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import earthaccess
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pyproj import Transformer



LOGGER = logging.getLogger(__name__)


class ModisDataRetrieverError(Exception):
    """Base exception for ModisDataRetriever."""


class ModisDependencyError(ModisDataRetrieverError):
    """Raised when an optional dependency required for HDF processing is missing."""


class ModisFileValidationError(ModisDataRetrieverError):
    """Raised when a downloaded or provided file does not look like an expected MODIS tile."""


@dataclass(frozen=True)
class ModisRetrieverConfig:
    """Configuration for safer, library-style MODIS retrieval."""

    short_name: str = "MOD13A3"
    version: str = "061"
    # TODO: Eliminar ruta por defecto y plantear ruta temporal.
    local_path: Path = Path("modis_data")
    dataset_name: str = "1 km monthly NDVI"
    auth_strategy: str = "netrc"
    cloud_hosted: bool = True
    max_results_per_month: Optional[int] = None
    strict_file_validation: bool = False
    cleanup_downloads: bool = False
    cleanup_only_hdf: bool = True


class ModisDataRetriever:
    """
    Retrieve MODIS monthly NDVI values for sites stored in ``values_dict``.

    Main behavior intentionally stays close to the original script:
    - authenticate with Earthdata
    - search monthly MOD13A3 granules
    - download matching files
    - extract one NDVI value per file for each site's coordinates
    - write values back into each site's ``climate_data`` DataFrame under ``NDVI``

    Improvements over the original prototype:
    - lazy import of ``pyhdf`` so the module itself can be imported safely
    - narrower public methods with validation and smaller units of work
    - configurable cache directory and logging instead of print statements
    - optional result limiting and safer path handling
    - works on a copy of the input structure by default
    """

    # Global MODIS grid constants (1 km resolution)
    X_MIN_GLOBAL = -20015109.354
    Y_MAX_GLOBAL = 10007554.677
    RES_1KM = 926.62543305
    TILE_SIZE = 1200
    TILE_WIDTH = RES_1KM * TILE_SIZE
    TILE_PATTERN = re.compile(r"h(\d{2})v(\d{2})", re.IGNORECASE)

    def __init__(
        self,
        values_dict: Mapping[str, Mapping[str, Any]],
        *,
        config: Optional[ModisRetrieverConfig] = None,
        logger: Optional[logging.Logger] = None,
        copy_input: bool = True,
    ) -> None:
        self.config = config or ModisRetrieverConfig()
        self.logger = logger or LOGGER
        self.values_dict: Dict[str, Dict[str, Any]] = (
            copy.deepcopy(values_dict) if copy_input else dict(values_dict)
        )
        self._transformer = Transformer.from_crs(
            "EPSG:4326",
            "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext",
            always_xy=True,
        )

    # ---------- Public API ----------

    def retrieve_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Authenticate, search, download, and write monthly NDVI values back into the site tables.
    
        This is the main high-level retrieval workflow of the class. For each site
        stored in ``values_dict``, the method:
    
        1. authenticates with NASA Earthdata,
        2. prepares the site's ``climate_data`` table,
        3. iterates over all monthly periods between ``min_year`` and ``max_year``,
        4. searches for and downloads matching MODIS files,
        5. extracts one NDVI value per file at the site's coordinates, and
        6. writes the result into the ``NDVI`` column of the site's climate table.
    
        The updated climate tables are written back into ``self.values_dict`` under
        each site's ``"climate_data"`` key.
    
        Returns
        -------
        dict of dict
            Updated site dictionary with NDVI values added to each site's
            ``climate_data`` DataFrame.
    
        Notes
        -----
        - The method expects each site entry in ``values_dict`` to contain:
          ``"latitude"``, ``"longitude"``, ``"min_year"``, ``"max_year"``, and
          ``"climate_data"``.
        - The ``climate_data`` table must contain a ``"valid_time"`` column.
        - If no files are found for a given month, that month is skipped.
        - If ``config.cleanup_downloads`` is True, downloaded files are deleted
          after processing, even if an error occurs.
        - The method operates on a deep copy of the input structure by default,
          depending on the value of ``copy_input`` used at initialization.
    
        Raises
        ------
        TypeError
            If a site's ``climate_data`` object is not a pandas DataFrame.
        KeyError
            If a site's ``climate_data`` table does not contain a ``"valid_time"``
            column, or if required site metadata are missing.
        ModisDependencyError
            If HDF processing is required but ``pyhdf`` is not installed.
        FileNotFoundError
            If a downloaded file path does not exist when being processed.
        ModisFileValidationError
            If file validation fails under strict validation settings.
    
        Examples
        --------
        >>> retriever = ModisDataRetriever(values_dict)
        >>> updated = retriever.retrieve_data()
        >>> updated["site_a"]["climate_data"].head()
        """
        self.login()
    
        try:
            for site_name, site_info in self.values_dict.items():
                self.logger.info("Retrieving data for %s", site_name)
                climate_df = self._prepare_climate_dataframe(site_info["climate_data"])
                latitude = float(site_info["latitude"])
                longitude = float(site_info["longitude"])
                min_year = int(site_info["min_year"])
                max_year = int(site_info["max_year"])

    
                for current in self._iter_month_starts(min_year=min_year, max_year=max_year):
                    
                    # Temporal directory
                    tmpdir = tempfile.mkdtemp(prefix="tmp_", dir=self.config.local_path.parent)
                    self.logger.debug("Created temporary directory for downloads: %s", tmpdir)
                    
                    files = self.download_month(
                        latitude=latitude,
                        longitude=longitude,
                        current=current,
                        tmpdir=tmpdir,  
                    )

                    if not files:
                        self.logger.debug("No MODIS files found for %s at %s", site_name, current.date())
                        continue
    
                    current_string = current.strftime("%Y-%m-%d")
                    for file_path in files:
                        ndvi_value = self.process_hdf(file_path, latitude, longitude)
                        self.logger.info("File: %s, NDVI Value: %s", file_path, ndvi_value)
                        climate_df.loc[climate_df["valid_time"] == current_string, "NDVI"] = ndvi_value

                    if self.config.cleanup_downloads:
                        shutil.rmtree(tmpdir)
                    
    
                self.values_dict[site_name]["climate_data"] = climate_df
    
            return self.values_dict
        
        except Exception as exc:
            self.logger.error("An error occurred during MODIS data retrieval", exc_info=True)
            raise exc
    

    def login(self) -> Any:
        """Authenticate with Earthdata using the configured strategy."""
        self.logger.debug("Authenticating with Earthdata using strategy=%s", self.config.auth_strategy)
        return earthaccess.login(strategy=self.config.auth_strategy)

    def search_month(self, *, latitude: float, longitude: float, current: datetime) -> Sequence[Any]:
        """
        Search MODIS granules for a single monthly time window.
    
        This method queries NASA Earthdata for MODIS products matching the
        configured product short name and version over the month defined by
        ``current``. The search is spatially constrained to the provided
        latitude/longitude coordinate.
    
        Parameters
        ----------
        latitude : float
            Latitude of the target site in decimal degrees.
        longitude : float
            Longitude of the target site in decimal degrees.
        current : datetime.datetime
            Datetime representing the monthly period to search. Only the year and
            month are used; the method searches from the first day of that month
            to the last day of that month.
    
        Returns
        -------
        sequence
            Sequence of Earthdata search results for the requested month. The
            concrete object type depends on the ``earthaccess`` client.
    
        Notes
        -----
        - The search uses ``config.short_name`` and ``config.version``.
        - The temporal search window is built as ``(first_day, last_day)`` of the
          month containing ``current``.
        - The spatial query uses a point-like bounding box:
          ``(longitude, latitude, longitude, latitude)``.
        - If ``config.max_results_per_month`` is set, the returned sequence is
          truncated to that number of results.
    
        Examples
        --------
        >>> results = retriever.search_month(
        ...     latitude=-34.60,
        ...     longitude=-58.38,
        ...     current=datetime(2020, 1, 1),
        ... )
        >>> len(results)
        """
        first_day = current.strftime("%Y-%m-%d")
        last_day = (current + relativedelta(day=31)).strftime("%Y-%m-%d")
        self.logger.debug(
            "Searching MODIS data for (%s, %s) between %s and %s",
            latitude,
            longitude,
            first_day,
            last_day,
        )
        results = earthaccess.search_data(
            short_name=self.config.short_name,
            version=self.config.version,
            temporal=(first_day, last_day),
            bounding_box=(longitude, latitude, longitude, latitude),
            cloud_hosted=self.config.cloud_hosted,
        )
        if self.config.max_results_per_month is not None:
            results = results[: self.config.max_results_per_month]
        return results

    def download_month(self, *, latitude: float, longitude: float, current: datetime, tmpdir: str) -> List[Path]:
        """
        Search for and download MODIS files for a single month.
    
        This method first performs a monthly Earthdata search using
        :meth:`search_month`, then downloads all matching files into the configured
        local directory. Downloaded paths are validated before being returned.
    
        Parameters
        ----------
        latitude : float
            Latitude of the target site in decimal degrees.
        longitude : float
            Longitude of the target site in decimal degrees.
        current : datetime.datetime
            Datetime representing the monthly period to download. Only the year and
            month are used when building the search window.
    
        Returns
        -------
        list of pathlib.Path
            List of validated local file paths for the downloaded MODIS granules.
            Returns an empty list if no matching files are found.
    
        Notes
        -----
        - Files are downloaded into ``config.local_path``.
        - The local directory is created automatically if it does not exist.
        - Each returned file path is checked to ensure that it exists and refers
          to a regularsys.path.append(str(Path().resolve().parent / "src")) file.
        - Validation of filename contents beyond file existence is handled later
          during HDF processing.
    
        Raises
        ------
        FileNotFoundError
            If a downloaded path does not exist after download.
        ModisFileValidationError
            If a downloaded path is not a regular file.
    
        Examples
        --------
        >>> files = retriever.download_month(
        ...     latitude=-34.60,
        ...     longitude=-58.38,
        ...     current=datetime(2020,         finally:
            if self.config.cleanup_downloads:
                self.cleanup_downloads(tempfile.tempdir)1, 1),
        ... )
        >>> files[0]
        PosixPath('...')
        """
        results = self.search_month(latitude=latitude, longitude=longitude, current=current)
        if not results:
            return []

        files = earthaccess.download(results, local_path=str(tmpdir))
        downloaded = [self._validate_local_file_path(file_path) for file_path in files]
        self.logger.debug("Downloaded %d files into %s", len(downloaded), tmpdir)
        return downloaded

    def process_hdf(self, file_path: str | Path, latitude: float, longitude: float) -> float:
        """
        Open a local MODIS HDF file and extract the NDVI value at a site coordinate.
    
        The method validates the file path, parses the MODIS tile indices from the
        filename, opens the configured dataset inside the HDF file, applies fill
        value masking and scale handling, transforms the input geographic
        coordinates into the MODIS sinusoidal grid, and extracts the corresponding
        pixel value.
    
        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to a downloaded MODIS HDF file.
        latitude : float
            Latitude of the target site in decimal degrees.
        longitude : float
            Longitude of the target site in decimal degrees.
    
        Returns
        -------
        float
            NDVI value extracted from the HDF tile at the requested coordinate.
            Returns ``nan`` if the tile cannot be identified and strict validation
            is disabled, or if the computed pixel is outside the raster bounds.
    
        Notes
        -----
        - The HDF subdataset to read is defined by ``config.dataset_name``.
        - Missing values are replaced with ``nan`` using ``_FillValue`` or
          ``FillValue`` metadata when available.
        - Scale handling follows the behavior of the current implementation to
          preserve compatibility with prior outputs.
        - Coordinates are transformed from geographic coordinates (EPSG:4326) to
          the MODIS sinusoidal projection before row and column indices are
          computed.
        - If ``config.strict_file_validation`` is True, a filename that does not
          contain MODIS tile indices of the form ``hXXvYY`` raises an exception.
    
        Raises
        ------
        FileNotFoundError
            If ``file_path`` does not exist.
        ModisFileValidationError
            If the path is not a file, or if strict file validation is enabled and
            tile indices cannot be parsed from the filename.
        ModisDependencyError
            If ``pyhdf`` is not installed.
        Exception
            Propagates lower-level HDF reading errors raised by ``pyhdf``.
    
        Examples
        --------
        >>> ndvi = retriever.process_hdf(
        ...     "MOD13A3.A2020001.h12v10.061.hdf",
        ...     latitude=-34.60,
        ...     longitude=-58.38,
        ... )
        >>> ndvi
        0.42
        """
        validated_path = self._validate_local_file_path(file_path)
        tile = self._extract_tile_indices(validated_path)
        if tile is None:
            if self.config.strict_file_validation:
                raise ModisFileValidationError(
                    f"Could not find MODIS tile indices (hXXvYY) in filename: {validated_path.name}"
                )
            return float("nan")

        SD, SDC = self._import_pyhdf()
        hdf = None
        try:
            hdf = SD(str(validated_path), SDC.READ)
            sds = hdf.select(self.config.dataset_name)

            data = sds.get().astype("float32")
            attrs = sds.attributes()
            
            fill = attrs.get("_FillValue") or attrs.get("FillValue")
            if fill is not None:
                data[data == fill] = np.nan

            scale_factor = attrs.get("scale_factor", 0.0001)
            # Preserve the original script's behavior to avoid changing outputs.
            scaled_data = data / scale_factor

            x, y = self._transformer.transform(longitude, latitude)
            h_idx, v_idx = tile
            xmin = self.X_MIN_GLOBAL + h_idx * self.TILE_WIDTH
            ymax = self.Y_MAX_GLOBAL - v_idx * self.TILE_WIDTH
            col = int((x - xmin) / self.RES_1KM)
            row = int((ymax - y) / self.RES_1KM)

            if 0 <= row < scaled_data.shape[0] and 0 <= col < scaled_data.shape[1]:
                return float(scaled_data[row, col])
            return float("nan")
        finally:
            if hdf is not None:
                try:
                    hdf.end()
                except Exception:
                    self.logger.debug("Could not close HDF handle cleanly for %s", validated_path, exc_info=True)

    # ---------- Internal helpers ----------

    def _prepare_climate_dataframe(self, climate_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(climate_df, pd.DataFrame):
            raise TypeError("site['climate_data'] must be a pandas DataFrame")
        if "valid_time" not in climate_df.columns:
            raise KeyError("site['climate_data'] must contain a 'valid_time' column")

        prepared = climate_df.copy()
        prepared["valid_time"] = pd.to_datetime(prepared["valid_time"]).dt.strftime("%Y-%m-%d")
        if "NDVI" not in prepared.columns:
            prepared["NDVI"] = np.nan
        return prepared

    def _iter_month_starts(self, *, min_year: int, max_year: int) -> Iterable[datetime]:
        current = datetime(min_year, 1, 1)
        end_date = datetime(max_year, 12, 31)
        while current <= end_date:
            yield current
            current += relativedelta(months=1)

    def _validate_local_file_path(self, file_path: str | Path) -> Path:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Downloaded file does not exist: {path}")
        if not path.is_file():
            raise ModisFileValidationError(f"Expected a file but got: {path}")
        return path

    # TODO: Eliminar método de limpieza para manejarlo con directorios temporales
    def cleanup_downloads(self, tmpdir: str) -> None:
        """Delete downloaded MODIS files from the local download directory."""
        # local_path = Path(self.config.local_path).expanduser().resolve()
    
        if not local_path.exists():
            return
    
        if getattr(self.config, "cleanup_only_hdf", True):
            for file_path in local_path.glob("*.hdf"):
                try:
                    file_path.unlink()
                except Exception:
                    self.logger.warning("Could not delete file %s", file_path, exc_info=True)
        else:
            try:
                shutil.rmtree(local_path)
            except Exception:
                self.logger.warning("Could not remove directory %s", local_path, exc_info=True)
    
    
    def _extract_tile_indices(self, file_path: Path) -> Optional[tuple[int, int]]:
        match = self.TILE_PATTERN.search(file_path.name)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _import_pyhdf() -> tuple[Any, Any]:
        try:
            from pyhdf.SD import SD, SDC
        except ImportError as exc:
            raise ModisDependencyError(
                "pyhdf is required to process MODIS HDF files. "
                "Install pyhdf in a supported environment before calling process_hdf() or retrieve_data()."
            ) from exc
        return SD, SDC


__all__ = [
    "ModisDataRetriever",
    "ModisDataRetrieverError",
    "ModisDependencyError",
    "ModisFileValidationError",
    "ModisRetrieverConfig",
]

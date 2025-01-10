import xarray as xr
import pandas as pd
import numpy as np
import os
import time
import dotenv

"""
ERA5 Monthly Data Processor

This module processes ERA5 data stored in Zarr format by splitting it into monthly NetCDF files. 
It handles both surface and pressure-level fields, calculates derived fields (e.g., relative humidity), 
and organizes the output files by month.

Features:
- Validates existing files to avoid redundant processing.
- Calculates relative humidity using the hypsometric equation.
- Processes data efficiently in chunks based on monthly intervals.

Functions:
- is_valid_netcdf: Checks the validity of a NetCDF file.
- hypsometric_pressure: Computes pressure at a given height using the hypsometric equation.
- q_2_rh: Converts specific humidity to relative humidity.
- process_month: Processes and saves data for a specific month.
- main: The main entry point for processing data, which accepts parameters for start and end dates, 
  output directory, and dataset path.

Usage:
- Run the script directly for default processing or import as a module and call `main()` with custom parameters.
"""
def is_valid_netcdf(file_path):
    """Check if a NetCDF file is valid."""
    try:
        xr.open_dataset(file_path).close()
        return True
    except:
        return False


def hypsometric_pressure(surface_pressure, geopotential_height, temperature):
    """
    Calculate the pressure at a given geopotential height using the hypsometric equation.
    
    Args:
        surface_pressure (array): Surface pressure in Pascals.
        geopotential_height (array): Geopotential height in meters.
        temperature (array): Temperature in Kelvin.
    
    Returns:
        array: Pressure at the given geopotential height in Pascals.
    """
    L = 0.0065  # Temperature lapse rate (K/m)
    R = 287.05  # Gas constant for dry air (J/(kg·K))
    g = 9.81  # Gravitational acceleration (m/s²)
    M = 0.0289644  # Molar mass of Earth's air (kg/mol)
    
    pressure = surface_pressure * (1 - (L * geopotential_height) / (R * temperature))**(g * M / (R * L))
    return pressure


def q_2_rh(temp, surface_pressure, geopotential_height, qair):
    """
    Convert specific humidity (q) to relative humidity (RH).
    
    Args:
        temp (array): Temperature in Kelvin.
        surface_pressure (array): Surface pressure in Pascals.
        geopotential_height (array): Geopotential height in meters.
        qair (array): Specific humidity in kg/kg.
    
    Returns:
        array: Relative humidity in percentage (0-100%).
    """
    pressure = hypsometric_pressure(surface_pressure, geopotential_height, temp)
    mr = qair / (1 - qair)
    e = mr * pressure / (0.62197 + mr)
    es = 611.2 * np.exp(17.67 * (temp - 273.15) / (temp - 29.65))
    rh = (e / es) * 100
    rh = np.clip(rh, 0, 100)  # Ensure RH is within bounds
    return rh


def process_month(ds, start, end, output_dir, surface_vars, pressure_vars):
    """Process and save data for a given month."""
    surface_file = os.path.join(output_dir, f"SURF_{start.strftime('%Y%m')}.nc")
    pressure_file = os.path.join(output_dir, f"PLEV_{start.strftime('%Y%m')}.nc")

    if is_valid_netcdf(surface_file) and is_valid_netcdf(pressure_file):
        print(f"Skipping {start.strftime('%Y-%m')} - Files already exist and are valid.")
        return

    ds_month = ds.sel(time=slice(start, end))

    # Process and save surface fields
    if not is_valid_netcdf(surface_file):
        temp_surface_file = surface_file + ".tmp"
        ds_surface = ds_month[surface_vars]
        if 'z_surf' in ds_surface.variables:
            ds_surface = ds_surface.rename({'z_surf': 'z'})
        ds_surface.to_netcdf(temp_surface_file, mode='w')
        os.rename(temp_surface_file, surface_file)

    # Process and save pressure-level fields
    if not is_valid_netcdf(pressure_file):
        temp_pressure_file = pressure_file + ".tmp"
        ds_pressure = ds_month[pressure_vars]

        sp = ds_surface['sp'].values
        q = ds_pressure['q'].values
        t = ds_pressure['t'].values
        z = ds_pressure['z'].values
        rh = q_2_rh(t, sp, z, q)
        ds_pressure['r'] = xr.DataArray(rh, dims=ds_pressure['q'].dims, coords=ds_pressure['q'].coords)
        ds_pressure['r'].attrs['long_name'] = 'Relative Humidity'
        ds_pressure['r'].attrs['units'] = '%'

        ds_pressure.to_netcdf(temp_pressure_file, mode='w')
        os.rename(temp_pressure_file, pressure_file)


def main(start_date, end_date, output_dir, dataset_path):
    """Main function to process ERA5 data."""
    dotenv.load_dotenv()
    os.makedirs(output_dir, exist_ok=True)
    ds = xr.open_zarr(dataset_path)

    surface_vars = ['d2m', 't2m', 'sp', 'ssrd', 'strd', 'tp', 'z_surf']
    pressure_vars = ['q', 't', 'u', 'v', 'z']

    time_ranges = pd.date_range(start=start_date, end=end_date, freq='MS')
    total_months = len(time_ranges)

    start_time = time.time()
    for idx, start in enumerate(time_ranges, start=1):
        month_start_time = time.time()
        end = (start + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        process_month(ds, start, end, output_dir, surface_vars, pressure_vars)
        month_end_time = time.time()

        elapsed_time = month_end_time - month_start_time
        completed_months = idx
        avg_time_per_month = (month_end_time - start_time) / completed_months
        remaining_months = total_months - completed_months
        estimated_remaining_time = avg_time_per_month * remaining_months

        print(f"Processed {start.strftime('%Y-%m')} in {elapsed_time:.2f} seconds.")
        print(f"Completed {completed_months}/{total_months} months. Estimated time remaining: {estimated_remaining_time / 60:.2f} minutes.")

    total_time = time.time() - start_time
    return f"Script completed in {total_time / 60:.2f} minutes."


if __name__ == "__main__":
    # Example usage
    result = main(
        start_date="2015-01-01",
        end_date="2015-12-31",
        output_dir="./inputs/climate/",
        dataset_path="s3://spi-pamir-c7-sdsc/era5_data/central_asia.zarr/"
    )

    print(result)

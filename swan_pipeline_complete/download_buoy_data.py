from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date


REMOTE_URL = "http://goosbrasil.org:8080/pnboia/Bsantos.nc"

OUTPUT_NC = Path("./data/buoy/Bsantos.nc")
OUTPUT_DATES_CSV = Path("./data/buoy/Bsantos_available_dates.csv")
OUTPUT_OBSERVATIONS_CSV = Path("./data/buoy/Bsantos_observations.csv")


def copy_remote_netcdf(remote_url: str, output_path: Path) -> None:
    """
    Copies a remote NetCDF/OPeNDAP dataset to a local NetCDF file.

    The function preserves:
    - dimensions;
    - global attributes;
    - variables;
    - variable attributes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening remote dataset:\n{remote_url}")

    with Dataset(remote_url, mode="r") as source:
        print(f"\nSaving local copy to:\n{output_path.resolve()}")

        with Dataset(output_path, mode="w", format="NETCDF4") as target:
            # Copy global attributes.
            target.setncatts({
                attribute: source.getncattr(attribute)
                for attribute in source.ncattrs()
            })

            # Copy dimensions.
            for dimension_name, dimension in source.dimensions.items():
                dimension_size = (
                    None if dimension.isunlimited() else len(dimension)
                )

                target.createDimension(
                    dimension_name,
                    dimension_size,
                )

            # Copy variables.
            for variable_name, source_variable in source.variables.items():
                print(f"Copying variable: {variable_name}")

                fill_value = None

                if "_FillValue" in source_variable.ncattrs():
                    fill_value = source_variable.getncattr("_FillValue")

                creation_options: dict[str, Any] = {
                    "varname": variable_name,
                    "datatype": source_variable.datatype,
                    "dimensions": source_variable.dimensions,
                }

                if fill_value is not None:
                    creation_options["fill_value"] = fill_value

                # Compression is useful for numeric variables.
                if source_variable.datatype.kind not in {"S", "U", "O"}:
                    creation_options["zlib"] = True
                    creation_options["complevel"] = 4

                target_variable = target.createVariable(**creation_options)

                variable_attributes = {
                    attribute: source_variable.getncattr(attribute)
                    for attribute in source_variable.ncattrs()
                    if attribute != "_FillValue"
                }

                target_variable.setncatts(variable_attributes)

                # Copy all values.
                target_variable[:] = source_variable[:]

    print("\nLocal NetCDF file created successfully.")


def print_dataset_structure(dataset_path: Path) -> None:
    """Prints dimensions and variables available in the local file."""
    with Dataset(dataset_path, mode="r") as dataset:
        print("\n" + "=" * 80)
        print("DATASET DIMENSIONS")
        print("=" * 80)

        for name, dimension in dataset.dimensions.items():
            print(
                f"{name}: size={len(dimension)}, "
                f"unlimited={dimension.isunlimited()}"
            )

        print("\n" + "=" * 80)
        print("DATASET VARIABLES")
        print("=" * 80)

        for name, variable in dataset.variables.items():
            units = getattr(variable, "units", None)
            standard_name = getattr(variable, "standard_name", None)
            long_name = getattr(variable, "long_name", None)

            print(f"\nVariable: {name}")
            print(f"  Dimensions: {variable.dimensions}")
            print(f"  Shape: {variable.shape}")
            print(f"  Type: {variable.dtype}")

            if units:
                print(f"  Units: {units}")

            if standard_name:
                print(f"  Standard name: {standard_name}")

            if long_name:
                print(f"  Long name: {long_name}")


def find_time_variable(dataset: Dataset) -> str:
    """
    Attempts to identify the temporal variable.

    Priority:
    1. common variable names;
    2. standard_name='time';
    3. axis='T';
    4. variables containing a NetCDF time unit.
    """
    common_names = [
        "time",
        "TIME",
        "datetime",
        "date_time",
        "timestamp",
        "data",
        "date",
    ]

    for name in common_names:
        if name in dataset.variables:
            variable = dataset.variables[name]

            if hasattr(variable, "units"):
                return name

    for name, variable in dataset.variables.items():
        standard_name = str(
            getattr(variable, "standard_name", "")
        ).lower()

        axis = str(getattr(variable, "axis", "")).upper()
        units = str(getattr(variable, "units", "")).lower()

        if standard_name == "time":
            return name

        if axis == "T":
            return name

        if " since " in units:
            return name

    raise KeyError(
        "A time variable could not be identified automatically. "
        "Inspect the printed variable list and configure its name manually."
    )


def convert_time_values(time_variable) -> pd.DatetimeIndex:
    """Converts a NetCDF time variable to a pandas DatetimeIndex."""
    units = getattr(time_variable, "units", None)

    if not units:
        raise ValueError(
            "The detected time variable does not have a 'units' attribute."
        )

    calendar = getattr(time_variable, "calendar", "standard")
    raw_values = time_variable[:]

    if np.ma.isMaskedArray(raw_values):
        raw_values = raw_values.compressed()
    else:
        raw_values = np.asarray(raw_values).ravel()

    converted = num2date(
        raw_values,
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
    )

    normalized_values: list[datetime] = []

    for value in np.asarray(converted).ravel():
        # Handles regular Python datetimes and cftime objects.
        normalized_values.append(
            datetime(
                value.year,
                value.month,
                value.day,
                value.hour,
                value.minute,
                value.second,
            )
        )

    return pd.DatetimeIndex(normalized_values)


def extract_available_dates(
    dataset_path: Path,
    output_csv: Path,
) -> pd.DatetimeIndex:
    """Reads the file, extracts timestamps and saves unique dates."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(dataset_path, mode="r") as dataset:
        time_variable_name = find_time_variable(dataset)
        time_variable = dataset.variables[time_variable_name]

        print("\n" + "=" * 80)
        print("TIME INFORMATION")
        print("=" * 80)

        print(f"Detected time variable: {time_variable_name}")
        print(f"Time units: {getattr(time_variable, 'units', 'not available')}")
        print(
            f"Calendar: "
            f"{getattr(time_variable, 'calendar', 'standard')}"
        )

        timestamps = convert_time_values(time_variable)

    timestamps = timestamps.dropna().sort_values().unique()
    timestamps = pd.DatetimeIndex(timestamps)

    if timestamps.empty:
        raise ValueError("No valid timestamps were found in the dataset.")

    available_dates = pd.DatetimeIndex(
        timestamps.normalize().unique()
    ).sort_values()

    print(f"\nNumber of timestamps: {len(timestamps)}")
    print(f"Number of unique dates: {len(available_dates)}")
    print(f"First timestamp: {timestamps.min()}")
    print(f"Last timestamp:  {timestamps.max()}")
    print(f"First date: {available_dates.min().date()}")
    print(f"Last date:  {available_dates.max().date()}")

    date_table = pd.DataFrame({
        "date": available_dates.strftime("%Y-%m-%d"),
        "year": available_dates.year,
        "month": available_dates.month,
        "day": available_dates.day,
    })

    date_table.to_csv(output_csv, index=False)

    print(f"\nAvailable dates saved to:\n{output_csv.resolve()}")

    return timestamps


def variable_to_1d_array(variable, expected_size: int) -> np.ndarray | None:
    """
    Converts a NetCDF variable into a one-dimensional array when its shape
    is compatible with the time dimension.
    """
    values = variable[:]

    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)

    values = np.asarray(values)

    if values.ndim == 0:
        return np.repeat(values.item(), expected_size)

    values = np.squeeze(values)

    if values.ndim == 1 and len(values) == expected_size:
        return values

    return None


def export_observations_to_csv(
    dataset_path: Path,
    output_csv: Path,
) -> None:
    """
    Exports all one-dimensional variables aligned with the time dimension
    to a CSV file.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(dataset_path, mode="r") as dataset:
        time_variable_name = find_time_variable(dataset)
        timestamps = convert_time_values(
            dataset.variables[time_variable_name]
        )

        number_of_records = len(timestamps)

        data: dict[str, Any] = {
            "datetime": timestamps,
        }

        for variable_name, variable in dataset.variables.items():
            if variable_name == time_variable_name:
                continue

            values = variable_to_1d_array(
                variable,
                expected_size=number_of_records,
            )

            if values is not None:
                data[variable_name] = values

    dataframe = pd.DataFrame(data)
    dataframe = dataframe.sort_values("datetime")

    dataframe.to_csv(output_csv, index=False)

    print(f"\nObservation table saved to:\n{output_csv.resolve()}")
    print(f"Rows exported: {len(dataframe)}")
    print(f"Columns exported: {len(dataframe.columns)}")


def main() -> None:
    copy_remote_netcdf(
        remote_url=REMOTE_URL,
        output_path=OUTPUT_NC,
    )

    print_dataset_structure(OUTPUT_NC)

    extract_available_dates(
        dataset_path=OUTPUT_NC,
        output_csv=OUTPUT_DATES_CSV,
    )

    export_observations_to_csv(
        dataset_path=OUTPUT_NC,
        output_csv=OUTPUT_OBSERVATIONS_CSV,
    )


if __name__ == "__main__":
    main()
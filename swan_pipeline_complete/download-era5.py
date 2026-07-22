from __future__ import annotations

import cdsapi

from config import (
    ERA5_DAYS,
    ERA5_MARGIN_DEGREES,
    ERA5_MONTH,
    ERA5_TIMES,
    ERA5_YEAR,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    WAVES_FILE,
    WIND_FILE,
)

def get_era5_area() -> list[float]:
    """
    Retorna a área no formato esperado pelo CDS:

        [North, West, South, East]

    adicionando uma margem ao domínio SWAN.
    """

    return [
        LAT_MAX + ERA5_MARGIN_DEGREES,  # North
        LON_MIN - ERA5_MARGIN_DEGREES,  # West
        LAT_MIN - ERA5_MARGIN_DEGREES,  # South
        LON_MAX + ERA5_MARGIN_DEGREES,  # East
    ]

def download_dataset(
    client: cdsapi.Client,
    destination,
    variables: list[str],
    description: str,
) -> None:

    if destination.exists():
        print(f"{destination.name} já existe.")
        return

    area = get_era5_area()

    print(f"\nBaixando {description}")
    print(f"Área utilizada: {area}")

    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [ERA5_YEAR],
        "month": [ERA5_MONTH],
        "day": ERA5_DAYS,
        "time": ERA5_TIMES,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }

    client.retrieve(
        "reanalysis-era5-single-levels",
        request,
        str(destination),
    )




def main() -> None:
    client = cdsapi.Client()

    download_dataset(
        client=client,
        destination=WIND_FILE,
        variables=[
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        description="vento ERA5",
    )

    download_dataset(
        client=client,
        destination=WAVES_FILE,
        variables=[
            (
                "significant_height_of_combined_"
                "wind_waves_and_swell"
            ),
            "mean_wave_direction",
            "mean_wave_period",
        ],
        description="ondas ERA5",
    )


if __name__ == "__main__":
    main()


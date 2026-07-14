import cdsapi
import os

c = cdsapi.Client()

wind_path = "data/raw/wind.nc" #path considera o de execução do script, ou seja, a raiz do projeto
waves_path = "data/raw/waves.nc"

# 🔹 VENTO
if not os.path.exists(wind_path):
    print("Baixando wind.nc...")
    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            "product_type": ["reanalysis"],
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind"
            ],
            "year": ["2023"],
            "month": ["01"],
            "day": ["01", "02", "03"],
            "time": [
                "00:00", "06:00", "12:00",
                "18:00"
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [-22, -48, -25, -44]
        },
        wind_path
    )
else:
    print("wind.nc já existe, pulando download.")

# 🔹 ONDAS
if not os.path.exists(waves_path):
    print("Baixando waves.nc...")
    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            "product_type": ["reanalysis"],
            "variable": [
                "significant_height_of_combined_wind_waves_and_swell",
                "mean_wave_direction",
                "mean_wave_period"
            ],
            "year": ["2023"],
            "month": ["01"],
            "day": ["01", "02", "03"],
            "time": [
                "00:00", "06:00", "12:00",
                "18:00"
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [-22, -48, -25, -44]
        },
        waves_path
    )
else:
    print("waves.nc já existe, pulando download.")
# install dependencies
- py -m pip install xarray numpy pandas netCDF4 scipy cdsapi
- install docker and pull swan image, openeuler/swan:latest

# create copernicus api file
- account -> https://cds.climate.copernicus.eu/ -> get api on application
- configuration file -> C:\Users\USER\.cdsapirc -> paste api access key
    - If 403 error, click on link and accept the terms of use


# Download GRID Data
 - https://www.gebco.net/data-products/historical-data-sets#gebco_2023
   - for this example we will be using the GEBCO_2023 Grid
   - paste at data/raw folder

Running application
# get era5 data
- py scripts/download_era5.py
  - will generate the raw data wind.nc and waves.nc
  - this part can take several minutes as the request get queued on the api server
  - Suggestion: call the api, if take more than 5 minutes, cancel the script with "ctrl+c" then look at the website https://cds.climate.copernicus.eu/requests?tab=all
     - once request status get's "Successful" download the file and paste at data/raw with the respective name wind.nc or waves.nc

# Process GRID Data
- py scripts/process_bathy.py
  - this will process the raw file to the depth.bot, considering the area selected and converting the depth to positive as SWAN will use this way

# Process wind and waves
- py scripts/process_wind.py
- py scripts/process_waves.py
   - both will generate the processed file on data/processed/* used to input at swan script


# Possible errors
   - missing DLL on swan folder
      - https://github.com/niXman/mingw-builds-binaries/releases/download/15.2.0-rt_v13-rev1/x86_64-15.2.0-release-win32-seh-msvcrt-rt_v13-rev1.7z
      - download extract and get those on mingw64/bin




Install docker, pull swan image
pip install scipy

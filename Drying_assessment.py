import xarray as xr
import rioxarray
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
import os


# Options: 'ssp126', 'ssp245', 'ssp370', 'ssp585'
Target_Scenario = 'ssp585'

Models = ['CanESM5-1', 'EC-Earth3', 'MIROC6', 'MPI-ESM1-2-HR', 'NorESM2-MM']
Variable = "pr"
Baseline = "20152024"
Future = "20452054"

baseline_years = list(range(2015, 2025))
future_years = list(range(2045, 2055))

# Load Drainage Basins
basins_path = r"C:\Thesis_Merwin\HydroBASINS\HydroBASIN_Global_Level3.shp"
basins = gpd.read_file(basins_path).to_crs("EPSG:4326")

basins["geometry"] = basins["geometry"].buffer(0)

# Calculate exact area in km2
basins['total_area_m2'] = basins.to_crs({'proj': 'cea'}).geometry.area
basins['Area_km2'] = basins['total_area_m2'] / 1e6

# Filter basins smaller than 100000 km2
initial_count = len(basins)
basins = basins[basins['Area_km2'] >= 100000].copy()
print(f"Initial Filter: {initial_count} down to {len(basins)} basins.")

out_dir = r"C:\Thesis_Merwin\pr_changes"
os.makedirs(out_dir, exist_ok=True)

master_df = basins[['HYBAS_ID', 'Area_km2']].copy()

area_series = master_df.set_index('HYBAS_ID')['Area_km2']

for Model in Models:
    print(f"  -> Processing Model: {Model}")

    file_base = rf"C:\Thesis_Merwin\CMIP6\Forcing_data_{Model}\{Target_Scenario}\{Variable}\{Variable}_{Target_Scenario}_{Model}_{Baseline}.nc"
    file_fut = rf"C:\Thesis_Merwin\CMIP6\Forcing_data_{Model}\{Target_Scenario}\{Variable}\{Variable}_{Target_Scenario}_{Model}_{Future}.nc"

    data_base = xr.open_dataset(file_base)[Variable]
    data_fut = xr.open_dataset(file_fut)[Variable]
    pr_data = xr.concat([data_base, data_fut], dim="time")

    pr_mm = pr_data * 86400
    pr_mm = pr_mm.assign_coords(lon=((pr_mm.lon + 180) % 360) - 180)
    pr_mm = pr_mm.sortby("lon").rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    pr_yearly = pr_mm.resample(time="1YE").sum()
    pr_yearly = pr_yearly.sortby("lat", ascending=False)

    lon = pr_yearly.lon.values
    lat = pr_yearly.lat.values
    res_lon = np.abs(np.diff(lon)[0])
    res_lat = np.abs(np.diff(lat)[0])

    grid_polys = []
    grid_indices = []
    for j, t in enumerate(lat):
        for i, l in enumerate(lon):
            grid_polys.append(box(l - res_lon / 2, t - res_lat / 2, l + res_lon / 2, t + res_lat / 2))
            grid_indices.append((j, i))

    grid_gdf = gpd.GeoDataFrame({"geometry": grid_polys, "grid_idx": grid_indices}, crs="EPSG:4326")

    intersection = gpd.overlay(basins[['HYBAS_ID', 'geometry']], grid_gdf, how='intersection')
    intersection['area_part_m2'] = intersection.to_crs({'proj': 'cea'}).geometry.area

    lat_indices = [idx[0] for idx in intersection['grid_idx']]
    lon_indices = [idx[1] for idx in intersection['grid_idx']]

    model_results = {}
    for i in range(len(pr_yearly.time)):
        year_data = pr_yearly.isel(time=i)
        year = int(pr_yearly.time.dt.year[i])

        intersection['precip_val'] = year_data.values[lat_indices, lon_indices]
        intersection['volume_m3'] = (intersection['precip_val'] / 1000) * intersection['area_part_m2']
        model_results[year] = intersection.groupby('HYBAS_ID')['volume_m3'].sum()

    model_df = pd.DataFrame(model_results) / 1e9
    model_df.index.name = "HYBAS_ID"

    # Compute Statistics
    mean_base = model_df[baseline_years].mean(axis=1)
    mean_fut = model_df[future_years].mean(axis=1)
    sd_base = model_df[baseline_years].std(axis=1)

    pct_change = ((mean_fut - mean_base) / mean_base) * 100
    snr = (mean_fut - mean_base) / sd_base

    mean_base_mm = (mean_base / area_series) * 1_000_000
    mean_fut_mm = (mean_fut / area_series) * 1_000_000

    mm_loss = ((mean_fut - mean_base) / area_series) * 1_000_000

    temp_df = pd.DataFrame({
        f'{Model} avg yearly pr in km3 for 2015-2024': mean_base,
        f'{Model} avg yearly pr in km3 for 2045-2054': mean_fut,
        f'{Model} avg yearly pr in mm for 2015-2024': mean_base_mm,
        f'{Model} avg yearly pr in mm for 2045-2054': mean_fut_mm,
        f'{Model} percent change yearly pr': pct_change,
        f'{Model} sd in km3 for 2015-2024': sd_base,
        f'{Model} SNR': snr,
        f'{Model} mm loss': mm_loss
    })

    master_df = master_df.merge(temp_df, on='HYBAS_ID', how='left')

# Calculate Multi-Model Metrics
pct_cols = [f'{m} percent change yearly pr' for m in Models]
snr_cols = [f'{m} SNR' for m in Models]
loss_cols = [f'{m} mm loss' for m in Models]

master_df['Number of models agreeing on signal (drying vs wetting)'] = (master_df[pct_cols] < 0).sum(axis=1)

pre_agreement_count = len(master_df)
master_df = master_df[master_df['Number of models agreeing on signal (drying vs wetting)'] >= 4].copy()
print(f"  -> Secondary Filter: Kept {len(master_df)} out of {pre_agreement_count} basins.")

# Calculate Multi-Model Means with all 5 models
master_df['Multi Model Mean percent change yearly pr'] = master_df[pct_cols].mean(axis=1)
master_df['Multi Model Mean SNR'] = master_df[snr_cols].mean(axis=1)
master_df['Multi Model Mean mm/m2 loss'] = master_df[loss_cols].mean(axis=1)

# Log-Transform the absolute precipitation decrease
master_df['Log-transformed mean mm/m2 loss'] = np.log(master_df['Multi Model Mean mm/m2 loss'].abs())

# Max Normalization
def max_normalize(series):
    if series.max() == 0:
        return series * 0
    return series / series.max()

# Apply Normalization
master_df['Normalized Mean PR Change'] = max_normalize(master_df['Multi Model Mean percent change yearly pr'].abs())
master_df['Normalized Mean SNR'] = max_normalize(master_df['Multi Model Mean SNR'].abs())
master_df['Normalized Log-transformed Mean mm/m2 Loss'] = max_normalize(master_df['Log-transformed mean mm/m2 loss'])

# Compute Composite Drying Index
master_df['3-Way Composite Drying Index'] = master_df[
    ['Normalized Mean PR Change',
     'Normalized Mean SNR',
     'Normalized Log-transformed Mean mm/m2 Loss']
].mean(axis=1)

# Ranking
master_df = master_df.sort_values(by='3-Way Composite Drying Index', ascending=False)

# Exporting
out_file = os.path.join(out_dir, f"Table_{Target_Scenario}_new2.xlsx")
master_df.to_excel(out_file, index=False)
print(f"✅ Successfully saved and ranked: {out_file}")
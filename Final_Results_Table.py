import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

# Paths
excel_in_path = r"C:\Thesis_Merwin\pr_changes\Final table pr changes.xlsx"
excel_out_path = r"C:\Thesis_Merwin\pr_changes\Final_Results_Thesis.xlsx"
nc_dir = r"C:\Thesis_Merwin\Results_Maps"
df_all = pd.read_excel(excel_in_path)

df = df_all.dropna(subset=['Short name']).copy()
if df['Short name'].dtype == object:
    df = df[df['Short name'].astype(str).str.strip() != '']

# Scenario name fix
df['merge_scenario'] = df['Scenario'].astype(str).str.lower().str.replace('-', '', regex=False).str.replace('.', '',
                                                                                                            regex=False)

# 2045-2054
vol_cols_2045 = [
    "CanESM5-1 avg yearly pr in km3 for 2045-2054",
    "EC-Earth3 avg yearly pr in km3 for 2045-2054",
    "MIROC6 avg yearly pr in km3 for 2045-2054",
    "MPI-ESM1-2-HR avg yearly pr in km3 for 2045-2054",
    "NorESM2-MM avg yearly pr in km3 for 2045-2054"
]
df['Mean yearly total volume 2045-2054 (km3)'] = df[vol_cols_2045].mean(axis=1)
df['Mean yearly precipitation 2045-2054 (mm/year)'] = (df['Mean yearly total volume 2045-2054 (km3)'] / df[
    'Area_km2']) * 1_000_000

# 2015-2024
vol_cols_2015 = [
    "CanESM5-1 avg yearly pr in km3 for 2015-2024",
    "EC-Earth3 avg yearly pr in km3 for 2015-2024",
    "MIROC6 avg yearly pr in km3 for 2015-2024",
    "MPI-ESM1-2-HR avg yearly pr in km3 for 2015-2024",
    "NorESM2-MM avg yearly pr in km3 for 2015-2024"
]

if all(col in df.columns for col in vol_cols_2015):
    df['Mean yearly total volume 2015-2024 (km3)'] = df[vol_cols_2015].mean(axis=1)
    df['Mean yearly precipitation 2015-2024 (mm/year)'] = (df['Mean yearly total volume 2015-2024 (km3)'] / df[
        'Area_km2']) * 1_000_000

# Precipitation decrease
if 'Multi Model Mean mm loss' not in df.columns:
    df['Multi Model Mean mm loss'] = df['Mean yearly precipitation 2015-2024 (mm/year)'] - df[
        'Mean yearly precipitation 2045-2054 (mm/year)']

valid_hybas_ids = set(df['HYBAS_ID'].astype(int).tolist())

# Precipitation change
nc_files = glob.glob(os.path.join(nc_dir, "**", "Precipitation_Change_*.nc"), recursive=True)

nc_results = []

for file in nc_files:
    basename = os.path.basename(file)
    parts = basename.replace('.nc', '').split('_')

    if len(parts) >= 4:
        hybas_id = int(parts[2])
        nc_scenario = parts[3].lower()
        display_hybas_id = str(hybas_id)
    else:
        continue

    if hybas_id not in valid_hybas_ids:
        continue

    ds = xr.open_dataset(file)
    da = ds['precipitation_gain_mm_yr']

    # Only increase
    da_pos = da.where(da > 0)

    # Rate to volume
    R = 6371000
    lat = np.deg2rad(da_pos.lat)
    lon_res = np.deg2rad(np.abs(da_pos.lon[1] - da_pos.lon[0]))
    lat_res = np.deg2rad(np.abs(da_pos.lat[1] - da_pos.lat[0]))

    area_grid = (R ** 2) * np.cos(lat) * lat_res * lon_res

    da_meters = da_pos * 0.001
    volume_per_cel = da_meters * area_grid

    # Best grid cell
    max_vol = volume_per_cel.max(skipna=True)

    # Volume and Area
    positive_volume_km3 = float(volume_per_cel.sum(skipna=True).item()) / 1e9
    best_pixel_volume_km3 = float(max_vol.item()) / 1e9
    best_pixel_area_m2 = float(area_grid.where(volume_per_cel == max_vol).max(skipna=True).item())
    best_pixel_area_km2 = best_pixel_area_m2 / 1_000_000

    nc_results.append({
        'HYBAS_ID': hybas_id,
        'Display_HYBAS_ID': display_hybas_id,
        'merge_scenario': nc_scenario,
        'internal_volume_increase_km3': positive_volume_km3,
        'best_pixel_volume_km3': best_pixel_volume_km3,
        'best_pixel_area_km2': best_pixel_area_km2
    })
    ds.close()

df_nc = pd.DataFrame(nc_results)

# Compute metrics in table
if not df_nc.empty:
    df_final = pd.merge(df, df_nc, on=['HYBAS_ID', 'merge_scenario'], how='inner')

    df_final['Mean positive precipitation increase (mm/year)'] = (df_final['internal_volume_increase_km3'] / df_final[
        'Area_km2']) * 1_000_000

    df_final['Best pixel contribution (mm/year)'] = (df_final['best_pixel_volume_km3'] / df_final[
        'Area_km2']) * 1_000_000

    df_final['Relative change (%)'] = (df_final['Mean positive precipitation increase (mm/year)'] / df_final[
        'Mean yearly precipitation 2045-2054 (mm/year)']) * 100

    df_final['contrib_best_pixel'] = (df_final['Best pixel contribution (mm/year)'] / df_final[
        'Mean positive precipitation increase (mm/year)']) * 100

    df_final['internal_volume_increase_m3'] = df_final['internal_volume_increase_km3'] * 1e9
    df_final['best_pixel_volume_m3'] = df_final['best_pixel_volume_km3'] * 1e9

    scenario_order = ["ssp126", "ssp245", "ssp370", "ssp585"]

    df_final['merge_scenario'] = pd.Categorical(
        df_final['merge_scenario'],
        categories=scenario_order,
        ordered=True
    )

    df_final = df_final.sort_values(by=['merge_scenario', 'HYBAS_ID'], ascending=[True, True])

    final_columns = [
        'Display_HYBAS_ID',
        'Scenario',
        'Region',
        'Short name',
        'Area_km2',
        'Mean yearly precipitation 2015-2024 (mm/year)',
        'Mean yearly precipitation 2045-2054 (mm/year)',
        'Multi Model Mean mm loss',
        'Mean positive precipitation increase (mm/year)',
        'internal_volume_increase_km3',
        'Relative change (%)',
        'best_pixel_area_km2',
        'best_pixel_volume_m3',
        'Best pixel contribution (mm/year)',
        'contrib_best_pixel'
    ]

    df_export = df_final[final_columns].copy()

    # Rename variables
    df_export = df_export.rename(columns={
        'Display_HYBAS_ID': 'HYBAS id',
        'Scenario': 'Scenario',
        'Region': 'Region',
        'Short name': 'Short name',
        'Area_km2': 'Area (km2)',
        'Mean yearly precipitation 2015-2024 (mm/year)': 'Mean yearly precipitation 2015-2024 (mm/year)',
        'Mean yearly precipitation 2045-2054 (mm/year)': 'Mean yearly precipitation 2045-2054 (mm/year)',
        'Multi Model Mean mm loss': 'Mean yearly precipitation loss between 2015-2024 and 2045-2054 (mm/year)',
        'Mean positive precipitation increase (mm/year)': 'Mean precipitation increase with maximum reforestation (mm/year)',
        'internal_volume_increase_km3': 'Total precipitation increase with maximum reforestation (km3)',
        'Relative change (%)': 'Relative increase with maximum reforestation (%)',
        'best_pixel_area_km2': 'Size of best pixel (km2)',
        'best_pixel_volume_m3': 'Precipitation increase with reforestation of best pixel (m3)',
        'Best pixel contribution (mm/year)': 'Precipitation increase with reforestation of best pixel (mm/year)',
        'contrib_best_pixel': 'Contribution to precipitation increase with reforestation of best pixel (%)'
    })

    df_export = df_export.round(3)

    df_export.to_excel(excel_out_path, index=False)
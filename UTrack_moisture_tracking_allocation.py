import os
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd

models = ['CanESM5-1', 'EC-Earth3', 'MIROC6', 'MPI-ESM1-2-HR', 'NorESM2-MM']
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

# File paths
shapefile_path = r"C:\Thesis_Merwin\HydroBASINS\HydroBASIN_Global_Level3.shp"
master_excel_path = r"C:\Thesis_Merwin\pr_changes\Final_Results_Thesis.xlsx"
output_path = r"C:\Thesis_Merwin\UTrack_outputs\UTrack_moisture_tracking_allocation.xlsx"
hybas_gdf_shp = gpd.read_file(shapefile_path)

baseline_dfs = {}
for scenario in scenarios:
    baseline_path = rf"C:\Thesis_Merwin\pr_changes\Table_{scenario}_new.xlsx"
    df_temp = pd.read_excel(baseline_path)
    df_temp.columns = df_temp.columns.str.strip()
    df_temp['HYBAS_ID'] = pd.to_numeric(df_temp['HYBAS_ID'], errors='coerce')
    baseline_dfs[scenario] = df_temp

df_master = pd.read_excel(master_excel_path)
df_master.columns = df_master.columns.str.strip()

output_rows = []

for index, row in df_master.iterrows():
    hybas_id = int(row['HYBAS id'])
    scenario = str(row['Scenario']).strip()

    target_basin = hybas_gdf_shp[hybas_gdf_shp['HYBAS_ID'] == hybas_id].copy()

    basin_area_m2 = target_basin.to_crs("ESRI:54034").geometry.area.iloc[0]
    basin_area_km2 = basin_area_m2 / 1e6

    row_data = {
        'Drainage basin': hybas_id,
        'Scenario': scenario,
        'Area(km2)': basin_area_km2
    }

    mm_precip_list = []
    mm_allocated_list = []

    for model in models:
        input_col_name = f'{model} avg yearly pr in mm for 2045-2054'
        model_cmip6_precip_mm = np.nan

        if scenario in baseline_dfs:
            df_base = baseline_dfs[scenario]
            match_row = df_base[df_base['HYBAS_ID'] == hybas_id]
            if not match_row.empty and input_col_name in match_row.columns:
                model_cmip6_precip_mm = match_row.iloc[0][input_col_name]

        #Load moisture tracking result
        model_allocated_mm = np.nan
        utrack_file = rf"C:\Thesis_Merwin\UTrack_outputs\{scenario}\{hybas_id}_output\UTrack-Backward-{model}_{scenario}_31-12-2054_16-1-2045_{hybas_id}.nc"

        if os.path.exists(utrack_file):
            with xr.open_dataset(utrack_file) as ds_utrack:
                footprint = ds_utrack['footprint'].values / 10
                lats_u = ds_utrack.lat.values
                lons_u = ds_utrack.lon.values

                R = 6371000.0
                lat_res = abs(lats_u[1] - lats_u[0])
                lon_res = abs(lons_u[1] - lons_u[0])
                lat_rad = np.radians(lats_u)

                cell_area = (R ** 2) * np.radians(lon_res) * np.abs(
                    np.sin(np.radians(lats_u + lat_res / 2)) - np.sin(np.radians(lats_u - lat_res / 2)))
                global_area_m2 = np.tile(cell_area[:, np.newaxis], (1, len(lons_u)))

                allocated_vol_m3 = np.nansum((footprint * 0.001) * global_area_m2)
                model_allocated_mm = (allocated_vol_m3 / basin_area_m2) * 1000
        else:
            print(f"  [!] Missing NetCDF: {model} {scenario}")

        #Calculate Allocation fraction and Correction factor
        model_fraction = np.nan
        correction_factor = np.nan
        model_corrected_allocated_mm = np.nan

        if pd.notna(model_cmip6_precip_mm) and model_cmip6_precip_mm > 0 and pd.notna(model_allocated_mm):
            model_fraction = model_allocated_mm / model_cmip6_precip_mm

            # Correction for accurate mass balance
            if model_fraction > 1.0:
                correction_factor = 1.0 / model_fraction
            else:
                correction_factor = 1.0

            # Apply correction
            model_corrected_allocated_mm = model_allocated_mm * correction_factor

        row_data[f'{model} annual precipitation for 2045-2054 (mm)'] = model_cmip6_precip_mm
        row_data[f'{model} raw allocated moisture (mm)'] = model_allocated_mm
        row_data[f'{model} fraction allocated'] = model_fraction
        row_data[f'{model} correction factor'] = correction_factor
        row_data[f'{model} corrected allocated moisture (mm)'] = model_corrected_allocated_mm

        if pd.notna(model_cmip6_precip_mm):
            mm_precip_list.append(model_cmip6_precip_mm)
        if pd.notna(model_corrected_allocated_mm):
            mm_allocated_list.append(model_corrected_allocated_mm)

    # New values
    mm_precip_mean = np.mean(mm_precip_list) if mm_precip_list else np.nan
    mm_allocated_mean = np.mean(mm_allocated_list) if mm_allocated_list else np.nan
    mm_fraction = np.nan

    if pd.notna(mm_precip_mean) and mm_precip_mean > 0:
        mm_fraction = mm_allocated_mean / mm_precip_mean

    row_data['multi-model annual precipitation'] = mm_precip_mean
    row_data['multi model CORRECTED allocated moisture'] = mm_allocated_mean
    row_data['multi model CORRECTED fraction allocated'] = mm_fraction

    output_rows.append(row_data)

# Export
df_clean_output = pd.DataFrame(output_rows)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_clean_output.to_excel(output_path, index=False)
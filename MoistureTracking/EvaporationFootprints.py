import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, FuncFormatter
from rasterio.features import rasterize
from affine import Affine
from shapely.geometry import box, mapping
from UTrack_functions import *

# ==========================================
# Parameters
# ==========================================
models = ['CanESM5-1', 'EC-Earth3', 'MIROC6', 'MPI-ESM1-2-HR', 'NorESM2-MM']
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

scenario = "ssp245"
hybas_id = 9030005280
margin = 51
year_label = "2045-2054"

# abc label "left" or "right"
abc_position = "left"

# Cumulative moisture
cumulative_vol_pct = 0.50

shapefile_path = r"C:\Thesis_Merwin\HydroBASINS\HydroBASIN_Global_Level3.shp"
metrics_table_path = r"C:\Thesis_Merwin\UTrack_outputs\UTrack_moisture_tracking_allocation.xlsx"
hybas_gdf_shp = gpd.read_file(shapefile_path)

# Allocation correction
df_metrics = pd.read_excel(metrics_table_path)
basin_metrics = df_metrics[(df_metrics['Drainage basin'] == hybas_id) & (df_metrics['Scenario'] == scenario)]

# master grid (regridding)
master_model = 'EC-Earth3'
master_file_path = rf"C:\Thesis_Merwin\UTrack_outputs\{scenario}\{hybas_id}_output\UTrack-Backward-{master_model}_{scenario}_31-12-2054_16-1-2045_{hybas_id}.nc"

with xr.open_dataset(master_file_path) as ds_master:
    ref_lats = ds_master['lat'].values
    ref_lons = ds_master['lon'].values

agreement_array = np.zeros((len(ref_lats), len(ref_lons)))
sum_array = np.zeros((len(ref_lats), len(ref_lons)))

R = 6371000.0
lat_res = abs(ref_lats[1] - ref_lats[0])
lon_res = abs(ref_lons[1] - ref_lons[0])
lon_res_rad = np.radians(lon_res)

lat_bounds_upper = np.radians(ref_lats + lat_res / 2)
lat_bounds_lower = np.radians(ref_lats - lat_res / 2)

area_per_lat = (R ** 2) * lon_res_rad * np.abs(np.sin(lat_bounds_upper) - np.sin(lat_bounds_lower))

cell_areas = np.tile(area_per_lat[:, np.newaxis], (1, len(ref_lons)))

model_footprints = {}

for model in models:
    file_path = rf"C:\Thesis_Merwin\UTrack_outputs\{scenario}\{hybas_id}_output\UTrack-Backward-{model}_{scenario}_31-12-2054_16-1-2045_{hybas_id}.nc"

    with xr.open_dataset(file_path) as ds:
        lon_left = ds.copy().assign_coords(lon=ds['lon'] - 360)
        lon_right = ds.copy().assign_coords(lon=ds['lon'] + 360)
        ds_cyclic = xr.concat([lon_left, ds, lon_right], dim='lon').sortby('lon')

        ds_interp = ds_cyclic.interp(lat=ref_lats, lon=ref_lons, method='linear')
        output_array = ds_interp['footprint'].values

        clean_array = np.nan_to_num(output_array, nan=0.0)

        correction_factor = 1.0
        col_name = f'{model} correction factor'

        if not basin_metrics.empty and col_name in basin_metrics.columns:
            val = basin_metrics.iloc[0][col_name]
            if pd.notna(val):
                correction_factor = val

        model_footprints[model] = clean_array

        volume_array = (clean_array * 0.001) * cell_areas
        total_volume = np.sum(volume_array)

        flat_volumes = volume_array.flatten()
        sort_indices = np.argsort(flat_volumes)[::-1]

        sorted_volumes = flat_volumes[sort_indices]
        cum_sum_volumes = np.cumsum(sorted_volumes)

        target_volume = total_volume * cumulative_vol_pct
        cutoff_idx = np.argmax(cum_sum_volumes >= target_volume)

        model_mask_flat = np.zeros_like(flat_volumes)
        model_mask_flat[sort_indices[:cutoff_idx + 1]] = 1
        model_mask = model_mask_flat.reshape(clean_array.shape)

        agreement_array += model_mask
        sum_array += clean_array

average_array = sum_array / len(models)

any_agreement_avg = np.where(agreement_array > 0, average_array, np.nan)
high_agreement_avg = np.where(agreement_array >= 4, average_array, np.nan)

lats = ref_lats
lons = ref_lons

# Rolling the map
width = len(lons)
shift_amount = width // 2

agreement_rolled = np.roll(agreement_array, shift_amount, axis=1)
any_avg_rolled = np.roll(any_agreement_avg, shift_amount, axis=1) / 10
avg_rolled = np.roll(high_agreement_avg, shift_amount, axis=1) / 10

agreement_plot = np.where(agreement_rolled > 0, agreement_rolled, np.nan)

hybas_gdf = hybas_gdf_shp.copy()
hybas_gdf['geometry'] = hybas_gdf.translate(xoff=180)
target_basin = hybas_gdf[hybas_gdf['HYBAS_ID'] == hybas_id]

evap_any_mm = any_agreement_avg / 10
evap_high_mm = high_agreement_avg / 10

total_depth_any = np.nansum(evap_any_mm)
avg_depth_any = np.nanmean(evap_any_mm)
total_vol_any_m3 = np.nansum(evap_any_mm * 0.001 * cell_areas)
total_vol_any_km3 = total_vol_any_m3 / 1e9

footprint_mask = (agreement_array > 0)
high_agreement_mask = (agreement_array >= 4)

total_footprint_area_m2 = np.sum(cell_areas[footprint_mask])
high_agreement_area_m2 = np.sum(cell_areas[high_agreement_mask])

if total_footprint_area_m2 > 0:
    percentage_high_agreement = (high_agreement_area_m2 / total_footprint_area_m2) * 100
else:
    percentage_high_agreement = 0.0

total_depth_high = np.nansum(evap_high_mm)
avg_depth_high = np.nanmean(evap_high_mm)
total_vol_high_m3 = np.nansum(evap_high_mm * 0.001 * cell_areas)
total_vol_high_km3 = total_vol_high_m3 / 1e9

original_basin = hybas_gdf_shp[hybas_gdf_shp['HYBAS_ID'] == hybas_id]
basin_area_m2 = original_basin.to_crs("ESRI:54034").geometry.area.iloc[0]
basin_area_km2 = basin_area_m2 / 1e6

depth_any_normalized_mm = (total_vol_any_km3 * 1_000_000) / basin_area_km2
depth_high_normalized_mm = (total_vol_high_km3 * 1_000_000) / basin_area_km2

# Print metrics
print(f"METRICS {hybas_id} ({scenario.upper()})")
print("1. Map 1 (Total 50% Cumulative Footprint - All Models):")
print(f"Basin-normalized depth: {depth_any_normalized_mm:,.2f} mm/year")

print("\n2. Model Agreement Area (4/5 or 5/5 models agree):")
print(f"Percentage of footprint area: {percentage_high_agreement:.2f}%")

print("\n3. Map 3 (Evaporation in High Agreement Area):")
print(f"Basin-normalized depth: {depth_high_normalized_mm:,.2f} mm/year")


# Footprint Contribution to Basin Precipitation
average_depth_mm_year = average_array / 10
average_volume_m3_year = (average_depth_mm_year * 0.001) * cell_areas

high_ag_mask = (agreement_array >= 4)
vol_high_ag_m3 = np.sum(average_volume_m3_year[high_ag_mask])
vol_high_ag_km3 = vol_high_ag_m3 / 1e9

total_basin_precip_vol_m3 = np.sum(average_volume_m3_year)
basin_total_precip_mm = (total_basin_precip_vol_m3 / basin_area_m2) * 1000

precip_from_high_ag_mm = (vol_high_ag_m3 / basin_area_m2) * 1000
pct_of_total_precip = (precip_from_high_ag_mm / basin_total_precip_mm) * 100

# Print metrics

print("FOOTPRINT CONTRIBUTION TO BASIN PRECIPITATION")
print(f"Drainage Basin Area: {basin_area_km2:,.2f} km²")
print(f"Total Basin Precipitation: {basin_total_precip_mm:.3f} mm/year")
print(f"Moisture originated from the High-Agreement Footprint (4/5 or 5/5):")
print(f"> Contributes directly: {precip_from_high_ag_mm:.2f} mm/year to basin rainfall")
print(f"> Represents: {pct_of_total_precip:.2f}% of total basin precipitation")

# Land vs Open Water
high_agreement_vol_array = ((high_agreement_avg / 10) * 0.001 * cell_areas) / 1e9
total_high_agreement_vol = np.nansum(high_agreement_vol_array)

resolution_factor = 10
lon_step = (ref_lons[-1] - ref_lons[0]) / (len(ref_lons) - 1)
lat_step = (ref_lats[-1] - ref_lats[0]) / (len(ref_lats) - 1)
sub_lon_res = lon_step / resolution_factor
sub_lat_res = lat_step / resolution_factor

transform = (Affine.translation(ref_lons[0] - lon_step / 2, ref_lats[0] - lat_step / 2) * Affine.scale(sub_lon_res,
                                                                                                       sub_lat_res))
west_poly = box(-180, -90, 0, 90)
east_poly = box(0, -90, 180, 90)
world_west = gpd.clip(hybas_gdf_shp, west_poly)
world_east = gpd.clip(hybas_gdf_shp, east_poly)
world_west_shifted = world_west.copy()
world_west_shifted['geometry'] = world_west.translate(xoff=360)
hybas_360 = pd.concat([world_east, world_west_shifted], ignore_index=True)

geom_value_pairs = [(mapping(geom), 1) for geom in hybas_360.geometry]
hr_shape = (len(ref_lats) * resolution_factor, len(ref_lons) * resolution_factor)
land_mask_hr = rasterize(geom_value_pairs, out_shape=hr_shape, transform=transform, all_touched=True, dtype=float)

fractional_land_mask = land_mask_hr.reshape(len(ref_lats), resolution_factor, len(ref_lons), resolution_factor).mean(
    axis=(1, 3))

land_vol_km3 = np.nansum(high_agreement_vol_array * fractional_land_mask)
ocean_vol_km3 = total_high_agreement_vol - land_vol_km3

land_pct = (land_vol_km3 / total_high_agreement_vol) * 100
ocean_pct = (ocean_vol_km3 / total_high_agreement_vol) * 100

# Convert volume to mm/year
total_high_agreement_depth_mm = (total_high_agreement_vol * 1_000_000) / basin_area_km2
land_depth_mm = (land_vol_km3 * 1_000_000) / basin_area_km2
ocean_depth_mm = (ocean_vol_km3 * 1_000_000) / basin_area_km2

print(f"Total High-Agreement evaporation: {total_high_agreement_depth_mm:.2f} mm/year")
print(f"Contribution from Land: {land_depth_mm:.2f} mm/year ({land_pct:.1f}%)")
print(f"Contribtion from Ocean: {ocean_depth_mm:.2f} mm/year ({ocean_pct:.1f}%)")

# Square map based on center of basin
if margin is not None:
    min_lon, min_lat, max_lon, max_lat = target_basin.total_bounds

    # Center of the basin
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    max_dim = max(max_lon - min_lon, max_lat - min_lat)

    half_span = (max_dim / 2) + margin

    plot_xlim = (center_lon - half_span, center_lon + half_span)
    plot_ylim = (center_lat - half_span, center_lat + half_span)
else:
    world_min_lon, world_min_lat, world_max_lon, world_max_lat = hybas_gdf.total_bounds
    plot_xlim = (world_min_lon - 2, world_max_lon + 2)
    plot_ylim = (world_min_lat - 2, world_max_lat + 2)

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 8))
plt.subplots_adjust(wspace=0.15)

cmap_avg = plt.get_cmap('Blues').copy()
cmap_avg.set_bad(color=(0, 0, 0, 0))
vmax_moisture = np.nanmax(any_avg_rolled)

# Panel a
hybas_gdf.plot(ax=ax1, facecolor='lightgrey', edgecolor='black', linewidth=0.4, zorder=1, aspect=None)
target_basin.boundary.plot(ax=ax1, color='#00AA00', linewidth=1, zorder=3, aspect=None)

img1 = ax1.imshow(
    any_avg_rolled, cmap=cmap_avg, vmin=0, vmax=vmax_moisture, interpolation='nearest',
    origin='lower', extent=(0, 360, lats.min(), lats.max()), aspect='equal', zorder=2
)

divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.14)
cbar1 = fig.colorbar(img1, cax=cax1)
cbar1.set_label("Annual Evaporation (mm)", fontweight='bold')
ax1.set_title("Average Evaporation (All Models)", fontsize=12, fontweight='bold')

# Panel b
hybas_gdf.plot(ax=ax2, facecolor='lightgrey', edgecolor='black', linewidth=0.4, zorder=1, aspect=None)
target_basin.boundary.plot(ax=ax2, color='#00AA00', linewidth=1, zorder=3, aspect=None)

colors_ag = ['#fee8c8', '#fdbb84', '#fc8d59', '#e34a33', '#b30000']
cmap_ag = mcolors.ListedColormap(colors_ag)
cmap_ag.set_bad(color=(0, 0, 0, 0))
bounds_ag = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
norm_ag = mcolors.BoundaryNorm(bounds_ag, cmap_ag.N)

img2 = ax2.imshow(
    agreement_plot, cmap=cmap_ag, norm=norm_ag, interpolation='nearest',
    origin='lower', extent=(0, 360, lats.min(), lats.max()), aspect='equal', zorder=2
)

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.14)
cbar2 = fig.colorbar(img2, cax=cax2, ticks=[1, 2, 3, 4, 5])
cbar2.set_label("Model Consensus", fontweight='bold')
ax2.set_title("Spatial Model Consensus", fontsize=12, fontweight='bold')

# Panel c
hybas_gdf.plot(ax=ax3, facecolor='lightgrey', edgecolor='black', linewidth=0.4, zorder=1, aspect=None)
target_basin.boundary.plot(ax=ax3, color='#00AA00', linewidth=1, zorder=3, aspect=None)

img3 = ax3.imshow(
    avg_rolled, cmap=cmap_avg, vmin=0, vmax=vmax_moisture, interpolation='nearest',
    origin='lower', extent=(0, 360, lats.min(), lats.max()), aspect='equal', zorder=2
)

divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="5%", pad=0.14)
cbar3 = fig.colorbar(img3, cax=cax3)
cbar3.set_label("Annual Evaporation (mm)", fontweight='bold')
ax3.set_title("Average Evaporation (Model consensus ≥4)", fontsize=12, fontweight='bold')

def format_lon(x, pos):
    real_lon = x - 180
    if real_lon == 0:
        return "0°"
    elif real_lon == 180 or real_lon == -180:
        return "180°"
    elif real_lon > 0:
        return f"{int(real_lon)}°E"
    else:
        return f"{int(-real_lon)}°W"

def format_lat(y, pos):
    if y == 0:
        return "0°"
    elif y > 0:
        return f"{int(y)}°N"
    else:
        return f"{int(-y)}°S"

for ax, letter in zip([ax1, ax2, ax3], ['a', 'b', 'c']):

    # Square map
    ax.set_xlim(plot_xlim)
    ax.set_ylim(plot_ylim)

    ax.xaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_major_locator(MultipleLocator(15))
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.set_xlabel("")
    ax.set_ylabel("")

    # Add abc label
    box_props = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.9, edgecolor='grey')
    if abc_position.lower() == "left":
        ax.text(0.03, 0.97, letter, transform=ax.transAxes, fontsize=24, fontweight='bold',
                va='top', ha='left', bbox=box_props, zorder=10)
    elif abc_position.lower() == "right":
        ax.text(0.97, 0.97, letter, transform=ax.transAxes, fontsize=24, fontweight='bold',
                va='top', ha='right', bbox=box_props, zorder=10)

# Add statistics to Panel c
stats_text = (
    f"Total annual footprint evap: {total_high_agreement_depth_mm:{',.1f' if total_high_agreement_depth_mm < 10 else ',.0f'}} mm\n"
    f"Land contribution: {land_pct:.1f}%\n"
    f"Open water contribution: {ocean_pct:.1f}%"
)

props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
ax3.text(0.03, 0.03, stats_text, transform=ax3.transAxes, fontsize=13,
         verticalalignment='bottom', bbox=props, zorder=5, fontweight='normal')

for ax_to_clean in [ax2, ax3]:
    ax_to_clean.set_yticklabels([])

plt.show(block=False)
plt.pause(0.1)

save_choice = input("Type 'ok' to save: ").strip().lower()

if save_choice == 'ok':
    save_path = rf"C:\Thesis_Merwin\UTrack_outputs\New"
    os.makedirs(save_path, exist_ok=True)

    output_filename = f"Evaporation_Footprint_{hybas_id}_{scenario}_3.png"
    full_save_path = os.path.join(save_path, output_filename)

    moisture_da = xr.DataArray(
        high_agreement_avg / 10,
        coords={'lat': lats, 'lon': lons},
        dims=["lat", "lon"],
        name="yearly_avg_moisture"
    )

    data_filename = f"Evaporation_Footprint_{hybas_id}_{scenario}_Allocated.nc"
    data_full_path = os.path.join(save_path, data_filename)

    moisture_da.to_netcdf(data_full_path)

    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')

plt.close(fig)

import os
import math
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import numpy as np
import copy
import geopandas as gpd
import pandas as pd
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors

# Dataset selection - Forestation, Evap_increase, Precip_increase
dataset = "Forestation"

# Scenario selection - ssp126, ssp245, ssp370, ssp585
scenario = "ssp126"

# Data Paths
excel_path = r"C:\Thesis_Merwin\pr_changes\Final_Results_Thesis.xlsx"
shapefile_path = r"C:\Thesis_Merwin\HydroBASINS\HydroBASIN_Global_Level3.shp"
save_dir = r"C:\Thesis_Merwin\Results_Maps\NewMaps"
os.makedirs(save_dir, exist_ok=True)

# Dynamicly adjust for dataset selected
if dataset == "Forestation":
    data_nc_path = r"C:\Thesis_Merwin\Forest_restoration\Forest_restoration_potential\Restoration_potential_regridded.nc"
    save_png_path = os.path.join(save_dir, f"Forestation_plots_{scenario}.png")
    cmap_name = 'YlGn'
elif dataset == "Evap_increase":
    data_nc_path = r"C:\Thesis_Merwin\Forest_restoration\Evap_Change_Max_Reforestation\Evap_Change_Max_Reforestation.nc"
    save_png_path = os.path.join(save_dir, f"Evap_increase_plots_{scenario}.png")
    cmap_name = 'Blues'
elif dataset == "Precip_increase":
    data_nc_path = r"C:\Thesis_Merwin\Forest_restoration\Evap_Change_Max_Reforestation\Evap_Change_Max_Reforestation.nc"
    save_png_path = os.path.join(save_dir, f"Precip_increase_plots_{scenario}.png")
    cmap_name = 'Blues'

df = pd.read_excel(excel_path, usecols=["HYBAS id", "Scenario", "Short name"])
df["HYBAS id"] = df["HYBAS id"].astype(int)
df["Scenario"] = df["Scenario"].str.strip().str.lower()

basin_names = df.drop_duplicates('HYBAS id').set_index('HYBAS id')['Short name'].to_dict()
hybas_ids = df.loc[df["Scenario"] == scenario, "HYBAS id"].tolist()
hybas_gdf = gpd.read_file(shapefile_path)


N = len(hybas_ids)
n_cols = 2
n_rows = math.ceil(N / 2)

if dataset == "Forestation":
    da_main = xr.open_dataset(data_nc_path)['restoration_potential']
else:
    da_main = xr.open_dataset(data_nc_path)['Evap_Change_Max_Reforestation']
    da_main = da_main.where(da_main > 0)

da_main = da_main.assign_coords(lon=(((da_main.lon + 180) % 360) - 180)).sortby('lon')

memory_data = {}
global_max = 0

# Load evaporation for fractional precipitation contribution
if dataset == "Precip_increase":
    total_evap_path = rf"C:\Thesis_Merwin\Forest_restoration\Total_Evaporation\Total_Evaporation_{scenario}.nc"
    with xr.open_dataset(total_evap_path) as da_total_ds:
        da_total = da_total_ds['ensemble_evapotranspiration']
        if 'latitude' in da_total.coords: da_total = da_total.rename({'latitude': 'lat'})
        da_total = da_total.assign_coords(lon=(((da_total.lon + 180) % 360) - 180)).sortby('lon')

for hybas_id in hybas_ids:
    hybas_id = int(hybas_id)

    target_basin = hybas_gdf[hybas_gdf['HYBAS_ID'] == hybas_id]
    basin_area_km2 = target_basin.to_crs(epsg=6933).area.sum() / 1e6

    footprint_path = rf"C:\Thesis_Merwin\UTrack_outputs\Maps\{scenario}\Evaporation_Footprint_{hybas_id}_{scenario}_Allocated.nc"

    with xr.open_dataset(footprint_path) as da_foot_ds:
        da_foot = da_foot_ds['yearly_avg_moisture']
        if 'latitude' in da_foot.coords: da_foot = da_foot.rename({'latitude': 'lat'})
        da_foot = da_foot.assign_coords(lon=(((da_foot.lon + 180) % 360) - 180)).sortby('lon')

        if dataset == "Forestation":
            masked_data = da_main.interp(lat=da_foot.lat, lon=da_foot.lon, method='nearest').where(da_foot > 0)
            lats, lons = masked_data.lat.values, masked_data.lon.values
            lat_res, lon_res = abs(lats[1] - lats[0]), abs(lons[1] - lons[0])
            lat_bounds_upper = np.radians(lats + lat_res / 2)
            lat_bounds_lower = np.radians(lats - lat_res / 2)
            area_per_lat_km2 = ((6371000.0 ** 2) * np.radians(lon_res) * np.abs(
                np.sin(lat_bounds_upper) - np.sin(lat_bounds_lower))) / 1e6
            cell_areas_km2 = np.tile(area_per_lat_km2[:, np.newaxis], (1, len(lons)))

            total_metric = np.nansum(cell_areas_km2 * (masked_data.values / 100.0))
            val1, val2 = round(total_metric, 0), None

        else:
            lats, lons = da_foot.lat.values, da_foot.lon.values
            lat_res, lon_res = abs(lats[1] - lats[0]), abs(lons[1] - lons[0])
            lat_bounds_upper = np.radians(lats + lat_res / 2)
            lat_bounds_lower = np.radians(lats - lat_res / 2)
            R = 6371000.0
            area_per_lat_km2 = ((R ** 2) * np.radians(lon_res) * np.abs(
                np.sin(lat_bounds_upper) - np.sin(lat_bounds_lower))) / 1e6
            cell_areas_km2 = np.tile(area_per_lat_km2[:, np.newaxis], (1, len(lons)))

            if dataset == "Evap_increase":
                masked_data = da_main.interp(lat=da_foot.lat, lon=da_foot.lon, method='nearest').where(da_foot > 0)
                volume_grid_km3 = (masked_data.values * cell_areas_km2) / 1e6
                total_volume_km3 = np.nansum(volume_grid_km3)
                basin_relative_depth_mm = (total_volume_km3 * 1e6) / basin_area_km2 if basin_area_km2 > 0 else 0

                val1, val2 = round(basin_relative_depth_mm, 2), round(total_volume_km3, 2)

            elif dataset == "Precip_increase":
                da_total_aligned = da_total.interp(lat=da_foot.lat, lon=da_foot.lon, method='nearest')
                fraction_map = (da_foot / da_total_aligned.where(da_total_aligned > 0)).fillna(0).clip(min=0, max=1)
                masked_evap = da_main.interp(lat=da_foot.lat, lon=da_foot.lon, method='nearest').where(da_foot > 0)
                masked_data = masked_evap * fraction_map

                volume_grid_km3 = (masked_data.values * cell_areas_km2) / 1e6
                total_volume_km3 = np.nansum(volume_grid_km3)
                basin_relative_depth_mm = (total_volume_km3 * 1e6) / basin_area_km2 if basin_area_km2 > 0 else 0

                val1, val2 = round(basin_relative_depth_mm, 3), round(total_volume_km3, 3)

        # Compute max for color bar
        local_max = float(masked_data.max())
        if not np.isnan(local_max) and local_max > global_max:
            global_max = local_max

        memory_data[hybas_id] = {
            "masked_data": masked_data.load(),
            "val1": val1,
            "val2": val2
        }

# Failsafe
    global_max = max(global_max, 0.001)

# Plotting
cmap = copy.copy(mpl.colormaps[cmap_name])
cmap.set_bad(color='white')

fig, axes = plt.subplots(
    nrows=n_rows, ncols=n_cols,
    figsize=(3.5 * n_cols, 2.8 * n_rows),
    subplot_kw={'projection': ccrs.PlateCarree()},
)

axes_flat = axes.flatten() if N > 1 else np.array([axes])

fig.subplots_adjust(bottom=0.05, top=0.90, left=0.05, right=0.88, hspace=0.4, wspace=0.05)

for idx, hybas_id in enumerate(hybas_ids):
    hybas_id = int(hybas_id)
    ax = axes_flat[idx]

    basin_data = memory_data[hybas_id]
    masked_data = basin_data["masked_data"]
    val1 = basin_data["val1"]

    target_basin = hybas_gdf[hybas_gdf['HYBAS_ID'] == hybas_id]
    basin_minx, basin_miny, basin_maxx, basin_maxy = target_basin.total_bounds

    mask_threshold = 1.0 if dataset == "Forestation" else 0.01
    extent_mask = masked_data.where(masked_data >= mask_threshold, drop=True)
    if extent_mask.size == 0:
        extent_mask = masked_data.where(masked_data > 0, drop=True)

    foot_minx, foot_maxx = float(extent_mask.lon.min()), float(extent_mask.lon.max())
    foot_miny, foot_maxy = float(extent_mask.lat.min()), float(extent_mask.lat.max())

    combined_minx = min(foot_minx, basin_minx)
    combined_maxx = max(foot_maxx, basin_maxx)
    combined_miny = min(foot_miny, basin_miny)
    combined_maxy = max(foot_maxy, basin_maxy)

    center_lon = (combined_maxx + combined_minx) / 2
    center_lat = (combined_maxy + combined_miny) / 2
    local_span_lon = (combined_maxx - combined_minx) + 5
    local_span_lat = (combined_maxy - combined_miny) + 5

    target_ratio = 4.0 / 3.0
    if (local_span_lon / local_span_lat) < target_ratio:
        local_span_lon = local_span_lat * target_ratio
    else:
        local_span_lat = local_span_lon / target_ratio

    minx = center_lon - (local_span_lon / 2)
    maxx = center_lon + (local_span_lon / 2)
    miny = max(-90, center_lat - (local_span_lat / 2))
    maxy = min(90, center_lat + (local_span_lat / 2))

    # Dynamic plotting
    if dataset == "Forestation":
        img = masked_data.plot.imshow(
            ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=0, vmax=global_max,
            add_colorbar=False, add_labels=False, zorder=1
        )
    elif dataset == "Evap_increase":
        img = masked_data.plot.imshow(
            ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
            cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=global_max),
            add_colorbar=False, add_labels=False, zorder=1
        )
    elif dataset == "Precip_increase":
        img = masked_data.plot.imshow(
            ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
            cmap=cmap, norm=mcolors.LogNorm(vmin=0.01, vmax=global_max),
            add_colorbar=False, add_labels=False, zorder=1
        )

    local_gdf = hybas_gdf.cx[minx:maxx, miny:maxy]
    local_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.2, alpha=0.5,
                   transform=ccrs.PlateCarree(), zorder=2)
    target_basin.boundary.plot(ax=ax, color='#00AA00', linewidth=1, zorder=3)

    #Specific adaptation for Greenland
    if hybas_id == 9030005280:
        minx, maxx = -65, -30
        miny, maxy = 38, 72

    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())
    ax.set_box_aspect(3.0 / 4.0)

    def format_lon(x, pos):
        return f"{int(x)}°E" if x > 0 else (f"{int(-x)}°W" if x < 0 else "0°")

    def format_lat(y, pos):
        return f"{int(y)}°N" if y > 0 else (f"{int(-y)}°S" if y < 0 else "0°")

    if hybas_id == 9030005280:
        dynamic_x_ticks = [x for x in range(-180, 181, 30) if minx <= x <= maxx]
        dynamic_y_ticks = [y for y in range(-90, 91, 10) if miny <= y <= maxy]
    else:
        dynamic_x_ticks = [x for x in range(-180, 181, 10) if minx <= x <= maxx]
        dynamic_y_ticks = [y for y in range(-90, 91, 10) if miny <= y <= maxy]

    ax.set_xticks(dynamic_x_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(dynamic_y_ticks, crs=ccrs.PlateCarree())
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.set_xlabel("")
    ax.set_ylabel("")

    short_name = basin_names.get(hybas_id, str(hybas_id))

    # Dynamic Titles
    if dataset == "Forestation":
        ax.set_title(f"{short_name}\n Total FP: {val1:,.0f} km²", fontsize=10, pad=3)
    elif dataset == "Evap_increase":
        ax.set_title(f"{short_name}\n Total Evap Inc: {val1:,.2f} mm/yr", fontsize=9, pad=3)
    elif dataset == "Precip_increase":
        ax.set_title(f"{short_name}\n Total Precip Inc: {val1:,.2f} mm/yr", fontsize=9, pad=3)

    # abc labels
    letter_label = chr(97 + idx)
    ax.text(0.09, 0.96, letter_label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', ha='right', va='top',
            zorder=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

for idx in range(N, len(axes_flat)):
    axes_flat[idx].set_visible(False)

# Centre uneven plot
if N % 2 != 0 and N > 1:
    ax_last = axes_flat[N - 1]
    pos_left = axes[0, 0].get_position()
    pos_right = axes[0, 1].get_position()
    shift = (pos_right.x0 - pos_left.x0) / 2
    pos = ax_last.get_position()
    ax_last.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])

# Dynamic colorbar and titels
cax = fig.add_axes([0.88, 0.15, 0.015, 0.65])
cbar = fig.colorbar(img, cax=cax, orientation='vertical')

ssp_label = f"{scenario[:4].upper()}-{scenario[4]}.{scenario[5]}"

if dataset == "Forestation":
    cbar.set_label('Forestation Potential (%)', fontweight='bold', fontsize=12, x=0.9)

elif dataset == "Evap_increase":
    ticks = [10 ** i for i in range(-1, math.floor(math.log10(global_max)) + 1)]
    top_tick = int(global_max)

    if top_tick > ticks[-1]:
        next_tick = ticks[-1] * 10
        distance_to_next = next_tick - ticks[-1]
        # Do not plot top tick when within 1/10 of next tick
        if (top_tick - ticks[-1]) > (distance_to_next / 10.0):
            ticks.append(top_tick)

    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    cbar.set_label('Evaporation Increase (mm/year)', fontweight='bold', fontsize=12)

elif dataset == "Precip_increase":
    ticks = [10 ** i for i in range(-2, math.floor(math.log10(global_max)) + 1)]

    top_tick = int(global_max) if global_max >= 1 else round(global_max, 2)

    if top_tick > ticks[-1]:
        next_tick = ticks[-1] * 10
        distance_to_next = next_tick - ticks[-1]
        if (top_tick - ticks[-1]) > (distance_to_next / 10.0):
            ticks.append(top_tick)

    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    cbar.set_label('Precipitation Increase (mm/year)', fontweight='bold', fontsize=12)

plt.show(block=False)
plt.pause(0.1)

save_choice = input(f"\n[{scenario.upper()}] done. Type 'ok' to save: ").strip().lower()

if save_choice == 'ok':
    plt.savefig(save_png_path, dpi=400, bbox_inches='tight')

plt.close()
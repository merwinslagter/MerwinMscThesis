import os
import xarray as xr
import numpy as np
from numba import njit
import rioxarray as rio
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import copy
from matplotlib.ticker import MultipleLocator, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

models = ['CanESM5-1', 'EC-Earth3', 'MIROC6', 'MPI-ESM1-2-HR', 'NorESM2-MM']
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

# Selected scenario
scenario = "ssp585"

shapefile_path = r"C:\Thesis_Merwin\HydroBASINS\HydroBASIN_Global_Level3.shp"

#Safe paths
save_dir = rf"C:\Thesis_Merwin\Forest_restoration\Total_Evaporation"
os.makedirs(save_dir, exist_ok=True)
save_nc_path = os.path.join(save_dir, f"Total_Evaporation_{scenario}.nc")
save_png_path = os.path.join(save_dir, f"Total_Evaporation_{scenario}_visual.png")

@njit
def convert_hfls_to_evap(hfls, tas):
    L_ref = 2502.2
    L_slope = -2.4337
    tas_C = tas - 273.15
    L = (L_slope * tas_C + L_ref) * 1000
    evap_from_hfls = hfls / L
    return evap_from_hfls

master_grid_file = rf"C:\Thesis_Merwin\CMIP6\Forcing_data_EC-Earth3\{scenario}\tas\tas_{scenario}_EC-Earth3_20452054.nc"
with xr.open_dataset(master_grid_file) as data_master:
    ref_lats = data_master['lat'].values
    ref_lons = data_master['lon'].values

model_evap_grids = []

for model in models:
    print(f"Verwerken van model: {model}...")
    hfls_file = rf"C:\Thesis_Merwin\CMIP6\Forcing_data_{model}\{scenario}\hfls\hfls_{scenario}_{model}_20452054.nc"
    tas_file = rf"C:\Thesis_Merwin\CMIP6\Forcing_data_{model}\{scenario}\tas\tas_{scenario}_{model}_20452054.nc"

    hfls_data = xr.open_dataset(hfls_file)
    tas_data = xr.open_dataset(tas_file)

    hfls_squeeze = hfls_data['hfls'].squeeze()
    tas_squeeze = tas_data['tas'].squeeze()

    hfls_values = hfls_squeeze.values
    tas_values = tas_squeeze.values

    evspsbl_values = convert_hfls_to_evap(hfls_values, tas_values)

    evspsbl = xr.DataArray(
        evspsbl_values,
        coords=hfls_squeeze.coords,
        dims=hfls_squeeze.dims,
        name="evspsbl"
    )

    evspsbl_mm = evspsbl * 86400
    evspsbl_mm = evspsbl_mm.sortby("lon").rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    evspsbl_yearly = evspsbl_mm.resample(time="1YE").sum()
    evspsbl_yearly = evspsbl_yearly.sortby("lat", ascending=False)
    evspsbl_yearly_mean = evspsbl_yearly.mean(dim="time")
    evspsbl_yearly_mean_interp = evspsbl_yearly_mean.interp(lat=ref_lats, lon=ref_lons, method='linear')
    model_evap_grids.append(evspsbl_yearly_mean_interp)

# Ensemble average
multi_model_evap = xr.concat(model_evap_grids, dim="model").mean(dim="model")

# Rolling
multi_model_evap = multi_model_evap.assign_coords(lon=(((multi_model_evap.lon + 180) % 360) - 180)).sortby('lon')

multi_model_evap.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
multi_model_evap.rio.write_crs("EPSG:4326", inplace=True)
hybas_gdf = gpd.read_file(shapefile_path)

# Clip
evap_masked = multi_model_evap.rio.clip(hybas_gdf.geometry, hybas_gdf.crs, drop=False, all_touched=True)
evap_masked = evap_masked.where(evap_masked > 0)
max_val = float(evap_masked.max())

def format_lon(x, pos):
    if x == 0: return "0°"
    if x == 180 or x == -180: return "180°"
    return f"{int(x)}° E" if x > 0 else f"{int(-x)}° W"

def format_lat(y, pos):
    if y == 0: return "0°"
    return f"{int(y)}° N" if y > 0 else f"{int(-y)}° S"

# Plotting
cmap = copy.copy(mpl.colormaps['Blues'])
cmap.set_bad(color='white')

fig = plt.figure(figsize=(15, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

img = evap_masked.plot.pcolormesh(
    ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
    cmap=cmap, vmin=0, vmax=max_val,
    add_colorbar=False, add_labels=False, zorder=1
)

# Custom color bar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
cbar = fig.colorbar(img, cax=cax)

cbar.ax.yaxis.set_major_locator(MultipleLocator(250))
cbar.set_label('Evapotranspiration (mm/year)', fontweight='bold')

hybas_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.2, alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

# Extent & padding
minx, miny, maxx, maxy = hybas_gdf.total_bounds
lat_padding = 2
padded_miny = max(miny - lat_padding, -90)
padded_maxy = min(maxy + lat_padding, 90)
ax.set_extent([minx, maxx, padded_miny, padded_maxy], crs=ccrs.PlateCarree())


all_y_ticks = range(-90, 91, 30)
dynamic_y_ticks = [y for y in all_y_ticks if y >= padded_miny and y <= padded_maxy]

ax.set_xticks(range(-180, 181, 60), crs=ccrs.PlateCarree())
ax.set_yticks(dynamic_y_ticks, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

ax.set_xlabel("")
ax.set_ylabel("")

title_text = f"Annual Mean Evapotranspiration - {scenario}"
ax.set_title(title_text, fontsize=20, pad=12, fontweight='bold')

plt.show(block=False)
plt.pause(0.1)

save_choice = input("\n Type 'ok' to safe: ").strip().lower()

if save_choice == 'ok':
    evap_save = evap_masked.copy()
    evap_save = evap_save.assign_coords(lon=(evap_save.lon % 360)).sortby('lon')

    evap_save.name = "ensemble_evapotranspiration"
    evap_save.to_netcdf(save_nc_path)

    plt.savefig(save_png_path, dpi=400, bbox_inches='tight')

plt.close()

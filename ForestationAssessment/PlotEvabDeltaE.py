import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import copy
import geopandas as gpd
import rioxarray
from rasterio.enums import Resampling
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm

# Apply regridding or not
APPLY_REGRIDDING = False

# Data paths
nc_path = r"C:\Thesis_Merwin\Forest_restoration\marginalETJoyce.nc"
shapefile_path = r"C:\Thesis_Merwin\HydroBASINS\HydroBASIN_Global_Level3.shp"
nc_template_path = r"C:\Thesis_Merwin\UTrack_outputs\ssp245\7030008720_output\UTrack-Backward-EC-Earth3_ssp245_31-12-2054_16-1-2045_7030008720.nc"

if APPLY_REGRIDDING:
    save_nc_path = r"C:\Thesis_Merwin\Forest_restoration\Delta_Evaporation_Forest\DeltaEvap_regridded_min.nc"
    save_png_path = r"C:\Thesis_Merwin\Forest_restoration\Delta_Evaporation_Forest\DeltaEvap_regridded_visual_min.png"
else:
    save_nc_path = r"C:\Thesis_Merwin\Forest_restoration\Delta_Evaporation_Forest\DeltaEvap_original.nc"
    save_png_path = r"C:\Thesis_Merwin\Forest_restoration\Delta_Evaporation_Forest\DeltaEvap_original_visual.png"

ds = xr.open_dataset(nc_path)
ds = ds.assign_coords({"lon": ds.longitude, "lat": ds.latitude, "time": ds.t})
ds.coords['lon'] = ds.coords['lon'] % 360
ds = ds.sortby('lon')

yearly_deltaE = (ds['deltaE_min'].sum(dim='time') * 30.436875) / 100

if APPLY_REGRIDDING:
    ds_template = xr.open_dataset(nc_template_path)
    ds_template = ds_template.rename({'lat': 'y', 'lon': 'x'})
    ds_template.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    ds_template.rio.write_crs("EPSG:4326", inplace=True)
    if ds_template.x.max() > 180:
        ds_template.coords['x'] = (ds_template.coords['x'] + 180) % 360 - 180
        ds_template = ds_template.sortby(ds_template.x)

    yearly_deltaE = yearly_deltaE.rename({'lat': 'y', 'lon': 'x'})
    yearly_deltaE.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    yearly_deltaE.rio.write_crs("EPSG:4326", inplace=True)
    yearly_deltaE = yearly_deltaE.rio.reproject_match(ds_template, resampling=Resampling.average)
    yearly_deltaE = yearly_deltaE.rename({'y': 'lat', 'x': 'lon'})

# Clipping
temp_data = yearly_deltaE.assign_coords(lon=(((yearly_deltaE.lon + 180) % 360) - 180)).sortby('lon')
temp_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
temp_data.rio.write_crs("EPSG:4326", inplace=True)

hybas_gdf = gpd.read_file(shapefile_path)
yearly_deltaE_masked = temp_data.rio.clip(hybas_gdf.geometry, hybas_gdf.crs, drop=False, all_touched=True)

def format_lon(x, pos):
    if x == 0: return "0°"
    if x == 180 or x == -180: return "180°"
    return f"{int(x)}° E" if x > 0 else f"{int(-x)}° W"

def format_lat(y, pos):
    if y == 0: return "0°"
    return f"{int(y)}° N" if y > 0 else f"{int(-y)}° S"

# Plotting

# Diverging colormap
cmap = copy.copy(mpl.colormaps['RdBu'])
cmap.set_bad(color='white')

fig = plt.figure(figsize=(15, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

min_val = float(yearly_deltaE_masked.min())
max_val = float(yearly_deltaE_masked.max())

# white at 0
norm = TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)

img = yearly_deltaE_masked.plot.pcolormesh(
    ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
    cmap=cmap,
    norm=norm,
    add_colorbar=False, add_labels=False, zorder=1
)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
cbar = fig.colorbar(img, cax=cax)


cbar.set_label('Annual Evaporation Change (mm per % forest increase)', fontweight='bold')

# Plot basin boundaries
hybas_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.2, alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

# Extent & Padding
minx, miny, maxx, maxy = hybas_gdf.total_bounds
lat_padding = 2
padded_miny = max(miny - lat_padding, -90)
padded_maxy = min(maxy + lat_padding, 90)
ax.set_extent([minx, maxx, padded_miny, padded_maxy], crs=ccrs.PlateCarree())

plt.show(block=False)
plt.pause(0.1)

save_choice = input("\nTyp 'ok' to save: ").strip().lower()
if save_choice == 'ok':
    to_save = yearly_deltaE_masked.assign_coords(lon=(yearly_deltaE_masked.lon % 360)).sortby('lon')
    to_save.name = "yearly_sum_deltaE"
    to_save.to_netcdf(save_nc_path)
    plt.savefig(save_png_path, dpi=400, bbox_inches='tight')

plt.close()

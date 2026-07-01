
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import numpy as np
import copy
import geopandas as gpd
import rioxarray
from rasterio.enums import Resampling
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")

# Apply regridding or not
APPLY_REGRIDDING = True

# Data paths
tif_path = r"C:\Thesis_Merwin\Forest_restoration\Forest_restoration_potential\Restoration_potential.tif"
shapefile_path = r"C:\Thesis_Merwin\HydroBASINS\HydroBASIN_Global_Level3.shp"
nc_template_path = r"C:\Thesis_Merwin\UTrack_outputs\ssp245\7030008720_output\UTrack-Backward-EC-Earth3_ssp245_31-12-2054_16-1-2045_7030008720.nc"

if APPLY_REGRIDDING:
    save_nc_path = r"C:\Thesis_Merwin\Forest_restoration\Forest_restoration_potential\Restoration_potential_regridded.nc"
    save_png_path = r"C:\Thesis_Merwin\Forest_restoration\Forest_restoration_potential\Restoration_potential_regridded.nc"
else:
    save_nc_path = r"C:\Thesis_Merwin\Forest_restoration\Forest_restoration_potential\Restoration_potential_original.nc"
    save_png_path = r"C:\Thesis_Merwin\Forest_restoration\Forest_restoration_potential\Restoration_potential_original_visual.png"

da_tif = rioxarray.open_rasterio(tif_path, mask_and_scale=True)
if 'band' in da_tif.dims:
    da_tif = da_tif.squeeze('band', drop=True)
da_tif = da_tif.fillna(0)
da_tif.name = "restoration_potential"

da_tif.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

if da_tif.rio.crs is None:
    da_tif.rio.write_crs("EPSG:4326", inplace=True)

if APPLY_REGRIDDING:
    ds_template = xr.open_dataset(nc_template_path)
    ds_template = ds_template.rename({'lat': 'y', 'lon': 'x'})

    target_x = ds_template.x.values
    target_y = ds_template.y.values

    if target_x.max() > 180:
        target_x = (target_x + 180) % 360 - 180

    x_sort_idx = np.argsort(target_x)
    target_x = target_x[x_sort_idx]

    dummy_grid = xr.DataArray(
        np.zeros((len(target_y), len(target_x))),
        dims=("y", "x"),
        coords={"y": target_y, "x": target_x}
    )
    dummy_grid.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    dummy_grid.rio.write_crs("EPSG:4326", inplace=True)

    da_tif = da_tif.rio.reproject_match(dummy_grid, resampling=Resampling.average)

# Clipping
hybas_gdf = gpd.read_file(shapefile_path)
hybas_gdf = hybas_gdf.to_crs("EPSG:4326")
da_tif.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
da_tif.rio.write_crs("EPSG:4326", inplace=True)

da_clipped = da_tif.rio.clip(
    hybas_gdf.geometry,
    hybas_gdf.crs,
    drop=False,
    all_touched=True
)

da_clipped = da_clipped.rename({'x': 'lon', 'y': 'lat'})
max_val = float(da_clipped.max())

def format_lon(x, pos):
    if x == 0: return "0°"
    if x == 180 or x == -180: return "180°"
    return f"{int(x)}° E" if x > 0 else f"{int(-x)}° W"

def format_lat(y, pos):
    if y == 0: return "0°"
    return f"{int(y)}° N" if y > 0 else f"{int(-y)}° S"

# Plotting
cmap = copy.copy(mpl.colormaps['YlGn'])
cmap.set_bad(color='white')

fig = plt.figure(figsize=(15, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

img = da_clipped.plot.imshow(
    ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
    cmap=cmap, vmin=0, vmax=max_val,
    add_colorbar=False, add_labels=False, zorder=1
)

# Custom color bar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
cbar = fig.colorbar(img, cax=cax)
cbar.set_label('Maximum Forestation Potential (%)', fontweight='bold')

# Plot basin boundaries
hybas_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.2, alpha=0.5, transform=ccrs.PlateCarree(),
               zorder=2)

# Extent & Padding
minx, miny, maxx, maxy = hybas_gdf.total_bounds
lat_padding = 3
padded_miny = max(miny - lat_padding, -90)
padded_maxy = min(maxy + lat_padding, 90)
ax.set_extent([minx, maxx, padded_miny, padded_maxy], crs=ccrs.PlateCarree())

plt.show(block=False)
plt.pause(0.1)

save_choice = input("\nType 'ok' to save: ").strip().lower()

if save_choice == 'ok':
    to_save = da_clipped.assign_coords(lon=(da_clipped.lon % 360)).sortby('lon')

    if 'spatial_ref' in to_save.coords:
        to_save = to_save.drop_vars('spatial_ref')

    to_save.to_netcdf(save_nc_path)
    plt.savefig(save_png_path, dpi=400, bbox_inches='tight')

plt.close()
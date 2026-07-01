import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import copy
import geopandas as gpd
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

warnings.filterwarnings("ignore")

# Data paths
et_nc_path = r"C:\Thesis_Merwin\Forest_restoration\Delta_Evaporation_Forest\DeltaEvap_regridded_min.nc"
pot_nc_path = r"C:\Thesis_Merwin\Forest_restoration\Forest_restoration_potential\Restoration_potential_regridded.nc"
shapefile_path = r"C:\Thesis_Merwin\HydroBASINS\HydroBASIN_Global_Level3.shp"

save_nc_path = r"C:\Thesis_Merwin\Forest_restoration\Evap_Change_Max_Reforestation\Evap_Change_Max_Reforestation.nc"
save_png_path = r"C:\Thesis_Merwin\Forest_restoration\Evap_Change_Max_Reforestation\Evap_Change_Max_Reforestation_visual.png"

da_et = xr.open_dataset(et_nc_path)['yearly_sum_deltaE']
da_pot = xr.open_dataset(pot_nc_path)['restoration_potential']

combined_change = da_et * da_pot
combined_change = combined_change.assign_coords(lon=(((combined_change.lon + 180) % 360) - 180)).sortby('lon')

def format_lon(x, pos):
    if x == 0: return "0°"
    if x == 180 or x == -180: return "180°"
    return f"{int(x)}° E" if x > 0 else f"{int(-x)}° W"

def format_lat(y, pos):
    if y == 0: return "0°"
    return f"{int(y)}° N" if y > 0 else f"{int(-y)}° S"

# Plotting
cmap = copy.copy(mpl.colormaps['RdBu'])
cmap.set_bad(color='white')

fig = plt.figure(figsize=(15, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

min_val = float(combined_change.min())
max_val = float(combined_change.max())
abs_max = max(abs(min_val), abs(max_val))

img = combined_change.plot.pcolormesh(
    ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
    cmap=cmap,
    vmin=-abs_max, vmax=abs_max,
    add_colorbar=False, add_labels=False, zorder=1
)

# Colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
cbar = fig.colorbar(img, cax=cax)
cbar.set_label('Projected change in evaporation (mm/year)', fontweight='bold')

# Basin boundaries
hybas_gdf = gpd.read_file(shapefile_path)
hybas_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.2, alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)

# Extent and padding
minx, miny, maxx, maxy = hybas_gdf.total_bounds
lat_padding = 2
padded_miny = max(miny - lat_padding, -90)
padded_maxy = min(maxy + lat_padding, 90)
ax.set_extent([minx, maxx, padded_miny, padded_maxy], crs=ccrs.PlateCarree())

# Dynamic ticks
all_y_ticks = range(-90, 91, 30)
dynamic_y_ticks = [y for y in all_y_ticks if y >= padded_miny and y <= padded_maxy]

ax.set_xticks(range(-180, 181, 60), crs=ccrs.PlateCarree())
ax.set_yticks(dynamic_y_ticks, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

ax.set_title("Change in Evaporation with Reforesting Maximum Potential", fontsize=18, pad=10, fontweight='bold')

plt.show(block=False)
plt.pause(0.1)

save_choice = input("\nType 'ok' to save: ").strip().lower()
if save_choice == 'ok':
    to_save = combined_change.assign_coords(lon=(combined_change.lon % 360)).sortby('lon')
    to_save.name = "Evap_Change_Max_Reforestation"
    to_save.to_netcdf(save_nc_path)
    plt.savefig(save_png_path, dpi=400, bbox_inches='tight')

plt.close()
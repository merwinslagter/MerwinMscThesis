import os
from netCDF4 import Dataset
import numpy as np
import xarray as xr
from numba import njit, prange
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import box, mapping
from shapely.ops import unary_union
from shapely.affinity import translate
from affine import Affine

parcel_data_columns = ['latitude', 'longitude', 'level', 'moisture_present', 'original_moisture', 'time', 'start_time',
                       'startlatidx',
                       'startlonidx', 'current_latidx', 'current_lonidx']

LAT_IDX = parcel_data_columns.index('latitude')
LON_IDX = parcel_data_columns.index('longitude')
LEVEL_IDX = parcel_data_columns.index('level')
MOISTURE_IDX = parcel_data_columns.index('moisture_present')
ORIGINAL_MOISTURE_IDX = parcel_data_columns.index('original_moisture')
TIME_IDX = parcel_data_columns.index('time')
START_TIME_IDX = parcel_data_columns.index('start_time')
START_LATIDX = parcel_data_columns.index('startlatidx')
START_LONIDX = parcel_data_columns.index('startlonidx')
CURRENT_LATIDX = parcel_data_columns.index('current_latidx')
CURRENT_LONIDX = parcel_data_columns.index('current_lonidx')

def create_nc_file(outputfn, output_directory, forward_tracking, model_choice, scenario, start_date, release_end_date, m, output_array, total_released):

    footprint_type = 'evaporation' if forward_tracking else 'precipitation'

    os.makedirs(output_directory, exist_ok=True)

    output_sum = np.sum(output_array)
    fraction = output_sum / total_released if total_released > 0 else 0
    footprint_mm = np.zeros_like(output_array)
    np.divide(output_array, m.surface_areas, out=footprint_mm, where=m.surface_areas != 0)

    with Dataset(outputfn, mode="w", format="NETCDF4") as output:
        output.title = f"{footprint_type} footprints"
        output.model = model_choice
        output.scenario = scenario
        output.simulation_period = f"{start_date.strftime('%Y-%m-%d')} to {release_end_date.strftime('%Y-%m-%d')}"
        output.total_released_moisture = total_released
        output.allocation_fraction = fraction

        # Create dimensions
        output.createDimension("lat", len(m.lats))
        output.createDimension("lon", len(m.lons))

        # Create variables
        lat = output.createVariable("lat", np.float64, ("lat",))
        lat.units = "degrees_north"
        lat.long_name = "latitude"
        lat[:] = m.lats

        lon = output.createVariable("lon", np.float64, ("lon",))
        lon.units = "degrees_east"
        lon.long_name = "longitude"
        lon[:] = m.lons

        footprint = output.createVariable("footprint", np.float64, ("lat", "lon"), zlib=True)
        footprint.units = "mm"
        footprint.long_name = f"Accumulated {footprint_type} depth in mm"
        footprint[:, :] = footprint_mm

    return outputfn

def check_file_availability(directory, start_date, end_date, model, scenario):

    variables = ['pr', 'prw', 'tas', 'hfls', 'wap', 'ua', 'va', 'hus', 'sftlf']

    # Create a set of all required years
    required_years = set(range(start_date.year, end_date.year + 1))

    # Dictionary to track available years for each variable
    available_years = {var: set() for var in variables}

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if model in filename and filename.endswith('.nc'):
                    if 'sftlf' in filename:
                        available_years['sftlf'] = required_years
                        continue

                    if scenario in filename:
                        date_range = filename.split('_')[-1].replace('.nc', '')
                        file_start_year = int(date_range[:4])
                        file_end_year = int(date_range[4:])

                        for variable in variables:
                            if f"{variable}_" in filename:
                                available_years[variable].update(range(file_start_year, file_end_year + 1))

                    continue

    missing_files_details = {}
    missing_dates = {}
    for variable in variables:
        missing_dates[variable] = sorted(list(required_years - available_years[variable]))
        if missing_dates[variable]:
            missing_files_details[variable] = missing_dates[variable]

    if missing_files_details:
        error_message = "Not all files available for the requested simulation:\n"
        for var, missing in missing_files_details.items():
            error_message += f"Missing files for {var}: {missing}\n"
        raise FileNotFoundError(error_message.strip())

@njit
def get_level_index(level):

    if level > 92500: return 0
    if level > 77500: return 1
    if level > 60000: return 2
    if level > 37500: return 3
    if level > 17500: return 4
    if level > 7500: return 5
    if level > 3000: return 6
    return 7

@njit
def get_nearest_index(lat, lon, lats, lons):
    if lat > 90:
        lat = 90 - (lat - 90)
        lon += 180
    elif lat < -90:
        lat = -90 - (lat + 90)
        lon += 180

    lon = lon % 360

    min_lat_diff = 360.0
    lat_idx = 0
    for i in range(len(lats)):
        diff = abs(lats[i] - lat)
        if diff < min_lat_diff:
            min_lat_diff = diff
            lat_idx = i

    min_lon_diff = 360.0
    lon_idx = 0
    for i in range(len(lons)):
        diff = abs(lons[i] - lon)
        if diff > 180.0:
            diff = 360.0 - diff

        if diff < min_lon_diff:
            min_lon_diff = diff
            lon_idx = i

    return lat_idx, lon_idx

@njit
def position_new_parcels(num_parcels, final_lats, final_lons, m_lats, m_lons, m_hus, m_levels):
    lat_indices = np.zeros(num_parcels, dtype=np.int32)
    lon_indices = np.zeros(num_parcels, dtype=np.int32)
    levels = np.zeros(num_parcels, dtype=np.float64)

    for i in range(num_parcels):
        lidx, coidx = get_nearest_index(final_lats[i], final_lons[i], m_lats, m_lons)
        lat_indices[i] = lidx
        lon_indices[i] = coidx
        levels[i] = get_starting_level(lidx, coidx, m_hus, m_levels)

    return lat_indices, lon_indices, levels

def create_parcels_mask(mask, parcel_start_time, delta_t, m, parcels_per_mm, forward_tracking, parcels_data_columns, surface_area_weighting):

    moisture_rate_mask = (np.maximum(m.evspsbl, 0) if forward_tracking else np.maximum(m.pr, 0)) * mask

    total_moisture_mass = np.sum(moisture_rate_mask * m.surface_areas) * delta_t.total_seconds()
    if total_moisture_mass <= 0:
        return np.empty((0, len(parcels_data_columns)))

    num_parcels = int(round(total_moisture_mass * parcels_per_mm))
    if num_parcels <= 0:
        return np.empty((0, len(parcels_data_columns)))

    parcel_moisture = total_moisture_mass / num_parcels

    weighted_probability = (moisture_rate_mask * m.surface_areas) if surface_area_weighting else moisture_rate_mask
    total_weighted_prob = np.sum(weighted_probability)
    if total_weighted_prob == 0:
        return np.empty((0, len(parcels_data_columns)))

    flat_indices = np.random.choice(weighted_probability.size, num_parcels, p=(weighted_probability / total_weighted_prob).ravel())
    rows, cols = np.unravel_index(flat_indices, weighted_probability.shape)

    lat_bounds_deg = np.degrees(m.lat_bounds)
    lon_bounds_deg = np.degrees(m.lon_bounds)

    lat_widths = np.abs(lat_bounds_deg[rows + 1] - lat_bounds_deg[rows])
    lon_widths = np.abs(lon_bounds_deg[cols + 1] - lon_bounds_deg[cols])

    final_lats = m.lats[rows] + (np.random.random(num_parcels) - 0.5) * lat_widths
    final_lons = m.lons[cols] + (np.random.random(num_parcels) - 0.5) * lon_widths

    lat_indices, lon_indices, levels = position_new_parcels(
        num_parcels, final_lats, final_lons, m.lats, m.lons, m.hus, m.levels
    )

    parcels_data = np.zeros((num_parcels, len(parcels_data_columns)))

    parcels_data[:, LAT_IDX] = final_lats
    parcels_data[:, LON_IDX] = final_lons
    parcels_data[:, LEVEL_IDX] = levels
    parcels_data[:, MOISTURE_IDX] = parcel_moisture
    parcels_data[:, ORIGINAL_MOISTURE_IDX] = parcel_moisture
    parcels_data[:, TIME_IDX] = parcel_start_time.timestamp()
    parcels_data[:, START_TIME_IDX] = parcel_start_time.timestamp()
    parcels_data[:, START_LATIDX] = lat_indices
    parcels_data[:, START_LONIDX] = lon_indices
    parcels_data[:, CURRENT_LATIDX] = lat_indices
    parcels_data[:, CURRENT_LONIDX] = lon_indices

    return parcels_data

@njit
def convert_hfls_to_evap(hfls, tas):
    # Constants
    L_ref = 2502.2  # Latent heat of vaporization at 0°C in kJ/kg
    L_slope = -2.4337  # Change in L per °C in kJ/kg

    tas_C = tas - 273.15  # convert to Celsius for the formula

    # Calculate latent heat of vaporization (L) based on T
    L = (L_slope * tas_C + L_ref) * 1000  # convert L to J/kg

    # Calculate evaporation from latent heat flux
    evap_from_hfls = hfls / L  # Result in kg m-2 s-1 just like evspsbl
    return evap_from_hfls

@njit
def get_starting_level(latidx, lonidx, hus, levels):
    H_sum = 0.0
    for i in range(len(levels)):
        val = hus[i, latidx, lonidx]
        if val > 0:
            H_sum += val

    if H_sum == 0:
        return 95000.0  # Default level if no humidity

    frac = np.random.random() * H_sum  # Random fraction of total humidity
    count = 0

    for i in range(len(levels)):
        if hus[i, latidx, lonidx] > 0:
            count += hus[i, latidx, lonidx]
            if count > frac:
                return levels[i]
    return 95000.0  # Fallback

@njit(parallel=True, nogil=True)
def update_parcels(parcels_data, delta_t_sec, m_ua, m_va, m_wap, m_pr, m_evspsbl, m_prw, m_hus, m_levels, m_lats, m_lons, lons_min, m_degreelength_lat, m_degreelength_lon, forward_tracking, output_array):
    n_parcels = parcels_data.shape[0]

    for i in prange(n_parcels):
        lat = parcels_data[i, LAT_IDX]
        lon = parcels_data[i, LON_IDX]
        level = parcels_data[i, LEVEL_IDX]
        moisture_present = parcels_data[i, MOISTURE_IDX]
        time = parcels_data[i, TIME_IDX]

        levidx = get_level_index(level)
        latidx, lonidx = get_nearest_index(lat, lon, m_lats, m_lons)

        w = m_wap[levidx, latidx, lonidx]
        while np.abs(w) > 10:
            level -= 5000
            levidx = get_level_index(level)
            w = m_wap[levidx, latidx, lonidx]

        u = m_ua[levidx, latidx, lonidx]
        v = m_va[levidx, latidx, lonidx]

        if not forward_tracking:
            u, v, w = -u, -v, -w

        d_lon = (delta_t_sec * u / m_degreelength_lon[latidx])
        d_lat = (delta_t_sec * v / m_degreelength_lat[latidx])

        if np.isfinite(d_lon) and np.isfinite(d_lat): # Prevents infinite parcel movement (teleportation)
            lon += d_lon
            lat += d_lat

        # --- Pole Crossing Physics ---
        # Handle pole crossings
        if lat > 90:
            lat = 90 - (lat - 90)
            lon += 180
        if lat < -90:
            lat = -90 - (lat + 90)
            lon += 180

        # Wrap input longitude to [-180, 180]
        lon = ((lon + 180) % 360) - 180

        # Adjust longitude based on the target array range
        if lon < 0:
            if lons_min >= 0:
                lon = lon % 360

        level += (delta_t_sec * w)

        if level < 1000: level = 1000 #Pa
        if level > 100000: level = 100000 #Pa

        if np.random.random() * 24 < (delta_t_sec / 3600):
            latidx, lonidx = get_nearest_index(lat, lon, m_lats, m_lons)
            level = get_starting_level(latidx, lonidx, m_hus, m_levels)

        P = m_pr[latidx, lonidx] * delta_t_sec
        E = m_evspsbl[latidx, lonidx] * delta_t_sec
        PW = m_prw[latidx, lonidx]

        if forward_tracking:
            fraction_allocated = P / PW if PW > 0 and P > 0 else 0
        else:
            fraction_allocated = E / PW if PW > 0 and E > 0 else 0

        allocated = fraction_allocated * moisture_present
        moisture_present -= allocated

        outlatidx, outlonidx = get_nearest_index(lat, lon, m_lats, m_lons)
        output_array[outlatidx, outlonidx] += allocated

        parcels_data[i, LAT_IDX] = lat
        parcels_data[i, LON_IDX] = lon
        parcels_data[i, LEVEL_IDX] = level
        parcels_data[i, MOISTURE_IDX] = moisture_present
        parcels_data[i, TIME_IDX] = time + delta_t_sec
        parcels_data[i, CURRENT_LATIDX] = latidx
        parcels_data[i, CURRENT_LONIDX] = lonidx

    return output_array

class Meteo:
    def __init__(self):
        self.time = None
        self.levels = None
        self.lats = None
        self.lons = None
        self.ua = None
        self.va = None
        self.evspsbl = None
        self.pr = None
        self.hus = None
        self.prw = None
        self.wap = None
        self.calendar = None
        self.degreelength_lat = None
        self.degreelength_lon = None
        self.surface_areas = None
        self.lat_bounds = None
        self.lon_bounds = None

    def compute_degree_lengths(self):
        degreelength_lat = np.empty_like(self.lats)
        degreelength_lon = np.empty_like(self.lats)

        for i, lat in enumerate(self.lats):
            curlatrad = lat * 2 * np.pi / 360
            degreelength_lat[i] = (
                    111132.92
                    + (-559.82 * np.cos(2 * curlatrad))
                    + 1.175 * np.cos(4 * curlatrad)
                    - 0.0023 * np.cos(6 * curlatrad)
            )
            degreelength_lon[i] = (
                    (111412.84 * np.cos(curlatrad))
                    - 93.5 * np.cos(3 * curlatrad)
                    + 0.118 * np.cos(5 * curlatrad)
            )

        self.degreelength_lat = degreelength_lat
        self.degreelength_lon = np.maximum(degreelength_lon, 100.0) # adding 100 meter prevents division by zero/near zero at poles

    def compute_grid_cell_areas(self):
        R = 6371000

        if self.lats is None or self.lons is None:
            raise ValueError('Lats and/or lons are not defined so surface area calculation is impossible')

        lats = np.array(self.lats)
        lons = np.array(self.lons)

        lat_bounds = np.zeros(len(lats) + 1)
        lat_bounds[1:-1] = 0.5 * (lats[:-1] + lats[1:])
        lat_bounds[0] = lats[0] - 0.5 * (lats[1] - lats[0])
        lat_bounds[-1] = lats[-1] + 0.5 * (lats[-1] - lats[-2])
        lat_bounds = np.radians(lat_bounds)

        lon_bounds = np.zeros(len(lons) + 1)
        lon_bounds[1:-1] = 0.5 * (lons[:-1] + lons[1:])
        lon_bounds[0] = lons[0] - 0.5 * (lons[1] - lons[0])
        lon_bounds[-1] = lons[-1] + 0.5 * (lons[-1] - lons[-2])
        lon_bounds = np.radians(lon_bounds)

        area_grid = np.zeros((len(lats), len(lons)))

        for i in range(len(lats)):
            phi1 = lat_bounds[i]
            phi2 = lat_bounds[i + 1]
            sin_diff = np.abs(np.sin(phi2) - np.sin(phi1))
            for j in range(len(lons)):
                lambda_diff = np.abs(lon_bounds[j + 1] - lon_bounds[j])
                area = R ** 2 * lambda_diff * sin_diff
                area_grid[i, j] = area

        self.surface_areas = area_grid
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds

def create_basin_mask(shapefile_path, hybas_id_to_use, lats, lons, resolution_factor=100):
    gdf = gpd.read_file(shapefile_path)
    basin = gdf[gdf['HYBAS_ID'] == hybas_id_to_use].geometry.iloc[0]

    west_poly = box(-180, -90, 0, 90)
    east_poly = box(0, -90, 180, 90)

    basin_west = basin.intersection(west_poly)
    basin_east = basin.intersection(east_poly)

    basin_west_shifted = translate(basin_west, xoff=360)

    basin_360 = unary_union([basin_east, basin_west_shifted])

    lat_res = lats[1] - lats[0]
    lon_res = lons[1] - lons[0]
    sub_lat_res = lat_res / resolution_factor
    sub_lon_res = lon_res / resolution_factor

    mask_hr = rasterize(
        [(mapping(basin_360), 1)],
        out_shape=(len(lats) * resolution_factor, len(lons) * resolution_factor),
        transform=Affine.translation(lons.min() - lon_res / 2, lats.min() - lat_res / 2) *
                  Affine.scale(sub_lon_res, sub_lat_res),
        all_touched=True,
        dtype=float
    )

    mask = mask_hr.reshape(len(lats), resolution_factor, len(lons), resolution_factor).mean(axis=(1, 3))

    return mask

def map_forcing_files(root_dir, model, scenario, current_year):
    target_vars = ['hfls', 'hus', 'pr', 'prw', 'tas', 'ua', 'va', 'wap']
    file_map = {}

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".nc") and model in file and scenario in file:

                    year_part = file.split('_')[-1].replace('.nc', '')
                    start_yr = int(year_part[:4])
                    end_yr = int(year_part[4:])

                    if not (start_yr <= current_year <= end_yr):
                        continue

                    var_name = file.split('_')[0]
                    if var_name in target_vars:
                        file_map[var_name] = os.path.join(root, file)

    return file_map

def load_year_to_ram(file_map, model, scenario, current_year):
    data_ram = {}

    for var, path in file_map.items():

        with xr.open_dataset(path, use_cftime=True) as ds:
            ds_year = ds.sel(time=(ds.time.dt.year == current_year))
            ds_year = ds_year.assign_coords(lon=(ds_year.lon % 360))
            ds_year = ds_year.sortby('lon')
            ds_year = ds_year.sortby('lat')

            data_ram[var] = ds_year[var].values

            if 'times' not in data_ram:
                data_ram['times'] = ds_year.time.values
                data_ram['lats'] = ds_year.lat.values
                data_ram['lons'] = ds_year.lon.values

            if 'levels' not in data_ram and 'plev' in ds_year:
                data_ram['levels'] = ds_year.plev.values

    return data_ram

def find_land_mask(forcing_directory, model):
    land_mask_data = None
    original_lats = None
    original_lons = None

    for root, dirs, files in os.walk(forcing_directory):
        for file in files:
            if model in file and 'sftlf' in file:
                    ds = xr.open_dataset(os.path.join(root, file))
                    land_mask_data = ds['sftlf'].values / 100
                    original_lats = ds['lat'].values
                    original_lons = ds['lon'].values
                    ds.close()


        if land_mask_data is not None:
            break

    original_lons = np.mod(original_lons, 360)

    if original_lats[0] > original_lats[1]:
        lat_slice = slice(None, None, -1)
    else:
        lat_slice = slice(None)
    if original_lons[0] > original_lons[1]:
        lon_slice = slice(None, None, -1)
    else:
        lon_slice = slice(None)

    land_mask_data = land_mask_data[lat_slice, :][:, lon_slice]

    return land_mask_data

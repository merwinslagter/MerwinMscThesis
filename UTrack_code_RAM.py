from UTrack_functions import *
import os
from datetime import datetime, timedelta, time
import calendar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd

#             [0]          [1]        [2]           [3]             [4]
models = ['CanESM5-1', 'EC-Earth3', 'MIROC6', 'MPI-ESM1-2-HR', 'NorESM2-MM']
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

###############################################################################
#############                SIMULATION SETTINGS                  #############
###############################################################################
start_date = '31-12-2054'
release_end_date = '16-01-2045'
parcels_per_mm = 1000.0  # Number of parcels to be released per mm precipitation
scenario = 'ssp126'  # SSP-RCP scenario chosen for the simulation
delta_t = 0.25  # Set timestep length in hours
forward_tracking = False  # implement forward or backward tracking
dynamic_plotting = False  # dynamically plot the parcels and allocation
surface_area_weighting = True  # Do you want the simulation to consider the actual grid cell sizes for distributing the parcels over the mask
tracking_days = 15  # days that parcels are tracked
kill_threshold = 0.01  # Set minimum amount of moisture present in a parcel
hybas_id = 6030007000

###############################################################################
#############      TIME CONVERSION FORWARD/BACKWARD CONFIG        #############
###############################################################################
start_date = datetime.strptime(start_date, "%d-%m-%Y")
release_end_date = datetime.strptime(release_end_date, "%d-%m-%Y")
tracking_time = timedelta(days=tracking_days)
dt = timedelta(hours=delta_t)

if forward_tracking:
    end_date = release_end_date + tracking_time
    track_key_word = 'Forward'
else:
    end_date = release_end_date - tracking_time
    track_key_word = 'Backward'

# check if the dates are set and if all files are available for the specific simulation
if forward_tracking and start_date > release_end_date:
    raise ValueError('For forward tracking the release_end_date must be later than the start_date.')
if not forward_tracking and start_date < release_end_date:
    raise ValueError('For backward tracking the release_end_date must NOT be later than the start_date.')
if abs(end_date - start_date) < tracking_time:
    raise ValueError('The difference in days between the start and end date cannot be smaller than the tracking time')

# Set the right output directory and check whether the simulation has already been executed before (file exists)


###############################################################################
#############                  MODEL LOOP                         #############
###############################################################################
for model in models:
    start_full_simulation = datetime.now()

    forcing_directory = rf"C:\Thesis_Merwin\CMIP6\Forcing_data_{model}"

    check_file_availability(forcing_directory, start_date, end_date, model, scenario)

    output_directory = rf"C:\Thesis_Merwin\UTrack_outputs\{hybas_id}_output"
    output_fn = os.path.join(output_directory,
                             f"UTrack-{track_key_word}-{model}_{scenario}_{start_date.day}-{start_date.month}-{start_date.year}_{release_end_date.day}-{release_end_date.month}-{release_end_date.year}_{hybas_id}.nc")
    if os.path.exists(output_fn):
        print(f"Output file already exists for {model}")
        continue

    ###############################################################################
    #############                 INITIALISATION                      #############
    ###############################################################################
    current_year = start_date.year
    file_map = map_forcing_files(forcing_directory, model, scenario, current_year)
    data_ram = load_year_to_ram(file_map, model, scenario, current_year)

    # Create empty meteo objects for holding forcing data
    m = Meteo()
    m.lats = data_ram['lats']
    m.lons = data_ram['lons']
    m.levels = data_ram['levels']

    # calculate the degree lengths initially before starting the simulation loop so it can be accessed through the meteo object throughout the simulation without the need to recalculate every loop
    m.compute_degree_lengths()
    m.compute_grid_cell_areas()

    mask = create_basin_mask(
        shapefile_path=rf"C:\Thesis_Merwin\HydroBASINS\HydroBASIN_Global_Level3.shp",
        hybas_id_to_use=hybas_id,
        lats=m.lats,
        lons=m.lons)

    basin_area_m2 = np.sum(mask* m.surface_areas)
    parcels_kg = parcels_per_mm / basin_area_m2

    # Create output array with the dimensions of the forcing data
    output_array = np.zeros([len(m.lats), len(m.lons)])

    parcel_data_columns = ['latitude', 'longitude', 'level', 'moisture_present', 'original_moisture',
                           'time', 'start_time', 'startlatidx', 'startlonidx', 'current_latidx', 'current_lonidx']

    # Define indexes for parcel columns in parcel_data array
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

    parcels_data = np.empty((0, len(parcel_data_columns)), dtype=np.float64)

    # SIMULATION
    print(
        f'Model: {model} Scenario: {scenario} From {start_date} till {end_date} for {hybas_id} with {parcels_per_mm} parcels_per_mm')

    ###############################################################################
    #############         DYNAMIC VISUALISATION CONFIGURATION         #############
    ###############################################################################
    width = len(m.lons)
    shift_amount = width // 2
    img, parcel_scatter = None, None

    if dynamic_plotting:

        fig, ax = plt.subplots(figsize=(10, 6))
        margin = 7

        hybas_gdf = gpd.read_file(rf"C:\Thesis_Merwin\HydroBASINS\HydroBASIN_Global_Level3.shp")

        lon_res = 360 / width
        lat_res = abs(m.lats[1] - m.lats[0])

        matrix = [1 / lon_res, 0, 0, 1 / lat_res, 180 / lon_res, abs(m.lats.min())/lat_res]

        hybas_gdf['geometry'] = hybas_gdf.geometry.affine_transform(matrix)
        hybas_gdf.boundary.plot(ax=ax, color='black', linewidth=0.6, zorder=2, aspect=None)

        target_basin = hybas_gdf[hybas_gdf['HYBAS_ID'] == hybas_id]
        target_basin.boundary.plot(ax=ax, color='#00AA00', linewidth=1.5, zorder=5, aspect=None)

        mask_rolled = np.roll(mask, shift_amount, axis=1)
        mask_rolled = np.where(mask_rolled > 0, mask_rolled, np.nan)

        moisture_plot = np.where(output_array > 0, output_array, np.nan)
        moisture_cmap = mcolors.LinearSegmentedColormap.from_list("white_to_blue", [(1, 1, 1), (0, 0, 1)], N=100)
        moisture_cmap.set_bad(color=(1,1,1, 0))

        if np.isnan(moisture_plot).all():
            vmax = 1e-9
        else:
            vmax = np.nanmax(moisture_plot)

        img = ax.imshow(moisture_plot, cmap=moisture_cmap, interpolation='nearest', origin='lower',
                         extent=(0, width, 0, len(m.lats)), vmin=0, vmax=vmax, zorder=1)
        fig.colorbar(img, ax=ax, label="mm moisture allocated")

        # Create a scatter plot for parcels outside the loop (initial empty scatter)
        parcel_scatter = ax.scatter([], [], color='red', marker='o', s=0.5, label="Parcels")

        rows, cols = np.where(mask_rolled > 0)
        if rows.size > 0:
            ax.set_xlim(cols.min() - margin, cols.max() + margin)
            ax.set_ylim(rows.min() - margin, rows.max() + margin)

        plt.ion()

    ###############################################################################
    #############                 SIMULATION LOOP                     #############
    ###############################################################################
    current_date = start_date.replace(hour=23, minute=45)
    last_printed_date = current_date.date()
    last_loaded_day = -1
    one_second = timedelta(seconds=1)

    parcels_in = parcels_data.shape[0]
    parcels_at_start_of_day = parcels_data.shape[0]
    total_released = 0
    daily_parcels_added = 0
    daily_parcels_killed = 0
    daily_parcels_expired = 0

    while True:
        #print(f"LOOP START: Clock is {current_date}")

        if current_date.year != current_year:
            current_year = current_date.year
            last_loaded_day = -1

            file_map = map_forcing_files(forcing_directory, model, scenario, current_year)

            del data_ram
            data_ram = load_year_to_ram(file_map, model, scenario, current_year)

        if not forward_tracking and current_date.time() == time(0, 0, 0):
            calc_date = current_date - one_second
        else:
            calc_date = current_date

        day_of_year = calc_date.timetuple().tm_yday - 1

        if m.calendar in ['noleap', '365_day'] and calendar.isleap(calc_date.year) and calc_date.month > 2:
            day_of_year -= 1

        if day_of_year < 0:
            day_of_year = 0
        elif day_of_year >= data_ram['ua'].shape[0]:
            day_of_year = data_ram['ua'].shape[0] - 1

        if day_of_year != last_loaded_day:
            m.ua = data_ram['ua'][day_of_year]
            m.va = data_ram['va'][day_of_year]
            m.wap = data_ram['wap'][day_of_year]
            m.pr = data_ram['pr'][day_of_year]
            m.hus = data_ram['hus'][day_of_year]
            m.tas = data_ram['tas'][day_of_year]
            m.hfls = data_ram['hfls'][day_of_year]
            m.prw = data_ram['prw'][day_of_year]
            m.evspsbl = convert_hfls_to_evap(m.hfls, m.tas)

            last_loaded_day = day_of_year

        if (forward_tracking and current_date <= release_end_date) or (
                not forward_tracking and current_date >= release_end_date):
            new_parcels = create_parcels_mask(mask, current_date, dt, m, parcels_kg, forward_tracking,
                                              parcel_data_columns, surface_area_weighting)
            #print(f"RELEASE: Created {len(new_parcels)} parcels at {current_date}")
            if new_parcels.size > 0:
                parcels_data = np.concatenate((parcels_data, new_parcels))
                total_released += np.sum(new_parcels[:, ORIGINAL_MOISTURE_IDX])
                daily_parcels_added += new_parcels.shape[0]

        if parcels_data.size > 0:
            output_array = update_parcels(parcels_data, dt.total_seconds(), m.ua, m.va, m.wap, m.pr, m.evspsbl,
                                          m.prw, m.hus, m.levels, m.lats, m.lons, m.lons.min(),
                                          m.degreelength_lat, m.degreelength_lon, forward_tracking, output_array)

        present_fraction = parcels_data[:, MOISTURE_IDX] / parcels_data[:, ORIGINAL_MOISTURE_IDX]
        kill_mask = present_fraction < kill_threshold
        expire_mask = (parcels_data[:, TIME_IDX] - parcels_data[:, START_TIME_IDX]) > tracking_time.total_seconds()
        combined_mask = np.logical_or(kill_mask, expire_mask)
        parcels_data = parcels_data[np.logical_not(combined_mask)]

        daily_parcels_killed += np.sum(kill_mask)
        daily_parcels_expired += np.sum(expire_mask & ~kill_mask)
        parcels_in = parcels_data.shape[0]

        if dynamic_plotting and (current_date.hour % 2 == 0 and current_date.minute == 0):
            output_rolled = np.roll(output_array, shift_amount, axis=1)
            parcel_latidxs = parcels_data[:, CURRENT_LATIDX].astype(int)
            parcel_lonidxs_rolled = (parcels_data[:, CURRENT_LONIDX].astype(int) + shift_amount) % width
            img.set_data(output_rolled)
            img.set_clim(0, np.max(output_rolled) if np.max(output_rolled) > 0 else 1e-5)
            parcel_scatter.set_offsets(np.c_[parcel_lonidxs_rolled, parcel_latidxs])

            plt.draw()
            plt.pause(0.001)

        current_date += dt if forward_tracking else -dt

        if current_date.date() != last_printed_date:
            total_run_time = (datetime.now() - start_full_simulation).total_seconds()

            print(f"{last_printed_date} | "
                  f"Total: {int(parcels_at_start_of_day):7d} | "
                  f"Added: {int(daily_parcels_added):6d} | "
                  f"Killed: {int(daily_parcels_killed):6d} | "
                  f"Expired: {int(daily_parcels_expired):6d} | "
                  f"Time: {total_run_time:5.1f}s | "
                  f"Frac alloc: {round(np.sum(output_array) / total_released, 2) if total_released else 0}")

            parcels_at_start_of_day = parcels_in
            daily_parcels_added = 0
            daily_parcels_killed = 0
            daily_parcels_expired = 0
            last_printed_date = current_date.date()

        if (forward_tracking and current_date >= end_date) or (not forward_tracking and current_date < end_date):
            break

    end_full_simulation = datetime.now()

    if dynamic_plotting:
        plt.ioff()
        plt.show()  # show plot at the end

    # Save output file
    create_nc_file(output_fn, output_directory, forward_tracking, model, scenario, start_date, release_end_date, m, output_array, total_released)

    del data_ram

    hours = int((end_full_simulation - start_full_simulation).total_seconds() / 3600)
    mins = int(((end_full_simulation - start_full_simulation).total_seconds() % 3600) / 60)
    secs = round(((end_full_simulation - start_full_simulation).total_seconds() % 3600) % 60, 2)
    print(f"Finished {model} in {hours}h {mins}m {secs}s")

print(
    f'Finished with all models')

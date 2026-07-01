import os
import xarray as xr

project = "CMIP6"
source_id = "NorESM2-MM"
experiment_id = "ssp126"
variable_id = "wap"

base_pathway = rf"C:\Users\Merwin\Documents\Globus\{project}\{source_id}\{experiment_id}\{variable_id}"

files = [
    rf"{base_pathway}\{variable_id}_day_{source_id}_{experiment_id}_r1i1p1f1_gn_20150101-20201231.nc",
    rf"{base_pathway}\{variable_id}_day_{source_id}_{experiment_id}_r1i1p1f1_gn_20210101-20301231.nc",
    rf"{base_pathway}\{variable_id}_day_{source_id}_{experiment_id}_r1i1p1f1_gn_20410101-20501231.nc",
    rf"{base_pathway}\{variable_id}_day_{source_id}_{experiment_id}_r1i1p1f1_gn_20510101-20601231.nc",
]

time_units = "days since 1850-01-01 00:00:00"
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

ds = xr.open_mfdataset(files, combine="by_coords", decode_times=time_coder, data_vars="all")
ds_sel = ds.sel(time=slice("2015-01-01", "2024-12-31"))
ds_sel.load()

print(ds_sel.time[0].values, ds_sel.time[-1].values)
print("Number of timesteps:", ds_sel.sizes["time"])
print("Calendar:", ds.time.encoding.get("calendar"))

calendar = ds_sel.time.encoding.get("calendar", ds.time.encoding.get("calendar"))

confirm = input("Does this look correct? Type 'ok' to continue: ")
if confirm.lower() == "ok":

    encoding = {}
    for var in ds_sel.data_vars:
        encoding[var] = {
            "zlib" : True,
            "complevel": 4,
            "dtype": "float64",
        }

    encoding["time"] = {
        "units": time_units,
        "calendar": calendar,
        "dtype": "float64"
    }
    if "time_bnds" in ds_sel:
        encoding["time_bnds"] = {
            "units": time_units,
            "calendar": calendar,
            "dtype": "float64"
        }

output_filename = f"baseline_{variable_id}_{experiment_id}_{source_id}.nc"
output_file = os.path.join(base_pathway, output_filename)

ds_sel.to_netcdf(output_file, encoding=encoding, format="NETCDF4")

print("Finished writing:", output_file)


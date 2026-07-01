import os
import requests
import globus_sdk
from globus_sdk import NativeAppAuthClient
from globus_sdk import TransferClient, TransferData, AccessTokenAuthorizer

# Globus initialization
CLIENT_ID = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
client= NativeAppAuthClient(CLIENT_ID)

flow = client.oauth2_start_flow(
    requested_scopes="urn:globus:auth:scope:transfer.api.globus.org:all",
    refresh_tokens=True
)

authorize_url = flow.get_authorize_url()
print(authorize_url)

auth_code = input("Authorization code: ").strip()
token_response = flow.exchange_code_for_tokens(auth_code)
transfer_access_token = token_response.by_resource_server['transfer.api.globus.org']['access_token']
transfer_refresh_token = token_response.by_resource_server['transfer.api.globus.org']['refresh_token']

authorizer = globus_sdk.RefreshTokenAuthorizer(
    transfer_refresh_token,
    client
)
tc = globus_sdk.TransferClient(authorizer=authorizer)

# ESGF Metagrid. Adjust parameters based on goal
url = "https://esgf-node.llnl.gov/esg-search/search"
params = {
    "project": "CMIP6",
    "source_id": "CanESM5-1",
    "experiment_id": "ssp585",
    "variable_id": "va",
    "frequency": "day",
    "variant_label": "r1i1p1f1",
    "type": "File",
    "format": "application/solr+json",
    "limit": 1000
}

response = requests.get(url, params=params)
response.raise_for_status()
data = response.json()
docs = data["response"]["docs"]

print("Files found:", data["response"]["numFound"])

# Globus transfer
destination_endpoint_id = "827b81c0-f517-11f0-a81d-02edfb1d9cf1"
base_folder = "/~/Globus"
destination_folder = f"{base_folder}/{params['project']}/{params['source_id']}/{params['experiment_id']}/{params['variable_id']}"

os.makedirs(destination_folder, exist_ok=True)

print("Destination folder:", destination_folder)

#Adjust ranges per model
desired_ranges = ["20150101-20151231",
                  "20160101-20161231",
                  "20170101-20171231",
                  "20180101-20181231",
                  "20190101-20191231",
                  "20200101-20201231",
                  "20210101-20211231",
                  "20220101-20221231",
                  "20230101-20231231",
                  "20240101-20241231",
                  "20450101-20451231",
                  "20460101-20461231",
                  "20470101-20471231",
                  "20480101-20481231",
                  "20490101-20491231",
                  "20500101-20501231",
                  "20510101-20511231",
                  "20520101-20521231",
                  "20530101-20531231",
                  "20540101-20541231"]

files_to_transfer = []
added_files = set()

for doc in docs:
    print(doc["title"])
    urls = doc.get("url", [])
    for u in urls:
        if "Globus" in u:
            globus_url = u.split("|")[0]
            if globus_url.startswith("globus://"):
                path_part = globus_url[len("globus://"):]
            elif globus_url.startswith("globus:"):
                path_part = globus_url[len("globus:"):]
            else:
                print("Skipping unknown format:", globus_url)
                continue

            parts = path_part.split("/")
            source_endpoint_id = parts[0]
            file_path = "/" + "/".join(parts[1:])
            filename = os.path.basename(file_path)

            if source_endpoint_id == "X":
                continue

            if any(r in filename for r in desired_ranges) and "_day_" in filename:
                if filename not in added_files:
                    local_path = f"{destination_folder}/{filename}"
                    files_to_transfer.append((source_endpoint_id, file_path, local_path))
                    added_files.add(filename)
# Check files
print("Files selected for transfer:")
for f in files_to_transfer:
    print(f"Source endpoint: {f[0]}, File: {f[1]}, Local path: {f[2]}")

proceed = input ("\nDo you want to proceed with the transfer? Type 'yes' to continue: ").strip().lower()
if proceed != "yes":
    print("Transfer cancelled.")
    exit()

# Submit transfer
for source_ep in set(f[0] for f in files_to_transfer):
    try:
        tc.get_endpoint(source_ep)
    except globus_sdk.GlobusAPIError as e:
        print(f"Skipping endpoint {source_ep}: {e.message}")
        continue

    tdata = TransferData(source_ep, destination_endpoint_id)
    for f in files_to_transfer:
        if f[0] == source_ep:
            tdata.add_item(f[1], f[2])
    transfer_result = tc.submit_transfer(tdata)
    print(f"Submitted transfer from {source_ep}. Task ID:", transfer_result["task_id"])

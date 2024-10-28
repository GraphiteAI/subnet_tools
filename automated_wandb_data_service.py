import wandb
from dotenv import load_dotenv
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
from huggingface_hub import HfApi
import time
import pandas as pd
import csv

load_dotenv()


def get_latest_file(folder_path):
    date_format = "%Y_%m_%d"
    
    files = os.listdir(folder_path)
    
    latest_file = None
    latest_date = None
    
    for file in files:
        if file.startswith("Metric_TSP_V2_") and file.endswith(".tsv"):
            try:
                date_str = file.split('_')[3] + '_' + file.split('_')[4] + '_' + file.split('_')[5].split('.')[0]
                file_date = datetime.strptime(date_str, date_format)
                
                if latest_date is None or file_date > latest_date:
                    latest_date = file_date
                    latest_file = file
            
            except ValueError:
                print(f"Skipping file due to date parsing error: {file}")

    if latest_file:
        return os.path.join(folder_path, latest_file)
    else:
        return None

def get_latest_date_from_file(folder_path):
    latest_file_path = get_latest_file(folder_path)
    
    if latest_file_path:
        df = pd.read_csv(latest_file_path, sep='\t')
        
        if 'created_at' not in df.columns:
            raise ValueError("The 'created_at' column is not present in the file.")
        
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        latest_date = df['created_at'].max()
        
        return latest_date
    else:
        raise FileNotFoundError("No valid files found in the folder.")

folder_path = 'past_data'
try:
    last_update = get_latest_date_from_file(folder_path)
    print(f"The latest date in 'created_at' column is: {last_update}")
except Exception as e:
    print(e)

# Define local directory
parent = Path(__file__).resolve().parent
LOCAL_DIR = parent / 'past_data'
# Create 'past_data' folder if it does not exist
if not LOCAL_DIR.exists():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created folder: {LOCAL_DIR}")
else:
    print(f"Folder already exists: {LOCAL_DIR}")

# Define global variables
SCRAPE_INTERVAL = 60 * 60 * 3 # 3 hours
# last_update = datetime.fromisoformat("2024-09-18T00:00:00Z".replace('Z', '+00:00'))
HUGGINGFACE_REPO = os.getenv("HF_REPO")
REPO_TYPE = "dataset"
PREFIX = "Metric_TSP_V2"

# function for getting 1 week ago string
def get_one_week_ago():
    # Get the current UTC time
    now = datetime.now(timezone.utc)

    # Calculate the time one week ago
    one_week_ago = now - timedelta(weeks=1)

    return one_week_ago

def get_reference_time():
    # compares the one_week_ago with the last_update
    one_week_ago = get_one_week_ago()
    if last_update:
        return one_week_ago.isoformat() if one_week_ago > last_update else last_update.isoformat()
    else:
        return one_week_ago.isoformat()

def process_date_string(date_string):
    iso_string = date_string.replace('Z', '+00:00')

    # Convert to datetime object
    dt = datetime.fromisoformat(iso_string)
    return dt

def new_run(run):
    '''
    evaluates of the current run has been checked before
    '''
    created_date = process_date_string(run._attrs["createdAt"])
    if last_update != None:
        if created_date <= last_update:
            return False
        else:
            return True
    else:
        return True

def extract_run_info(run):
    '''
    Constructs json object
    '''
    try:
        run_id = run._attrs['name']
        config = run._attrs['config']
        validator = json.loads(run._attrs['description'])['validator']
        n_nodes = config['n_nodes']
        created_at = run._attrs['createdAt']
        run_id = run._attrs['name']
        selected_ids = config['selected_ids']
        dataset_ref = config['dataset_ref']
        time_elapsed = config['time_elapsed']
        history = run.history()
        for column in history.columns:
            if column.startswith('distance'):
                distance_data = [round(float(distance), 5) for distance in history[column].tolist()]
            if column.startswith('rewards'):
                reward_data = [round(float(reward), 5) for reward in history[column].tolist()]
        selected_uids = config.get('selected_uids',None)
        problem_type = config['problem_type']
        n_salesmen = config.get('n_salesmen', None)
        depots = config.get("depots", None)
        return {
            'run_id': run_id,
            'validator': validator,
            'n_nodes': n_nodes,
            'created_at': created_at,
            'run_id': run_id,
            'selected_ids': selected_ids,
            'dataset_ref': dataset_ref,
            'time_elapsed': time_elapsed,
            'distances': distance_data,
            'rewards': reward_data,
            'selected_uids': selected_uids,
            'problem_type': problem_type,
            'n_salesmen': n_salesmen,
            'depots': depots
        }
    except KeyError as e:
        print(f"Error: {e} in {run._attrs['name']}")
        return None
    except UnboundLocalError as e:
        print(f"Error: {e} in {run._attrs['name']}")
        return None
    
def get_file_path(date_string):
    '''
    returns the expected file name given a date_string and checks if it exists
    '''
    date = process_date_string(date_string)
    date_str = date.strftime("%Y_%m_%d")
    return LOCAL_DIR / f"{PREFIX}_{date_str}.tsv"


# def append_row_to_tsv(file_path: Path, row_dicts: list) -> None:
#     # Ensure the file path is a Path object
#     file_path = Path(file_path)

#     # Check if the file exists
#     file_exists = file_path.exists()

#     # Open the file in append mode
#     with file_path.open('a', newline='') as file:
#         # Create a CSV writer object with tab delimiter
#         for row_dict in row_dicts:
#             writer = csv.DictWriter(file, fieldnames=row_dict.keys(), delimiter='\t')

#             # Write header if the file doesn't exist or is empty
#             if not file_exists or file_path.stat().st_size == 0:
#                 writer.writeheader()
#                 file_exists = True
            
#             # Write the new row
#             writer.writerow(row_dict)

# def append_row_to_tsv(file_path: Path, row_dicts: list) -> None:
#     # Ensure the file path is a Path object
#     file_path = Path(file_path)

#     # Check if the file exists
#     file_exists = file_path.exists()

#     # # Read existing data if the file exists
#     # existing_data = pd.DataFrame()
#     # Open the file in append mode
#     with file_path.open('a', newline='') as file:
#         # Create a CSV writer object with tab delimiter
#         for row_dict in row_dicts:
#             writer = csv.DictWriter(file, fieldnames=row_dict.keys(), delimiter='\t')

#             # Write header if the file doesn't exist or is empty
#             if not file_exists or file_path.stat().st_size == 0:
#                 writer.writeheader()
#                 file_exists = True
    
#             # Write the new row
#             writer.writerow(row_dict)

def append_row_to_tsv(file_path: Path, row_dicts: list) -> None:
    # Ensure the file path is a Path object
    file_path = Path(file_path)

    try:
        # Check if the file exists
        file_exists = file_path.exists()
        existing_data = []
        existing_headers = []

        if file_exists:
            # Read existing data and headers
            with file_path.open('r', newline='') as file:
                reader = csv.DictReader(file, delimiter='\t')
                existing_headers = reader.fieldnames
                existing_data = [row for row in reader]

        # Get the new fieldnames from the first row_dict
        new_fieldnames = row_dicts[0].keys() if row_dicts else []

        # Check if headers match
        if existing_headers != list(new_fieldnames):
            # Rewrite the whole CSV file with new headers
            with file_path.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=new_fieldnames, delimiter='\t')
                writer.writeheader()
                
                # Write existing rows, filling missing fields with None
                for row in existing_data:
                    new_row = {key: row.get(key, None) for key in new_fieldnames}
                    writer.writerow(new_row)

                # Write new rows
                for row_dict in row_dicts:
                    new_row = {key: row_dict.get(key, None) for key in new_fieldnames}
                    writer.writerow(new_row)

        else:
            # Append new rows if headers match
            with file_path.open('a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=existing_headers, delimiter='\t')

                # Write new rows
                for row_dict in row_dicts:
                    new_row = {key: row_dict.get(key, None) for key in existing_headers}
                    writer.writerow(new_row)

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except PermissionError:
        print(f"Error: Permission denied when accessing {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

def write_files(run_rows: list) -> None:
    file_row_dict = {}

    for row in run_rows:
        assigned_file = get_file_path(row['created_at'])
        if file_row_dict.get(assigned_file):
            file_row_dict[assigned_file].append(row)
        else:
            file_row_dict[assigned_file] = [row]

    for file_path, row_dicts in file_row_dict.items():
        append_row_to_tsv(file_path, row_dicts)
    
    return list(file_row_dict.keys())

def upload_file(file_path, hf_api):
    hf_api.upload_file(
        path_or_fileobj=file_path,
        repo_id="Graphite-AI/Graphite_Past_Problems",
        repo_type="dataset",
        path_in_repo=file_path.name,
    )

def scrape_service():
    global last_update
    hf_api_client = HfApi(token=os.getenv("HF_TOKEN"))
    # load in the wandb api
    wandb_key = os.getenv("WANDB_API_KEY")

    # log into wandb and scrape data
    wandb.login(key = wandb_key, relogin = True)

    # Instantiate the wandb API()
    wandb_api_client = wandb.Api()
    runs = wandb_api_client.runs('graphite-ai/Graphite-Subnet-V2', per_page=1000, order="+created_at", filters={"createdAt":{"$gte":get_reference_time()}})
    print(f"Scraping wandb at: {datetime.now(timezone.utc)} UTC with {len(runs)} called")

    # scrape to completion
    run_rows = []
    run = runs.next()
    i = 1
    while True:
        if new_run(run):
            run_row = extract_run_info(run)
            if run_row:
                run_rows.append(run_row)
                last_update = process_date_string(run._attrs['createdAt']) # update global reference
                print(f"appended run_id: {run._attrs['name']} created at {run._attrs['createdAt']}")
                if i % 100 == 0:
                    print(f"Scraped {i} runs out of {len(runs)}")
                time.sleep(1)
            else:
                print(f"skipping run due to missing data")
        else:
            time.sleep(1)
        if i > 4000:
            print(f"Uploading by parts")
            break
        try:
            run = runs.next()
            i += 1
        except StopIteration:
            break

    # write data to file
    files_to_upload = write_files(run_rows)

    # push files
    for file_path in files_to_upload:
        upload_file(file_path, hf_api_client)

def main():
    while True:
        scrape_service()
        time.sleep(SCRAPE_INTERVAL)

if __name__=="__main__":
    main()
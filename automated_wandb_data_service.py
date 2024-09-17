import wandb
from dotenv import load_dotenv
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
from huggingface_hub import HfApi
import time
import csv

load_dotenv()

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
SCRAPE_INTERVAL = 60 * 30 # 30 min
last_update = None # represents the most recent update
# last_update = datetime.fromisoformat("2024-09-15T23:45:42Z".replace('Z', '+00:00'))
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
        if created_date < last_update:
            return False
        else:
            return True
    else:
        return True

def extract_run_info(run):
    '''
    Constructs json object
    '''
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
        'rewards': reward_data
    }

def get_file_path(date_string):
    '''
    returns the expected file name given a date_string and checks if it exists
    '''
    date = process_date_string(date_string)
    date_str = date.strftime("%Y_%m_%d")
    return LOCAL_DIR / f"{PREFIX}_{date_str}.tsv"


def append_row_to_tsv(file_path: Path, row_dicts: list) -> None:
    # Ensure the file path is a Path object
    file_path = Path(file_path)

    # Check if the file exists
    file_exists = file_path.exists()

    # Open the file in append mode
    with file_path.open('a', newline='') as file:
        # Create a CSV writer object with tab delimiter
        for row_dict in row_dicts:
            writer = csv.DictWriter(file, fieldnames=row_dict.keys(), delimiter='\t')

            # Write header if the file doesn't exist or is empty
            if not file_exists or file_path.stat().st_size == 0:
                writer.writeheader()
                file_exists = True
            
            # Write the new row
            writer.writerow(row_dict)

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
            run_rows.append(extract_run_info(run))
            last_update = process_date_string(run._attrs['createdAt']) # update global reference
            print(f"appended run_id: {run._attrs['name']} created at {run._attrs['createdAt']}")
            if i % 100 == 0:
                print(f"Scraped {i} runs out of {len(runs)}")
            time.sleep(5)
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
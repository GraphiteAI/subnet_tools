from huggingface_hub import snapshot_download
from pathlib import Path
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

# download all files to "past_data" local folder

def sync_data_recent(lookback_days=7):
    # downloads all files from the last lookback_days days
    def generate_allow_pattern(lookback_days):
        current_date = datetime.now()
        return [f"Metric_TSP_V2_{(current_date - timedelta(days=i)).strftime('%Y_%m_%d')}.tsv" for i in range(lookback_days)]
    HF_REPO = os.getenv("HF_REPO")
    print(HF_REPO)
    # check that local_dir exists
    parent = Path(__file__).resolve().parent
    LOCAL_DIR = parent / 'past_data'
    if not LOCAL_DIR.exists():
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {LOCAL_DIR}")
    else:
        print(f"Folder already exists: {LOCAL_DIR}")
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=LOCAL_DIR, allow_patterns=generate_allow_pattern(lookback_days))

def sync_all_data():
    # downloads all files
    HF_REPO = os.getenv("HF_REPO")
    print(HF_REPO)
    # check that local_dir exists
    parent = Path(__file__).resolve().parent
    LOCAL_DIR = parent / 'past_data'
    if not LOCAL_DIR.exists():
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {LOCAL_DIR}")
    else:
        print(f"Folder already exists: {LOCAL_DIR}")
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=LOCAL_DIR)

if __name__=="__main__":
    # sync_all_data()
    sync_data_recent()
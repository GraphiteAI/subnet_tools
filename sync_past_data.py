from huggingface_hub import snapshot_download
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

# download all files to "past_data" local folder

def main():
    # downloads all files
    HF_REPO = os.getenv("HF_REPO")
    LOCAL_DIR = Path(os.getenv("LOCAL_DIR"))
    # check if
    if LOCAL_DIR is None:
        parent = Path(__file__).resolve().parent
        LOCAL_DIR = parent / 'past_data'
    # check that local_dir exists
    if not LOCAL_DIR.exists():
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {LOCAL_DIR}")
    else:
        print(f"Folder already exists: {LOCAL_DIR}")
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=LOCAL_DIR)

if __name__=="__main__":
    main()
'''
This file contains helper functions for reading in and managing the dataset.

The I/O functions for file creation, retrieval, and loading are designed with the parent "Graphite-Subnet" as the reference folder.
'''
import tempfile
import requests
from requests import HTTPError
import os
import io
import gzip
from pathlib import Path
import numpy as np
import hashlib
from huggingface_hub import hf_hub_download
import time
from sim_utils.dataset_constants import ASIA_MSB_DETAILS, WORLD_TSP_DETAILS
DATASET_DIR = Path(__file__).resolve().parent.joinpath("dataset")

# checks if directory exists and creates it if it doesn't
def create_directory_if_not_exists(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

#_________________________________________#
###### Helper functions for training ######
#_________________________________________#

def get_file_path(ref_id:str)->Path:
    '''
    Inputs: ref_id unique identifier of supported dataset

    returns the file paths of the zipped coordinates
    '''
    file_with_extension = ref_id + ".npz"
    return DATASET_DIR / file_with_extension

def get_checksum(coordinates:np.array)->str:
    '''
    Function getting 128 byte checksum for aligning datasets between validator and miner.

    Inputs: coordinates numpy array representing node coordinates of the dataset
    Output: md5 hash
    '''
    hash_algo='md5'
    hash_func=getattr(hashlib, hash_algo)()
    hash_func.update(coordinates.tobytes())
    return hash_func.hexdigest()

def load_dataset(ref_id:str)->dict:
    '''
    loads in coordinate information from the referenced .npz file and returns the coordinates

    Inputs: ref_id name of dataset
    '''
    filepath = get_file_path(ref_id)
    try:
        array_map = np.load(filepath)
        return {"data":array_map['data'],"checksum": get_checksum(array_map['data'])}
    except OSError as e:
        print(f"Error loading dataset: {e}")

def check_data_match(coordinates:np.array, checksum:str):
    '''
    Function called on the incoming synapse if both the validator and the miner support data checking.

    Inputs: coordinates numpy array loaded from .npz file
            checksum hash computed by the validator based on its reference data
    '''
    hash_algo='md5'
    hash_func=getattr(hashlib, hash_algo)()
    hash_func.update(coordinates.tobytes())
    return hash_func.hexdigest() == checksum

#_________________________________________#
###### Function for neuron data setup #####
#_________________________________________#
def check_and_get_msb():
    fp = get_file_path(ASIA_MSB_DETAILS['ref_id'])
    if fp.exists():
        # we have already downloaded and processed the data
        print(f"{ASIA_MSB_DETAILS['ref_id']} already downloaded")
        return
    else:
        try:
            create_directory_if_not_exists(DATASET_DIR)
            print(f"Downloading {ASIA_MSB_DETAILS['ref_id']} data from huggingface")
            hf_hub_download(repo_id="Graphite-AI/coordinate_data", filename="Asia_MSB.npz", repo_type="dataset", local_dir=DATASET_DIR)
        except:
            # Obtain byte content through get request to endpoint
            print(f"Manually {ASIA_MSB_DETAILS['endpoint']} data from source")



def check_and_get_wtsp():
    fp = get_file_path(WORLD_TSP_DETAILS['ref_id'])
    if fp.exists():
        # we have already downloaded and processed the data
        print(f"{WORLD_TSP_DETAILS['ref_id']} already downloaded")
        return
    else:
        try:
            create_directory_if_not_exists(DATASET_DIR)
            print(f"Downloading {WORLD_TSP_DETAILS['ref_id']} data from huggingface")
            hf_hub_download(repo_id="Graphite-AI/coordinate_data", filename="World_TSP.npz", repo_type="dataset", local_dir=DATASET_DIR)
        except:
            # Obtain byte content through get request to endpoint
            print(f"Manually {WORLD_TSP_DETAILS['endpoint']} data from source")

def download_default_datasets():
    check_and_get_msb()
    check_and_get_wtsp()

def load_default_dataset(neuron):
    '''
    Loads the default dataset into neuron as a dict of {"dataset_name":{"coordinates":np.array, "checksum":str}}
    '''
    create_directory_if_not_exists(DATASET_DIR)
    # check and process default datasets
    download_default_datasets()

    # set the neuron dataset attribute
    neuron.loaded_datasets = {
        ASIA_MSB_DETAILS['ref_id']: load_dataset(ASIA_MSB_DETAILS['ref_id']),
        WORLD_TSP_DETAILS['ref_id']: load_dataset(WORLD_TSP_DETAILS['ref_id'])
    }
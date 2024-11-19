import pandas as pd
from datetime import datetime
from sync_past_data import main as sync_data
import os
from pathlib import Path
import json
from json import JSONDecodeError
import numpy as np
from sim_utils.mock_protocol import GraphV2Problem, GraphV2ProblemMulti, GraphV2Synapse

# sync data
sync_data()

# point to folder with past_data
LOCAL_DIR = Path(os.getenv("LOCAL_DIR"))
SAVE_DIR = Path("sim_data")

if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created folder: {SAVE_DIR}")

def load_array(arr_string):
    # Clean up the possible erroneous values: 'Infinity', 'None', 'Null'
    arr_string = arr_string.lower()
    arr_string = arr_string.replace("infinity", "null").replace("none", "null").replace("inf", "null").replace("'","")
    try:
        arr = json.loads(arr_string)
        return arr
    except JSONDecodeError:
        print(arr_string)
        return []
    
def handled_json_loads(arr):
    if type(arr) != str:
        return None
    else:
        return json.loads(arr)


# get the latest dataset
def get_latest_dataset(past_data_dir):
    '''
    Input: path to .tsv data
    Output: loaded dataframe for data
    '''
    def get_dataset_date(filename):
        date_str = filename.replace(".tsv","").split("_V2_")[-1]
        date = datetime.strptime(date_str, "%Y_%m_%d")
        return date

    # Load in latest data
    latest_file = max([file for file in os.listdir(past_data_dir) if os.path.splitext(file)[-1]==".tsv"], key=lambda x:get_dataset_date(x))
    latest_dataset = pd.read_csv(os.path.join(past_data_dir,latest_file), sep="\t")
    

    # Clean the fields in latest_dataset
    latest_dataset["selected_uids"] = latest_dataset["selected_uids"].apply(load_array)
    latest_dataset["time_elapsed"] = latest_dataset["time_elapsed"].apply(load_array)
    latest_dataset["distances"] = latest_dataset["distances"].apply(load_array)
    latest_dataset["rewards"] = latest_dataset["rewards"].apply(load_array)
    latest_dataset["depots"] = latest_dataset["depots"].apply(handled_json_loads)
    latest_dataset["selected_ids"] = latest_dataset["selected_ids"].apply(handled_json_loads)
    # The other arrays should be loaded already because they must be JSON serializable (if not it is likely an invalid entry in wandb)
    latest_dataset["created_at"] = pd.to_datetime(latest_dataset["created_at"]) # load in created datetime for reference when saving dataset
    return latest_dataset

def save_dataset_json(problems_dict, filename):
    filepath = SAVE_DIR / Path(filename)
    try:
        if filepath.exists():
            while True:
                overwrite = input(f"{filepath} already exists. Overwrite [Y/N]?: ")
                if overwrite.lower() == "y":
                    print("Overwriting previous sample. Returning new saved data instead.")
                    break
                elif overwrite.lower() == "n":
                    print("Not overwriting previous sample. Returning previous saved data instead.")
                    return -1
                else:
                    print("Please return Y/N response")
                    continue
        with open(filepath, "w") as f:
            json.dump(problems_dict, f)
        return 1
    except Exception as e:
        return e

def get_problem_dict(row, problem_type):
    problem_dict = {}
    problem_dict["problem_type"] = problem_type
    problem_dict["n_nodes"] = row["n_nodes"]
    problem_dict["selected_ids"] = row["selected_ids"]
    problem_dict["dataset_ref"] = row["dataset_ref"]
    if "mTSP" in problem_type:
        problem_dict["n_salesmen"] = row["n_salesmen"]
        problem_dict["depots"] = row["depots"]
    return problem_dict

def get_valid_distances(distances):
    return [distance for distance in distances if distance!=None and np.isfinite(distance)]

def get_perf_dict(row):
    perf_dict = {}
    perf_dict["validator"] = row["validator"]
    valid_distances = get_valid_distances(row["distances"])
    perf_dict["best_solution"] = min(valid_distances)
    perf_dict["worst_solution"] = max(valid_distances)
    assert len(row["time_elapsed"]) == len(row["distances"]), ValueError("Mismatched distance and time_elapsed array sizes")
    perf_dict["best_solution_time_taken"] = row["time_elapsed"][row["distances"].index(perf_dict["best_solution"])]
    perf_dict["worst_solution_time_taken"] = row["time_elapsed"][row["distances"].index(perf_dict["worst_solution"])]
    return perf_dict

def get_random_sample(sample_size:int=30, problem_type:str='Metric mTSP', save:bool=True):
    '''
    Input: Takes in desired sample size (subject to number of valid rows in total dataset)
    '''
    # valid problem types
    valid_problem_types = ["Metric TSP", "Metric mTSP"]
    assert problem_type in valid_problem_types, ValueError(f"Requesting for invalid problem type: {problem_type}. Please choose from: {valid_problem_types}")
    # create a sample 
    latest_df = get_latest_dataset(LOCAL_DIR)
    # filter for desired problem
    problem_df = latest_df[latest_df["problem_type"]==problem_type]
    problem_df = problem_df[problem_df["distances"].apply(lambda x: len(get_valid_distances(x))>10)] # We define a valid problem as one that received at least 10 valid solutions; Otherwise, the comparison result will not be very meaningful.
    sampled_df = problem_df.sample(min(sample_size, len(problem_df)))
    print(f"Sampled {len(sampled_df)} past problems of type {problem_type}")
    # Create a list of dictionaries with the necessary information to recreate the problem
    sampled_problems = {row["run_id"]:{"problem_dict":None, "perf_data_dict":None} for _, row in sampled_df.iterrows()}
    # we need to save the problem_type, n_nodes, dataset_ref, selected_ids
    for row_idx, row in sampled_df.iterrows():
        sampled_problems[row["run_id"]]["problem_dict"] = get_problem_dict(row, problem_type)
        sampled_problems[row["run_id"]]["perf_data_dict"] = get_perf_dict(row)
    
    if save:
        filename = f"{sampled_df.iloc[0]['created_at'].date().strftime('%Y_%m_%d')}_{sample_size}_{problem_type.replace(' ', '_')}.json"
        return_code = save_dataset_json(sampled_problems, filename)
        if return_code == -1:
            reconstructed_synapses = load_from_past_dataset(filename)
        else:
            reconstructed_synapses = {run_id:{"synapse":recreate_synapse_from_dict(sampled_problems[run_id]["problem_dict"]), "perf_data_dict": sampled_problems[run_id]["perf_data_dict"]} for run_id in sampled_problems.keys()}
        return reconstructed_synapses, filename

        
    # return dict of the reconstructed Graph Synapses and the performance data associated with the synapse with the "run_id" as keys
    reconstructed_synapses = {run_id:{"synapse":recreate_synapse_from_dict(sampled_problems[run_id]["problem_dict"]), "perf_data_dict": sampled_problems[run_id]["perf_data_dict"]} for run_id in sampled_problems.keys()}
    return reconstructed_synapses, None


def recreate_synapse_from_dict(problem_dict):
    '''
    This method loads in the problem_dict
    '''
    if problem_dict["problem_type"] == "Metric TSP":
        problem_type = problem_dict["problem_type"]
        selected_ids = problem_dict["selected_ids"]
        dataset_ref = problem_dict["dataset_ref"]
        n_nodes = problem_dict["n_nodes"]
        test_problem = GraphV2Problem(problem_type=problem_type, selected_ids=selected_ids, dataset_ref=dataset_ref, n_nodes=n_nodes)
    elif problem_dict["problem_type"] == "Metric mTSP":
        problem_type = problem_dict["problem_type"]
        selected_ids = problem_dict["selected_ids"]
        dataset_ref = problem_dict["dataset_ref"]
        n_nodes = problem_dict["n_nodes"]
        n_salesmen = problem_dict["n_salesmen"]
        depots = problem_dict["depots"]
        test_problem = GraphV2ProblemMulti(problem_type=problem_type, selected_ids=selected_ids, dataset_ref=dataset_ref, n_nodes=n_nodes, n_salesmen=n_salesmen, depots=depots)
    else:
        raise ValueError(f"Received problem dict with invalid problem type: {problem_dict['problem_type']}")
    synapse = GraphV2Synapse(problem=test_problem)
    return synapse

def load_from_past_dataset(filename):
    filepath = SAVE_DIR / Path(filename)
    with open(filepath, "r") as f:
        sampled_problems = json.load(f)
    return {run_id:{'synapse':recreate_synapse_from_dict(sampled_problems[run_id]['problem_dict']), 
                    'perf_data_dict': sampled_problems[run_id]['perf_data_dict']} for run_id in sampled_problems.keys()}
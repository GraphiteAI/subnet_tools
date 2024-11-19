from sim_utils.dataset_utils import load_default_dataset
from sim_utils.distance import geom_edges, man_2d_edges, euc_2d_edges
from sim_utils.data_fetcher import get_random_sample, load_from_past_dataset
from sim_utils.mock_protocol import GraphV2Synapse, GraphV2Problem, GraphV2ProblemMulti
from sim_utils.utils import get_multi_minmax_tour_distance, get_tour_distance
from sim_utils.template_mock_solver import MockMTSPSolver, MockTSPSolver
from sync_past_data import main as sync_data

from typing import Union
from abc import abstractmethod
import numpy as np
import pandas as pd

from tqdm import tqdm
import time

import json
import copy
import os
from pathlib import Path

SIM_RES_DIR = Path("sim_results")
if not SIM_RES_DIR.exists():
    SIM_RES_DIR.mkdir(parents=True)
    print(f"Made directory to save simulation results")

class MockMiner:
    def __init__(self):
        # initialize MockMiner with dataset
        load_default_dataset(self)

    def recreate_edges(self, problem:Union[GraphV2Problem, GraphV2ProblemMulti]):
        # get coordinates
        node_coords_np = self.loaded_datasets[problem.dataset_ref]["data"]
        node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
        if problem.cost_function == "Geom":
            return geom_edges(node_coords)
        elif problem.cost_function == "Euclidean2D":
            return euc_2d_edges(node_coords)
        elif problem.cost_function == "Manhatten2D":
            return man_2d_edges(node_coords)
        else:
            return "Only Geom, Euclidean2D, and Manhatten2D supported for now."
    
    # replace with your miner's forward method
    @abstractmethod
    def forward(self, synapse:GraphV2Synapse):
        ...
    
class DummyMiner(MockMiner):
    def __init__(self):
        super().__init__()

    def forward(self, synapse: GraphV2Synapse):
        if not isinstance(synapse, GraphV2Synapse):
            raise ValueError(f"Forward only handles GraphV2Synpase synapses")
        
        # recreate edges
        synapse.problem.edges = self.recreate_edges(synapse.problem)

        # separately handle TSP and mTSP problems
        if synapse.problem.problem_type == "Metric TSP":
            solution = MockTSPSolver().solve_problem(synapse.problem)
        else:
            solution = MockMTSPSolver().solve_problem(synapse.problem)
        synapse.solution = solution
        # empty out the nodes and edges to reduce synapse size
        synapse.problem.nodes = None
        synapse.problem.edges = None
        return synapse

class SimulatedValidator:
    def __init__(self, communication_time_discount:float=1.5):
        '''
        communication_time_discount: float --> is the simulated number of seconds for the sending the synapse and receiving the response
        '''
        load_default_dataset(self)
        self.time_discount = communication_time_discount
        self.miners = []

    def recreate_edges(self, problem:Union[GraphV2Problem, GraphV2ProblemMulti]):
        # get coordinates
        node_coords_np = self.loaded_datasets[problem.dataset_ref]["data"]
        node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
        if problem.cost_function == "Geom":
            return geom_edges(node_coords)
        elif problem.cost_function == "Euclidean2D":
            return euc_2d_edges(node_coords)
        elif problem.cost_function == "Manhatten2D":
            return man_2d_edges(node_coords)
        else:
            return "Only Geom, Euclidean2D, and Manhatten2D supported for now."
    
    def register_mock_miner(self, miner:MockMiner):
        if miner not in self.miners:
            self.miners.append(miner)

    def get_recreation_time(self, synapse:GraphV2Synapse):
        start_time = time.time()
        self.recreate_edges(synapse.problem)
        recreation_time = time.time() - start_time
        return recreation_time
        
    def mock_request(self, synapse, allocated_time):
        miner_solutions = []
        elapsed_times = [] # mock the elapsed time of the miners. Note 
        # synchronous calls here as the tested miners might have
        for miner in self.miners:
            start_time = time.time()
            miner_solutions.append(miner.forward(synapse).solution)
            elapsed_time = time.time() - start_time
            # score response
            if elapsed_time > allocated_time:
                print(f"Miner of class: {miner.__class___.__name__} timed out on synapse: {synapse.problem.problem_type} {synapse.problem.n_nodes} Nodes")
            elapsed_times.append(elapsed_time)
        return miner_solutions, elapsed_times

    def score_request(self, synapse, performance_dict, miner_solutions, elapsed_times, allocated_time):
        # score the synapses
        distances = []
        rewards = []
        remarks = []
        for solution, elapsed_time in zip(miner_solutions, elapsed_times):
            synapse.solution = solution
            miner_distance, reward, remark = self.score_response(synapse, performance_dict, elapsed_time, allocated_time)
            distances.append(miner_distance)
            rewards.append(reward)
            remarks.append(remark)
        return distances, rewards, remarks
            
    def simulate_one_request(self, synapse, performance_dict):
        allocated_time = 30 + self.get_recreation_time(synapse) - self.time_discount
        miner_solutions, elapsed_times = self.mock_request(synapse, allocated_time)
        synapse.problem.edges = self.recreate_edges(synapse.problem)
        distances, rewards, remarks = self.score_request(synapse, performance_dict, miner_solutions, elapsed_times, allocated_time)
        return distances, rewards, remarks

    def run_full_simulation(self, miners, sample_size, problem_type, load_from_prev=False, sim_data_file=None):
        '''
        sim_data_file: base name of simulation data
        '''
        for miner in miners:
            self.register_mock_miner(miner)

        if load_from_prev and sim_data_file != None:
            simulation_data = load_from_past_dataset(sim_data_file)
        else:
            # Default behavior is to save the data; Change per your needs
            simulation_data, sim_data_file = get_random_sample(sample_size=sample_size, problem_type=problem_type, save=True)

        simulation_results = {
            run_id:{"rewards": None,
                    "remarks": None,
                    "distances": None,
                    "miners": [miner.__class__.__name__ for miner in self.miners.copy()]}
            for run_id in simulation_data.keys()
        }
        for run_id, problem_info in tqdm(simulation_data.items(), desc=f'Running simulation for {[miner.__class__.__name__ for miner in self.miners]}'):
            synapse = problem_info["synapse"]
            performance_dict = problem_info["perf_data_dict"]
            distances, rewards, remarks = self.simulate_one_request(synapse, performance_dict)
            simulation_results[run_id]["rewards"] = rewards
            simulation_results[run_id]["remarks"] = remarks
            simulation_results[run_id]["distances"] = [float(dist) if np.isfinite(dist) else 'inf' for dist in distances ]
        
        # save simulation_results
        if sim_data_file != None:
            save_file_name = Path(os.path.basename(sim_data_file).replace(".json","_results.json"))
        else:
            save_file_name = f"{len(simulation_data)}_{simulation_data[list(simulation_data.keys())[0]]['synapse'].problem.problem_type}_results.json"
        with open(SIM_RES_DIR / save_file_name, "w") as f:
            json.dump(simulation_results, f)


    def score_response(self, synapse, performance_dict, elapsed_time, allocated_time):
        '''
        As we already know the best and worst score from the subnet, we can compute the simulated rewards of a given solution
        '''
        # recreate the edges of the received synapse
        try:
            if synapse.problem.problem_type == "Metric TSP":
                solution_distance = get_tour_distance(synapse)
            else:
                solution_distance = get_multi_minmax_tour_distance(synapse)
        except ValueError as e:
            print(f"Encountered error in scoring solution: {e}")
            return np.inf, 0, "invalid solution"
        except AssertionError as e:
            print(f"Encountered error in scoring solution: {e}")
            return np.inf, 0, "invalid solution"
        
        best_distance = performance_dict["best_solution"]
        worst_distance = performance_dict["worst_solution"]
        if elapsed_time > allocated_time:
            return solution_distance, 0, "timed out"
        if not np.isfinite(solution_distance) or worst_distance < solution_distance:
            return solution_distance, 0, "worse than subnet worst solution"
        elif solution_distance < best_distance:
            return solution_distance, 1, "beat subnet best solution"
        else:
            return solution_distance,(1 - (solution_distance - best_distance)/(worst_distance - best_distance))*0.8 + 0.2, "valid solution"
        
def run_test(sim_data_file, dummy_miner, sim_vali):
    # test score response function for a simple mTSP
    # Construct the synapse and run the dummy miner
    simulation_data = load_from_past_dataset(sim_data_file)
    problem_info = simulation_data[list(simulation_data.keys())[0]] # just select the first problem for testing purposes
    synapse = problem_info["synapse"]
    synapse.problem.edges = sim_vali.recreate_edges(synapse.problem)
    test_solution = dummy_miner.solve_problem(synapse.problem)
    synapse.solution = test_solution

    if synapse.problem.problem_type == "Metric TSP":
        test_distance = get_tour_distance(synapse)
    else:
        test_distance = get_multi_minmax_tour_distance(synapse)

    # Test 1: testing for solution worst than subnet worst
    performance_dict = {
        "best_solution":test_distance - 1000,
        "worst_solution": test_distance - 100
    }
    elapsed_time = 20
    allocated_time = 30
    test_distance, test_reward, test_remark = sim_vali.score_response(synapse, performance_dict, elapsed_time, allocated_time)
    assert test_reward == 0 and test_remark == "worse than subnet worst solution", f"got reward of: {test_reward} with remark: {test_remark}"
    print("Test 1 Passed")

    # Test 2: testing for solution better than subnet best
    performance_dict = {
        "best_solution":test_distance + 100,
        "worst_solution": test_distance + 1000
    }
    elapsed_time = 20
    allocated_time = 30
    test_distance, test_reward, test_remark = sim_vali.score_response(synapse, performance_dict, elapsed_time, allocated_time)
    assert test_reward == 1 and test_remark == "beat subnet best solution", f"got reward of: {test_reward} with remark: {test_remark}"
    print("Test 2 Passed")

    # Test 3: testing for solution timeout
    performance_dict = {
        "best_solution":test_distance + 100,
        "worst_solution": test_distance + 1000
    }
    elapsed_time = 30
    allocated_time = 20
    test_distance, test_reward, test_remark = sim_vali.score_response(synapse, performance_dict, elapsed_time, allocated_time)
    assert test_reward == 0 and test_remark == "timed out", f"got reward of: {test_reward} with remark: {test_remark}"
    print("Test 3 Passed")

    # Test 4: scoreable solution
    performance_dict = {
        "best_solution":test_distance - 500,
        "worst_solution": test_distance + 500
    }
    elapsed_time = 20
    allocated_time = 30
    test_distance, test_reward, test_remark = sim_vali.score_response(synapse, performance_dict, elapsed_time, allocated_time)
    assert test_reward == 0.6 and test_remark == "valid solution", f"got reward of: {test_reward} with remark: {test_remark}"
    print("Test 4 Passed")

    # Test 5: scoreable solution
    wrong_synapse = copy.deepcopy(synapse)
    wrong_synapse.solution.pop() # remove a random element from solution (this will render an invalid solution for both TSP and mTSP)
    performance_dict = {
        "best_solution":test_distance - 500,
        "worst_solution": test_distance + 500
    }
    elapsed_time = 20
    allocated_time = 30
    test_distance, test_reward, test_remark = sim_vali.score_response(wrong_synapse, performance_dict, elapsed_time, allocated_time)
    assert test_reward == 0 and test_remark == "invalid solution", f"got reward of: {test_reward} with remark: {test_remark}"
    print("Test 5 Passed")
    
if __name__=="__main__":
    # Change this code per your defined miners for testing
    # sync data to latest data
    sync_data()
    simulated_vali = SimulatedValidator()
    dummy_miner = DummyMiner()
    dummy_miner_2 = DummyMiner()
    simulated_vali.run_full_simulation([dummy_miner,dummy_miner_2], 30, "Metric mTSP", False, None)
    simulated_vali.run_full_simulation([dummy_miner,dummy_miner_2], 30, "Metric TSP", False, None)

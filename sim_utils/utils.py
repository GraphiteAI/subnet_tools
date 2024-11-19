from sim_utils.mock_protocol import GraphV2Problem, GraphV2ProblemMulti, GraphV2Synapse
from typing import List, Iterable
import numpy as np
import math

### Generic functions for neurons
def is_valid_path(path:List[int])->bool:
    # a valid path should have at least 3 return values and return to the source
    return (len(path)>=3) and (path[0]==path[-1])


def is_valid_multi_path(paths: List[List[int]], depots: List[int], num_cities)->bool:
    '''
    Arguments:
    paths: list of paths where each path is a list of nodes that start and end at the same node
    depots: list of nodes indicating the valid sources for each traveling salesman

    Output:
    boolean indicating if the paths are valid or not
    '''
    assert len(paths) == len(depots), ValueError("Received unequal number of paths to depots. Note that if you choose to not use a salesman, you must still return a corresponding empty path.")
    # check that each subpath is valid
    if not all([is_valid_path(path) for path in paths if path != []]):
        return False

    # check if the source nodes match the depots
    if not all([path[0]==depots[i] for i, path in enumerate(paths) if path != []]):
        return False
        
    # check that each city is only visited once across all salesmen
    all_non_depot_nodes = []
    for path in paths:
        if path != []:
            all_non_depot_nodes.extend(path[1:-1])
    assert len(all_non_depot_nodes) == len(set(all_non_depot_nodes)), ValueError("Duplicate Visits")
    assert set(all_non_depot_nodes) == set(list(range(1,num_cities))), ValueError("Invalid number of cities visited")
    return True

def get_tour_distance(synapse:GraphV2Synapse)->float:
    '''
    Returns the total tour distance for the TSP problem as a float.

    Takes a synapse as its only argument
    '''
    problem = synapse.problem
    if 'TSP' not in problem.problem_type:
        raise ValueError(f"get_tour_distance is an invalid function for processing {problem.problem_type}")
    
    if not synapse.solution:
        return np.inf
    distance=np.nan
    if isinstance(synapse.problem, GraphV2Problem):
        # This is a General TSP problem
        # check if path and edges are of the appropriate size
        edges=problem.edges
        if not isinstance(edges, Iterable):
            raise ValueError("User passed in invalid edges to scoring function. Please recreate edges and set synpase.problem.edges.")
        path=synapse.solution
        if isinstance(path,list):
            assert is_valid_path(path), ValueError('Provided path is invalid')
            assert len(path) == problem.n_nodes+1, ValueError('An invalid number of cities are contained within the provided path')

            distance = 0
            for i, source in enumerate(path[:-1]):
                destination = path[i+1]
                distance += edges[source][destination]
    return distance if not np.isnan(distance) else np.inf

def get_multi_minmax_tour_distance(synapse: GraphV2Synapse)->float:
    '''
    Returns the maximum tour distance across salesmen for the mTSP as a float.

    Takes a synapse as its only argument
    '''
    problem = synapse.problem
    if 'mTSP' not in problem.problem_type:
        raise ValueError(f"get_multi_tour_distance is an invalid function for processing {problem.problem_type}")

    if not synapse.solution:
        return np.inf
    distance=np.nan
    assert isinstance(problem, GraphV2ProblemMulti), ValueError(f"Attempting to use multi-path function for problem of type: {type(problem)}")
    
    edges=problem.edges
    if not isinstance(edges, Iterable):
        raise ValueError("User passed in invalid edges to scoring function. Please recreate edges and set synpase.problem.edges.")
    
    assert len(problem.edges) == len(problem.edges[0]) and len(problem.edges)==problem.n_nodes, ValueError(f"Wrong distance matrix shape of: ({len(problem.edges[0])}, {len(problem.edges)}) for problem of n_nodes: {problem.n_nodes}")
    paths=synapse.solution
    depots=problem.depots

    if isinstance(paths,list):
        assert is_valid_multi_path(paths, depots, problem.n_nodes), ValueError('Provided path is invalid')
        # assert len(path) == problem.n_nodes+1, ValueError('An invalid number of cities are contained within the provided path')
        distances = []
        for path in paths:
            distance = 0
            for i, source in enumerate(path[:-1]):
                destination = path[i+1]
                distance += edges[source][destination]
            distances.append(distance)
        max_distance = max(distances)
    return max_distance if not np.isnan(distance) else np.inf
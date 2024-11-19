'''
This file is a mock protocol of Graphite's Problem and Synapse
'''

from pydantic import BaseModel, Field, model_validator, conint, confloat, ValidationError, field_validator
from typing import List, Union, Optional, Literal, Iterable
import numpy as np
import math
import json
import base64
import sys
import random

class GraphV2Problem(BaseModel):
    problem_type: Literal['Metric TSP', 'General TSP'] = Field('Metric TSP', description="Problem Type")
    objective_function: str = Field('min', description="Objective Function")
    visit_all: bool = Field(True, description="Visit All Nodes")
    to_origin: bool = Field(True, description="Return to Origin")
    n_nodes: conint(ge=2000, le=5000) = Field(2000, description="Number of Nodes (must be between 2000 and 5000)")
    selected_ids: List[int] = Field(default_factory=list, description="List of selected node positional indexes")
    cost_function: Literal['Geom', 'Euclidean2D', 'Manhatten2D', 'Euclidean3D', 'Manhatten3D'] = Field('Geom', description="Cost function")
    dataset_ref: Literal['Asia_MSB', 'World_TSP'] = Field('Asia_MSB', description="Dataset reference file")
    nodes: Union[List[List[Union[conint(ge=0), confloat(ge=0)]]], Iterable, None] = Field(default_factory=list, description="Node Coordinates")  # If not none, nodes represent the coordinates of the cities
    edges: Union[List[List[Union[conint(ge=0), confloat(ge=0)]]], Iterable, None] = Field(default_factory=list, description="Edge Weights")  # If not none, this represents a square matrix of edges where edges[source;row][destination;col] is the cost of a given edge
    directed: bool = Field(False, description="Directed Graph")  # boolean for whether the graph is directed or undirected / Symmetric or Asymmetric
    simple: bool = Field(True, description="Simple Graph")  # boolean for whether the graph contains any degenerate loop
    weighted: bool = Field(False, description="Weighted Graph")  # boolean for whether the value in the edges matrix represents cost
    repeating: bool = Field(False, description="Allow Repeating Nodes")  # boolean for whether the nodes in the problem can be revisited

    ### Expensive check only needed for organic requests
    # @model_validator(mode='after')
    # def unique_select_ids(self):
    #     # ensure all selected ids are unique
    #     self.selected_ids = list(set(self.selected_ids))

    #     # ensure the selected_ids are < len(file)
    #     with np.load(f"dataset/{self.dataset_ref}.npz") as f:
    #         node_coords_np = np.array(f['data'])
    #         largest_possible_id = len(node_coords_np) - 1

    #     self.selected_ids = [id for id in self.selected_ids if id <= largest_possible_id]
    #     self.n_nodes = len(self.selected_ids)

    #     return self

    @model_validator(mode='after')
    def force_obj_function(self):
        if self.problem_type in ['Metric TSP', 'General TSP']:
            assert self.objective_function == 'min', ValueError('Subnet currently only supports minimization TSP')
        return self
    
    def get_info(self, verbosity: int = 1) -> dict:
        info = {}
        if verbosity == 1:
            info["Problem Type"] = self.problem_type
        elif verbosity == 2:
            info["Problem Type"] = self.problem_type
            info["Objective Function"] = self.objective_function
            info["To Visit All Nodes"] = self.visit_all
            info["To Return to Origin"] = self.to_origin
            info["Number of Nodes"] = self.n_nodes
            info["Directed"] = self.directed
            info["Simple"] = self.simple
            info["Weighted"] = self.weighted
            info["Repeating"] = self.repeating
        elif verbosity == 3:
            for field in self.model_fields:
                description = self.model_fields[field].description
                value = getattr(self, field)
                info[description] = value
        return info
    

# Constants for problem formulation
MAX_SALESMEN = 10


class GraphV2ProblemMulti(GraphV2Problem):
    problem_type: Literal['Metric mTSP', 'General mTSP'] = Field('Metric mTSP', description="Problem Type")
    n_nodes: conint(ge=500, le=2000) = Field(500, description="Number of Nodes (must be between 500 and 2000) for mTSP")
    n_salesmen: conint(ge=2, le=MAX_SALESMEN) = Field(2, description="Number of Salesmen in the mTSP formulation")
    # Note that in this initial problem formulation, we will start with a single depot structure
    single_depot: bool = Field(True, description="Whether problem is a single or multi depot formulation")
    depots: List[int] = Field([0,0], description="List of selected 'city' indices for which the respective salesmen paths begin")
    dataset_ref: Literal['Asia_MSB', 'World_TSP'] = Field('Asia_MSB', description="Dataset reference file")

    ### Expensive check only needed for organic requests
    # @model_validator(mode='after')
    # def unique_select_ids(self):
    #     # ensure all selected ids are unique
    #     self.selected_ids = list(set(self.selected_ids))

    #     # ensure the selected_ids are < len(file)
    #     with np.load(f"dataset/{self.dataset_ref}.npz") as f:
    #         node_coords_np = np.array(f['data'])
    #         largest_possible_id = len(node_coords_np) - 1

    #     self.selected_ids = [id for id in self.selected_ids if id <= largest_possible_id]
    #     self.n_nodes = len(self.selected_ids)

    #     return self
    @model_validator(mode='after')
    def assert_salesmen_depot(self):
        assert len(self.depots) == self.n_salesmen, ValueError('Number of salesmen must match number of depots')
        return self

    @model_validator(mode='after')
    def force_obj_function(self):
        if self.problem_type in ['Metric mTSP', 'General mTSP']:
            assert self.objective_function == 'min', ValueError('Subnet currently only supports minimization TSP')
        return self
    
    def get_info(self, verbosity: int = 1) -> dict:
        info = {}
        if verbosity == 1:
            info["Problem Type"] = self.problem_type
        elif verbosity == 2:
            info["Problem Type"] = self.problem_type
            info["Objective Function"] = self.objective_function
            info["To Visit All Nodes"] = self.visit_all
            info["To Return to Origin"] = self.to_origin
            info["Number of Nodes"] = self.n_nodes
            info["Directed"] = self.directed
            info["Simple"] = self.simple
            info["Weighted"] = self.weighted
            info["Repeating"] = self.repeating
        elif verbosity == 3:
            for field in self.model_fields:
                description = self.model_fields[field].description
                value = getattr(self, field)
                info[description] = value
        return info
    
class GraphV2Synapse(BaseModel):
    problem: Union[GraphV2Problem, GraphV2ProblemMulti]
    solution: Optional[Union[List[List[int]], List[int], bool]] = None
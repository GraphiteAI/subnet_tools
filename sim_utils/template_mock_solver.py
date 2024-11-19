from sim_utils.mock_protocol import GraphV2Problem, GraphV2ProblemMulti
import random

class MockTSPSolver():
    def __init__(self):
        pass

    def solve_problem(self, problem:GraphV2Problem):
        # return a random solution
        visit_order = list(range(1, problem.n_nodes))
        random.shuffle(visit_order)
        random_solution = [0] + visit_order + [0]
        return random_solution

class MockMTSPSolver():
    def __init__(self):
        pass

    def solve_problem(self, problem:GraphV2ProblemMulti):
        visit_order = list(range(1, problem.n_nodes))
        random.shuffle(visit_order)
        # partition
        base_size = len(visit_order) // problem.n_salesmen
        remainder = len(visit_order) % problem.n_salesmen

        partitions = []
        start = 0

        for i in range(problem.n_salesmen):
            # Distribute the remainder among the first 'remainder' partitions
            end = start + base_size + (1 if i < remainder else 0)
            partitions.append(visit_order[start:end])
            start = end

        return [[0] + partition + [0] for partition in partitions]
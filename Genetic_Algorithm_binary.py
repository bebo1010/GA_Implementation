import time
import random
import logging
import os

import numpy as np

from function import Function
from utils import generate_random_number_in_bounds, generate_random_bit_string

class Genetic_Algorithm_Binary(object):
    def __init__(self, function: callable, name_bounds_dict: dict, precision_point: int = 4, population_size: int = 1000, crossover_rate: float = 0.7, mutation_rate: float = 0.005, max_iterations: int = 500, relative_change_threshold: float = 0.0003, stopping_iteration_threshold: int = 10):
        """
        Constructor for Genetic Algorithm.\n
        args:
        `function` (callable): a function that is to be used, e.g. f(x) = sin(x), etc.
        `name_bounds_dict` (dict): a dictionary that contains the name of variables and their upper and lower bounds.\n
        `precision_point` (int): How much of the precision point of answer should be. Default value is 4.\n
        `population_size` (int): The number of population in Genetic Algorithm. Default value is 1000.\n
        `crossover_rate` (float): Probability of Crossover in Genetic Algorithm, should be within [0, 1]. Default value is 0.7.\n
        `mutation_rate` (float): Probability of Mutation in Genetic Algorithm, should be within [0, 1]. Default value is 0.005.\n
        `max_iterations` (int): Maximum number of iterations in Hill Climbing Algorithm. Default value is 500.\n
        `relative_change_threshold` (float): Threshold for relative changes between iterations. Default value is 0.0003.\n
        `stopping_iteration_threshold` (int): Threshold for early stopping. When more than this number of iteration has passed and relative change are all below threshold, terminate the algorithm. Default value is 10.\n
        returns:\n
        None

        Note:\n
        `name_bounds_dict` should be the in format as follows:\n
        {
        "x": [1, 2],
        "y": [0, 1]
        }\n
        The keys are the name of the variables and the values are the upper and lower bounds.\n

        """
        self.func = Function(function, name_bounds_dict)
        self.precision_point = precision_point
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_iterations = max_iterations
        self.relative_change_threshold = relative_change_threshold
        self.stopping_iteration_threshold = stopping_iteration_threshold

        self.output_dir = "./output/"
        if self.output_dir:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(self.output_dir + "Genetic_Algorithm_Binary_log.txt", mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

        self.bit_string_length = []
        for (bound_A, bound_B) in self.func.bounds:
            upper_bound = max(bound_A, bound_B)
            lower_bound = min(bound_A, bound_B)

            bits = np.ceil(np.log2((upper_bound - lower_bound) * np.power(10, precision_point) + 1))
            self.bit_string_length.append(int(bits))

        self.best_value_history = []

    def start_iteration(self):
        current_states = self.generate_initial_state()
        function_values = self.evaluate_all_states(current_states)

        last_iteration_value = 1e-10
        early_stopping_counter = 0
        epsilon = 1e-10
        for i in range(self.max_iterations):
            best_value_index, best_value = np.argmin(function_values), min(function_values)
            self.logger.info(f"iteration {i}: function value {best_value} with parameters {current_states[best_value_index]}")
            self.best_value_history.append(best_value)
            self.iteration_counter = i

            # termination criteria
            relative_change = abs(((best_value + epsilon) / (last_iteration_value + epsilon)) - 1)
            if relative_change < self.relative_change_threshold:  
                if early_stopping_counter >= self.stopping_iteration_threshold:
                    self.logger.info(f"Algorithm Terminated at iteration {i}, with relative change {relative_change * 100:.5f} %")
                    self.final_state = current_states[best_value_index]
                    self.final_value = best_value
                    return
                else: 
                    early_stopping_counter = early_stopping_counter + 1
            last_iteration_value = best_value

            parent_states = self.truncation_selection(current_states, function_values)
            child_states = self.single_point_crossover(parent_states)
            mutated_states = self.bitwise_mutation(child_states)

            current_states = mutated_states
            function_values = self.evaluate_all_states(current_states)

        best_value_index, best_value = np.argmin(function_values), min(function_values)
        self.final_state = current_states[best_value_index]
        self.final_value = best_value

    def decode(self, bit_string_states: list[dict]) -> list[dict]:
        """Decode bit strings to floating point.
        
        Args:
            bit_string_states (list[dict]): Current states, encoded in bit strings.

        Returns:
            list[dict]: Current states, decoded to floating points.
        """
        def bit_string_decoder(bit_string: list, precision_point: int) -> float:
            """Decodes a bit string to a floating point number.

            Args:
                bit_string (list): Bit string encoded in Little-Endian.
                precision_point (int): Precision point of output floating point number.

            Returns:
                float: Decoded floating point number.
            """
            result = 0
            constant = 10**(-precision_point)
            for index, bit in enumerate(bit_string):
                result += (bit << index) 

            return result * constant

        decoded_states = []
        for state in bit_string_states:
            decoded_state = {}
            for variable, bit_string in state.items():
                # print(bit_string, bit_string_decoder(bit_string, self.precision_point))
                decoded_state[variable] = bit_string_decoder(bit_string, self.precision_point)

            decoded_states.append(decoded_state)

        return decoded_states

    def generate_initial_state(self) -> list[dict]:
        """
        Generates the initial states, length of list is determined by `self.population_size`.

        args:
            None
        returns:
            `list[dict]`: initial states

        Note:
        Initial states are generated using uniform distribution in the given bounds.
        """
        states_list = []

        for population in range(self.population_size):
            state = {}
            for variable, bits_length in zip(self.func.variable_names, self.bit_string_length):
                state[variable] = generate_random_bit_string(bits_length)

            states_list.append(state)

        return states_list

    def evaluate_all_states(self, states_list: list[dict]) -> list:
        """
        Evaluate all the states.

        args:
            `states_list` (list[dict]): current states.
        returns:
            `list`: list of values that contains all the function values of states.

        Note:\n
        Argument `states_list` and output list will be index aligned.
        """
        function_values_list = []

        decoded_states = self.decode(states_list)
        for state in decoded_states:
            function_values_list.append(self.func(*(state.values()), default_value=1e10))

        return function_values_list

    def truncation_selection(self, states_list: list[dict], function_values_list: list) -> list[dict]:
        """
        Perform truncation selection.

        Args:
            states_list (list[dict]): Current states.
            function_values_list (list): List of function values corresponding to states.

        Returns:
            list[dict]: Selected parent states.
        """
        selected_parents = []

        # Sort individuals based on function values in descending order
        sorted_indices = sorted(range(len(function_values_list)), key=lambda i: function_values_list[i])
        num_selected = int(0.3 * len(states_list))  # Select top 30% of individuals

        # Select top 30% of individuals as parents
        for i in range(num_selected):
            selected_parents.append(states_list[sorted_indices[i]])

        return selected_parents

    def single_point_crossover(self, parent_states_list: list[dict]) -> list[dict]:
        """
        Perform single point crossover in binary string.

        args:
            `parent_states_list` (list[dict]): parent states.

        returns:
            `list[dict]`: child states

        Note:\n

        """
        child_states_list = []
        for population in range(self.population_size):
            parent1, parent2 = random.sample(parent_states_list, 2)

            crossover_happening = generate_random_number_in_bounds(0.0, 1.0)
            
            if crossover_happening <= self.crossover_rate:
                child_state = {}
                for variable, bit_length in zip(self.func.variable_names, self.bit_string_length):
                    crossover_point = random.randint(0, bit_length - 1)
                    child_state[variable] = []
                    for bit in range(bit_length):
                        if bit < crossover_point:
                            child_state[variable].append(parent1[variable][bit])
                        else:
                            child_state[variable].append(parent2[variable][bit])

            else:
                child_state = parent1.copy()

            child_states_list.append(child_state)

        return child_states_list

    def bitwise_mutation(self, states_list: list[dict]) -> list[dict]:
        """
        Perform bitwise mutation in bit string.

        args:
            `parent_states_list` (list[dict]): original states.

        returns:
            `list[dict]`: mutated states

        Note:\n
        """

        mutated_states_list = []
        for state in states_list:
            mutation_happening = generate_random_number_in_bounds(0.0, 1.0)
            result_state = state.copy()

            if mutation_happening <= self.mutation_rate:
                
                for variable, bit_length in zip(self.func.variable_names, self.bit_string_length):
                    mutation_point = random.randint(0, bit_length - 1)

                    result_state[variable][mutation_point] = abs(result_state[variable][mutation_point] - 1)

            mutated_states_list.append(result_state)

        return mutated_states_list

if __name__ == "__main__":
    evaluating_performance = False
    # Define a sample function
    def my_function(x, y):
        return np.power((np.square(x) + np.square(y)), 0.25) * (np.square(np.sin(50 * np.power((np.square(x) + np.square(y)), 0.1))) + 1)

    # Define the bounds for variables
    variable_bounds = {
        "x": [0, 1],
        "y": [0, 1]
    }

    def run_algorithm_repeatly(function, variable_bounds, repeat_times):
        final_function_values = []
        execution_times = []

        for _ in range(repeat_times):
            start_time = time.time()
            GA_binary_object = Genetic_Algorithm_Binary(function, variable_bounds)
            GA_binary_object.start_iteration()
            elapsed_time = time.time() - start_time

            final_function_values.append(GA_binary_object.final_value)
            execution_times.append(elapsed_time)

        final_function_values = np.array(final_function_values)
        execution_times = np.array(execution_times)

        mean_function_value = np.mean(final_function_values)
        variance_function_value = np.var(final_function_values)
        mean_execution_time = np.mean(execution_times)
        variance_execution_time = np.var(execution_times)

        return mean_function_value, variance_function_value, mean_execution_time, variance_execution_time

    if evaluating_performance:
        mean_function_value, variance_function_value, mean_execution_time, variance_execution_time = run_algorithm_repeatly(my_function, variable_bounds, 100)
        print("Mean of final function value:", mean_function_value)
        print("Variance of final function value:", variance_function_value)
        print("Mean of execution time:", mean_execution_time, "seconds")
        print("Variance of execution time:", variance_execution_time, "seconds")
    else:

        start_time = time.time()
        GA_binary_object = Genetic_Algorithm_Binary(my_function, variable_bounds)
        GA_binary_object.start_iteration()

        elapsed_time = time.time() - start_time
        GA_binary_object.logger.info(f"Final: function value {GA_binary_object.final_value} with parameters {GA_binary_object.final_state}, elapsed time: {elapsed_time} seconds")
        print(f"My Final Function Value: {GA_binary_object.final_value} with parameters {GA_binary_object.final_state}, elapsed time: {elapsed_time} seconds")

import time
import random
import logging
import os

import numpy as np
import matplotlib.pyplot as plt

from function import Function
from utils import generate_random_number_in_bounds

class Genetic_Algorithm_Float(object):
    def __init__(self, function: callable, name_bounds_dict: dict, population_size: int = 1000, crossover_rate: float = 0.95, mutation_rate: float = 0.05, max_iterations: int = 500, relative_change_threshold: float = 0.0003, stopping_iteration_threshold: int = 10):
        """
        Constructor for Genetic Algorithm.\n
        args:
        `function` (callable): a function that is to be used, e.g. f(x) = sin(x), etc.
        `name_bounds_dict` (dict): a dictionary that contains the name of variables and their upper and lower bounds.\n
        `population_size` (int): The number of population in Genetic Algorithm. Default value is 1000.\n
        `crossover_rate` (float): Probability of Crossover in Genetic Algorithm, should be within [0, 1]. Default value is 0.95.\n
        `mutation_rate` (float): Probability of Mutation in Genetic Algorithm, should be within [0, 1]. Default value is 0.05.\n
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
        file_handler = logging.FileHandler(self.output_dir + "Genetic_Algorithm_Float_max_log.txt", mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

        self.search_step_size = 0.0001
        self.best_value_history = []

    def start_iteration(self):
        current_states = self.generate_initial_state()
        function_values = self.evaluate_all_states(current_states)

        last_iteration_value = 1e-10
        early_stopping_counter = 0
        for i in range(self.max_iterations):
            best_value_index, best_value = np.argmax(function_values), max(function_values)
            self.logger.info(f"iteration {i}: function value {best_value} with parameters {current_states[best_value_index]}")
            self.best_value_history.append(best_value)
            self.iteration_counter = i

            relative_change = abs((best_value / last_iteration_value) - 1)
            if relative_change < self.relative_change_threshold:  
                if early_stopping_counter >= self.stopping_iteration_threshold:
                    self.logger.info(f"Algorithm Terminated at iteration {i}, with relative change {relative_change * 100:.5f} %")
                    self.final_state = current_states[best_value_index]
                    self.final_value = best_value
                    return
                else: 
                    early_stopping_counter = early_stopping_counter + 1
            last_iteration_value = best_value

            parent_states = self.tournament_selection(current_states, function_values)
            child_states = self.single_point_crossover(parent_states)
            mutated_states = self.gaussian_mutation(child_states)

            current_states = mutated_states
            function_values = self.evaluate_all_states(current_states)

        best_value_index, best_value = np.argmax(function_values), max(function_values)
        self.final_state = current_states[best_value_index]
        self.final_value = best_value

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
            for variable, bounds in zip(self.func.variable_names, self.func.bounds):
                state[variable] = generate_random_number_in_bounds(*bounds)

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
        for state in states_list:
            function_values_list.append(self.func(*(state.values())))

        return function_values_list

    def tournament_selection(self, states_list: list[dict], function_values_list: list, tournament_size: int = 10) -> list[dict]:
        """
        Perform tournament selection.

        args:
            `states_list` (list[dict]): current states.
            `function_values_list` (list): list of values that contains all the function values of states.
            `tournament_size` (int): size of the tournament, i.e. how many individuals are compared to each other. Default value is 10.

        returns:
            `list[dict]`: parent states

        Note:\n
        Select the best performing individual as the parent in one tournament.
        """

        selected_parents = []
        for population in range(self.population_size // tournament_size):
            # Randomly sample individuals for the tournament
            tournament_indices = random.sample(range(len(states_list)), tournament_size)
            
            # Get the corresponding function values for the sampled individuals
            tournament_function_values = [function_values_list[i] for i in tournament_indices]
            
            # Find the index of the fittest individual in the tournament
            fittest_index = tournament_indices[tournament_function_values.index(max(tournament_function_values))]
            
            # Add the fittest individual to the list of selected parents
            selected_parents.append(states_list[fittest_index])

        return selected_parents

    def single_point_crossover(self, parent_states_list: list[dict]) -> list[dict]:
        """
        Perform single point crossover in floating point.

        args:
            `parent_states_list` (list[dict]): parent states.

        returns:
            `list[dict]`: child states

        Note:\n
        Crossover is implemented using two random numbers.\n
        First one (`crossover_happening`) controls whether the crossover happens.\n

        Second one (`crossover_point`) controls how does the crossover happen.\n
        If this number is smaller than 0.5, select `parent1` as child; else, select `parent2` as child. 
        """

        child_states_list = []
        for population in range(self.population_size):
            parent1, parent2 = random.sample(parent_states_list, 2)

            crossover_happening = generate_random_number_in_bounds(0.0, 1.0)
            crossover_point = generate_random_number_in_bounds(0.0, 1.0)

            if crossover_happening <= self.crossover_rate:
                child_state = {}
                for variable, bounds in zip(self.func.variable_names, self.func.bounds):
                    child_state[variable] = parent1[variable] if crossover_point <= 0.5 else parent2[variable]

            else:
                child_state = parent1.copy() if crossover_point <= 0.5 else parent2.copy()

            child_states_list.append(child_state)

        return child_states_list

    def gaussian_mutation(self, states_list: list[dict]) -> list[dict]:
        """
        Perform gaussian mutation in floating point.

        args:
            `parent_states_list` (list[dict]): original states.

        returns:
            `list[dict]`: mutated states

        Note:\n
        Mutation is implemented using two random numbers.\n
        First one (`mutation_happening`) controls whether the mutation happens.\n

        Second one (`random_gaussian_noise`) controls how much does the mutation happen.\n
        `random_gaussian_noise` will be added to the original state variables.
        """

        mutated_states_list = []
        for state in states_list:
            mutation_happening = generate_random_number_in_bounds(0.0, 1.0)

            if mutation_happening <= self.mutation_rate:
                result_state = {}
                for variable, bounds in zip(self.func.variable_names, self.func.bounds):
                    random_gaussian_noise = random.gauss(0, 0.1)

                    result = state[variable] + random_gaussian_noise
                    # clip back to bounds
                    result = np.clip(result, *bounds)
                    result_state[variable] = result

            else:
                result_state = state.copy()

            mutated_states_list.append(result_state)

        return mutated_states_list

    def display_results(self):
        """
        Display the results of the genetic algorithm.

        This function plots the best function value achieved over iterations.

        Args:
            None

        Returns:
            None
        """
        x = np.arange(0, (self.iteration_counter + 1))
        y = self.best_value_history

        # Plot the best function value over iterations
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o', color='b', linestyle='-')
        plt.title('Genetic Algorithm with Floating Point: Best Function Value over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Best Function Value')
        plt.grid(True)

        # Set x ticks to be every 1
        plt.xticks(np.arange(min(x), max(x)+1, 1))

        plt.savefig(os.path.join(self.output_dir, 'genetic_algorithm_floating_max_plot.png'))

        plt.show()

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
            GA_float_object = Genetic_Algorithm_Float(function, variable_bounds)
            GA_float_object.start_iteration()
            elapsed_time = time.time() - start_time

            final_function_values.append(GA_float_object.final_value)
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
        GA_float_object = Genetic_Algorithm_Float(my_function, variable_bounds)
        GA_float_object.start_iteration()

        elapsed_time = time.time() - start_time
        GA_float_object.logger.info(f"Final: function value {GA_float_object.final_value} with parameters {GA_float_object.final_state}, elapsed time: {elapsed_time} seconds")
        print(f"My Final Function Value: {GA_float_object.final_value} with parameters {GA_float_object.final_state}, elapsed time: {elapsed_time} seconds")

        GA_float_object.display_results()
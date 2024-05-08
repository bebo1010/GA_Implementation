import inspect

import numpy as np

class Function(object):
    def __init__(self, function: callable, name_bounds_dict: dict):
        """
        Constructor for basic Function.\n
        args:
        `function` (callable): a function that is to be used, e.g. f(x) = sin(x), etc.
        `name_bounds_dict` (dict): a dictionary that contains the name of variables and their upper and lower bounds.\n
        
        returns:\n
        None

        Note:\n
        `name_bounds_dict` should be the format as follows:\n
        {
        "x": [1, 2],
        "y": [0, 1]
        }\n
        The keys are the name of the variables and the values are the upper and lower bounds.
        """

        self.function = function
        self.variable_names = list(name_bounds_dict.keys())
        self.bounds = list(name_bounds_dict.values())

        function_arg_count = len(inspect.signature(self.function).parameters)
        assert function_arg_count == len(self.variable_names), "total arguments in functions and variables does not match."

        # Validate that the variable names in the dictionary match the function arguments
        function_arg_names = list(inspect.signature(self.function).parameters.keys())
        assert set(self.variable_names) == set(function_arg_names), "Variable names in dictionary do not match function arguments."


    def __call__(self, *args, default_value = 0):
        """
        Calculate the value of the function.
        
        Args:
            *args: Values of the function's variables.
            default_value (float): if the arguments are out of bounds, return default value. Default is 0.

        Returns:
            value: the value of the function using given args.

        Note:
        """
        # Check if the number of arguments matches the number of variables
        assert len(args) == len(self.variable_names), "Number of arguments provided does not match number of variables."

        # Check if arguments are within bounds
        for arg, (lower, upper) in zip(args, self.bounds):
            if not (lower - 0.001) <= arg <= (upper + 0.001):
                return default_value

        return self.function(*args)
    
    def first_derivative(self, *args, h=0.00001):
        """
        Calculate the first derivatives of the function using the central difference approximation.
        
        Args:
            *args: Values of the function's variables.
            h (float, optional): Step size for numerical differentiation. Default is 0.00001.

        Returns:
            list: List of first derivatives of the function with respect to each variable.

        Note:
            The central difference formula is used to approximate the derivatives:
            derivative_i = (f(x_i + h) - f(x_i - h)) / (2 * h)
        """
        derivatives = []
        for i, _ in enumerate(*args):
            args_list = list(*args)
            args_list[i] += h
            args_plus = tuple(args_list)
            
            args_list[i] -= 2 * h  # Subtracting 2*h instead of h to get args_minus
            args_minus = tuple(args_list)
            
            derivative = (self.__call__(*args_plus) - self.__call__(*args_minus)) / (2 * h)
            derivatives.append(derivative)
        return derivatives

if __name__ == "__main__":
    # Define a sample function
    def my_function(x, y):
        return np.power((np.square(x) + np.square(y)), 0.25) * (np.square(np.sin(50 * np.power((np.square(x) + np.square(y)), 0.1))) + 1)

    # Define the bounds for variables
    variable_bounds = {
        "x": [0, 1],
        "y": [0, 1]
    }

    # Create an instance of the Function class
    func = Function(my_function, variable_bounds)

    # Test the function with valid arguments
    valid_args = (0.5, 1)
    result = func(*valid_args)
    print("Result with valid arguments:", result)

    derivatives = func.first_derivative(valid_args)
    print("Derivative with valid arguments:", derivatives)

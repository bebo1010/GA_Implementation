import random

def generate_random_number_in_bounds(lower_bound: float, upper_bound: float) -> float:
    """
    Generates a random number within the bounds using uniform distribution.
    args:
        `lower_bound`: the lower bound of the random number.
        `upper_bound`: the upper bound of the random number

    returns:
        float: the random number within the bounds.
    """

    return random.uniform(lower_bound, upper_bound)

def generate_random_bit_string(bits_length: int) -> list:
    """
    Generates a random bit string using Little-Endian.
    args:
        `bits_length`: How long the bit string should be

    returns:
        int: randomly generated bit string.
    """
    bit_string = []
    for _ in range(bits_length):
        bit_string.append(random.randint(0, 1))

    return bit_string

if __name__ == '__main__':
    print(generate_random_number_in_bounds(0, 1))
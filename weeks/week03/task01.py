import numpy as np

def main():
    def create_dataset(n):
        return np.ndarray([(x, x*2) for x in range(n)])

    def initialize_weights(lower, upper, values):
        return np.random.Generator.uniform(lower, upper, values)
    
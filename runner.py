import numpy as np

import parameters as par
import data_generator as dg


def main():
    np.set_printoptions(precision=5)
    parameters = par.Parameters()
    data_generator = dg.DataGenerator(parameters)
    durations, resources_requirements = data_generator.generate_sequence()

    t = 1
    for duration, resource_requirement in zip(durations, resources_requirements):
        print("Job %d has duration %d and resource requirements:" % (t, duration))
        print(resource_requirement)

if __name__ == '__main__':
    main()
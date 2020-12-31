import numpy as np

import parameters as par
import data_generator as dg


def main():
    np.set_printoptions(precision=5)
    parameters = par.Parameters()
    data_generator = dg.DataGenerator(parameters)

    for k in range(5):
        duration, resource_requirements = data_generator.generate_job()
        print("Job %s generated with duration %s and resource requiremenets:" % (k, duration))
        print(resource_requirements)

if __name__ == '__main__':
    main()
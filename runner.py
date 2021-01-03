import numpy as np
import matplotlib.pyplot as plt

import parameters as par
import data_generator as dg
from environment import ResourceManagementEnv
from job import Job


def main():
    np.set_printoptions(precision=5)
    parameters = par.Parameters()
    # simple environment test execute several step and observe bahaviour
    env = ResourceManagementEnv(parameters, True)
    env.step(0)
    env.render()
    env.step(1)
    env.render()
    env.step(0)
    env.render()
    env.step(1)
    env.render()
    work_sequences = env.generate_work_sequences()
    job_lengths = np.array([[job.length if job is not None else 0 for job in seq] for seq in work_sequences], dtype=int)
    job_lengths = job_lengths.flatten()
    print(job_lengths)

    plt.hist(job_lengths)
    plt.title("Job lengths distribution")
    plt.xlabel("Job length")
    plt.show()

    # for i in range(work_sequences.shape[0]):
    #     for j, job in enumerate(work_sequences[i]):
    #         if job is not None:
    #             print("Job id: %d, duration: %d, resource vector: " % (job.id, job.length))
    #             print(job.resource_vector)
    #         else:
    #             print("Job %d is None" % ((i+1) * (j+1)))


if __name__ == '__main__':
    main()
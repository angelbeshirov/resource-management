from job import Job
from machine import Machine
from logger import Logger, LogLevel
from specific_agents import PackerAgent
from test_parameters import TestParameters
from environment import ResourceManagementEnv, TerminationType
import numpy as np
import pytest
import copy

def generate_work_sequence():
    job_lengths = np.array([3, 8, 50, 20, 11])
    job_resource_vectors = np.array([
        [10, 14],
        [11, 5],
        [2, 5],
        [23, 4],
        [25, 25],
    ])
    assert len(job_lengths) == len(job_resource_vectors), "Job lengths and resource vectors must be the same number"
    work_sequence = np.full((1, len(job_lengths)), None, dtype=object)

    for i in range(len(job_lengths)):
        job = Job(job_resource_vectors[i], \
            job_lengths[i], i + 1)
        work_sequence[0,i] = job 
    return work_sequence

def test_packer_action():
    """
    [10, 14] * [40, 36] = 904
    [11, 5] * [39, 45] = 654
    [2, 5] * [48, 45] = 321
    [23, 4] * [37, 46] = 1035
    [25, 25] * [25, 25] = 1250
    """
    work_sequences = generate_work_sequence()
    parameters = TestParameters()
    logger = Logger(LogLevel['info'])
    machine = Machine(2, 50, 60, logger)
    env = ResourceManagementEnv(parameters, logger, to_render=False, termination_type=TerminationType.AllJobsDone)
    env.machine = machine
    
    packer = PackerAgent(parameters, env, logger)

    work_queue = work_sequences[0]

    packer_action = packer.predict(work_queue)
    print("The packer agent picked job with id: %d" % packer_action)

    print("Dot products are:")
    print("Job 0 with dot product %d" % np.dot([10, 14], [40, 36]))
    print("Job 1 with dot product %d" % np.dot([11, 5], [39, 45]))
    print("Job 2 with dot product %d" % np.dot([2, 5], [48, 45]))
    print("Job 3 with dot product %d" % np.dot([23, 4], [37, 46]))
    print("Job 4 with dot product %d" % np.dot([25, 25], [25, 25]))

    assert packer_action == 4

def test_packer_invalid_action():
    work_sequences = generate_work_sequence()
    logger = Logger(LogLevel['info'])
    parameters = TestParameters()
    machine = Machine(2, 50, 60, logger)
    env = ResourceManagementEnv(parameters, logger, to_render=False, termination_type=TerminationType.AllJobsDone)
    env.machine = machine
    
    packer = PackerAgent(parameters, env, logger)

    work_queue = work_sequences[0]

    for job in work_queue:
        job.length = 61

    packer_action = packer.predict(work_queue)
    print(packer_action)

if __name__ == '__main__':
    test_packer_action()
    test_packer_invalid_action()




from job import Job
from machine import Machine
from logger import Logger, LogLevel
from specific_agents import PackerAgent, SJFAgent
from test_parameters import TestParameters
from environment import ResourceManagementEnv, TerminationType
import numpy as np
import pytest
import copy

def generate_work_sequence():
    job_lengths = np.array([[3, 8, 50, 20, 11], [24, 3, 11, 2, 18]])
    job_resource_vectors = np.array([[
        [10, 14],
        [11, 5],
        [2, 5],
        [23, 4],
        [25, 25],
    ], 
    [
        [10, 14],
        [11, 5],
        [2, 5],
        [23, 4],
        [25, 25],
    ]])
    assert len(job_lengths[0]) == len(job_resource_vectors[0]), "Job lengths and resource vectors must be the same number for ech sequence"
    assert len(job_lengths) == len(job_resource_vectors), "Work sequences must be the same number"
    work_sequence = np.full((2, len(job_lengths[0])), None, dtype=object)

    for i in range(len(job_lengths)):
        for j in range(len(job_lengths[0])):
            job = Job(job_resource_vectors[i, j], \
                job_lengths[i, j], i * len(job_lengths[0]) + j + 1)
            work_sequence[i,j] = job 
    return work_sequence

def generate_empty_work_sequence():
    return np.full((1, 5), None)

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


def test_sjf_action():
    work_sequences = generate_work_sequence()
    parameters = TestParameters()
    logger = Logger(LogLevel['info'])
    machine = Machine(2, 50, 60, logger)
    env = ResourceManagementEnv(parameters, logger, to_render=False, termination_type=TerminationType.AllJobsDone)
    env.machine = machine
    sjf_agent = SJFAgent(parameters, env, logger)

    for i in range(len(work_sequences)):
        work_queue = work_sequences[i]
        sjf_action = sjf_agent.predict(work_queue)    

        print("Sjf agent picked action with index: %d " % sjf_action)

        for j in range(len(work_queue)):
            print("Job {} with sjf score: {}".format(j, 1 / float(work_queue[j].length)))

        if i == 0:
            assert sjf_action == 0
        if i == 1:
            assert sjf_action == 3

def test_sjf_action_empty_queue():
    work_sequences = generate_empty_work_sequence()
    parameters = TestParameters()
    logger = Logger(LogLevel['info'])
    machine = Machine(2, 50, 60, logger)
    env = ResourceManagementEnv(parameters, logger, to_render=False, termination_type=TerminationType.AllJobsDone)
    env.machine = machine
    sjf_agent = SJFAgent(parameters, env, logger)

    work_queue = work_sequences[0]
    sjf_action = sjf_agent.predict(work_queue)
    assert sjf_action == len(work_queue), "Action should be invalid when work queue is empty"


if __name__ == '__main__':
    test_packer_action()
    test_packer_invalid_action()
    test_sjf_action()
    test_sjf_action_empty_queue()

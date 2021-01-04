from environment import ResourceManagementEnv, TerminationType
from job import Job
import test_parameters as pa
import numpy as np
import pytest
import copy

def generate_work_sequence():
    job_lengths = np.array([3, 8, 50, 20, 11, 1, 5, 55, 2, 7])
    job_resource_vectors = np.array([
        [10, 14],
        [11, 5],
        [2, 5],
        [23, 4],
        [25, 25],
        [12, 12],
        [5, 9],
        [14, 6],
        [13, 8],
        [5,5]
    ])
    assert len(job_lengths) == len(job_resource_vectors), "Job lengths and resource vectors must be the same number"
    work_sequence = np.full((1, len(job_lengths)), None, dtype=object)

    for i in range(len(job_lengths)):
        job = Job(job_resource_vectors[i], \
            job_lengths[i], i + 1)
        work_sequence[0,i] = job 
    return work_sequence

def test_environment():
    params = pa.TestParameters()
    env = ResourceManagementEnv(params, True, termination_type=TerminationType.AllJobsDone)
    env.work_sequences = generate_work_sequence()
    for i in range(params.work_queue_size):
        _, done = env.step(i)
        # env.render()
        assert done == False, "Simulation must not end before the whole sequence has been executed"
        assert env.current_time == (i+1), "Time step is %d, expected: %d"%(env.current_time, i+1)
        assert env.job_queue[i] == env.work_sequences[0, i], \
            "Env queue at pos %d, expected: %s, got: %s" % (i, env.work_sequences[0, i].to_string(), env.job_queue[i].to_string()) 

    # execute the job from the last slot
    _, done = env.step(4)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert env.job_queue[4] is None, "Slot must be free when allocating the job"
    work_sequence_idx = len(env.job_queue)

    # Add next job from sequence to queue
    _, done = env.step(4)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert env.job_queue[4] == env.work_sequences[0, work_sequence_idx], \
        "Add job in queue at pos 4, expected: %s got: %s" % (env.work_sequences[0, work_sequence_idx].to_string(), env.job_queue[4].to_string())
    work_sequence_idx += 1

    # try to allocate job for which there is not enough resources
    old_slot_job = copy.copy(env.job_queue[2])
    _, done = env.step(2)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert env.job_backlog.num_jobs == 1, "New job is added to backlog when action cannot be executed"
    assert env.job_queue[2] == old_slot_job, \
        "Job on the slot has changed. Expected: %s, got: %s" % (env.job_queue[2].to_string(), old_slot_job.to_string())
    assert np.all(env.job_queue) is not None, "There are no empty slots in environment queue"
    work_sequence_idx += 1

    # Allocate first resource
    backlog_job = env.job_backlog.front()
    _, done = env.step(0)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    # check if item from backlog is added to the queue at pos 0
    assert env.job_queue[0] == backlog_job, \
        "Job from backlog in queue. Expected: %s, got: %s" % (env.job_queue[0].to_string(), backlog_job.to_string())
    # Allocate resourse at position 1
    _, done = env.step(1)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    # slot should now be empty as no job is in backlog
    assert env.job_queue[1] is None, \
        "Job slot at position 1 is expected to be None, but is %s" % (env.job_queue[1].to_string())

    _, done = env.step(3)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    # slot should now be empty as no job is in backlog
    assert env.job_queue[3] is None, \
        "Job slot at position 1 is expected to be None, but is %s" % (env.job_queue[3].to_string())

    _, done = env.step(4)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    # slot should now be empty as no job is in backlog
    assert env.job_queue[4] is None, \
        "Job slot at position 1 is expected to be None, but is %s" % (env.job_queue[4].to_string())

    _, done = env.step(2)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    # Add next job from sequence
    _, done = env.step(2)
    assert done == False, "Simulation must not end before the whole sequence has been executed"

    # Add next work to the empty slots in the queue
    _, done = env.step(1)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert env.job_queue[2] == env.work_sequences[0, 8], \
        "Job from backlog in queue. Expected: %s, got: %s" % (env.job_queue[2].to_string(), env.work_sequences[0, 8].to_string())

    _, done = env.step(1)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert env.job_queue[3] == env.work_sequences[0, 9], \
        "Job from backlog in queue. Expected: %s, got: %s" % (env.job_queue[3].to_string(), env.work_sequences[0, 9].to_string())

    for _ in range(10):
        _, done = env.step(4)
        assert done == False, "Simulation must not end before the whole sequence has been executed"

    _, done = env.step(0)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    _, done = env.step(2)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    _, done = env.step(3)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    
    # Run 30 steps to ensure there is space for the last task
    for _ in range(30):
        _, done = env.step(4)
        assert done == False, "Simulation must not end before the whole sequence has been executed"
    # Allocate last task
    _, done = env.step(1)
    assert done == False, "Simulation must not end before the whole sequence has been executed"

    for _ in range(54):
        _, done = env.step(4)
        assert done == False, "Simulation must not end before the whole sequence has been executed"
    # Last step then all should be done 
    _, done = env.step(0)
    assert done == True, "Simulation must have finished"
    env.render()


if __name__ == '__main__':
    test_environment()




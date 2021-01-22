from environment import ResourceManagementEnv, TerminationType
from job import Job
import test_parameters as pa
import numpy as np
import pytest
import copy

from logger import LogLevel, Logger

def generate_work_sequence():
    job_lengths = np.array([3, 8, 59, 20, 11, 1, 5, 6, 2, 7])
    job_resource_vectors = np.array([
        [10, 14],
        [11, 5],
        [25, 25],
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
    env = ResourceManagementEnv(params, logger=Logger(LogLevel.debug), to_render=True, termination_type=TerminationType.AllJobsDone)
    env.work_sequences = generate_work_sequence()
    env.reset()
    env.render()

    # execute the job from the last slot
    _, _,  done, allocation = env.step(0) # Schedule job with index 0 - [10, 4], 3
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert allocation == True, "Job must be successfully allocated"
    assert env.job_queue[0] is not None, "Slot must be filled back up from the backlog"
    assert env.job_queue[0] == env.work_sequences[0, 5], "The job at that slot must be %s" % (env.work_sequences[0, 5].to_string())
    work_sequence_idx = len(env.job_queue)

    # try to allocate job for which there is not enough resources
    old_slot_job = copy.copy(env.job_queue[2])
    _, _,  done, allocation = env.step(2) # Try to schedule job with index 2 [2, 5], 60, +1 timestep
    assert done == False, "Simulation must not end before the whole sequence has been executed" 
    assert allocation == False, "Job must not be successfully allocated" # Proceed 1 step further
    assert env.job_queue[2] == old_slot_job, \
        "Job on the slot has changed. Expected: %s, got: %s" % (env.job_queue[4].to_string(), old_slot_job.to_string())
    assert np.all(env.job_queue) is not None, "There are no empty slots in environment queue"
    work_sequence_idx += 1

    # Allocate first resource
    backlog_job = env.job_backlog.front()

    # Schedule job with index 0 in the queue, this is the job with index 5 in the simulation queue
    # [12, 12], 1. The finish time of this job will be 1 + 1 + 2 = 4
    _, _,  done, allocation = env.step(0) 
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert allocation == True, "Job must be successfully allocated" # Freeze
    # check if item from backlog is added to the queue at pos 0
    assert env.job_queue[0] == backlog_job, \
        "Job from backlog in queue. Expected: %s, got: %s" % (env.job_queue[0].to_string(), backlog_job.to_string())

    # Allocate resourse at position 1
    # Schedule jobs with index 1 in the queue, this is the job with index 1 in the simulation queue
    # [11, 5] 8, finish time - 9
    _, _,  done, allocation = env.step(1) 
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert allocation == True, "Job must be successfully allocated" # Freeze

    # Schedule jobs with index 3 in the queue, this is the job with index 3 in the simulation queue
    # [23, 4] 20, finish time - 29
    _, _,  done, allocation = env.step(3)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert allocation == True, "Job must be successfully allocated" # Freeze
    # slot should now be empty as no job is in backlog
    assert env.job_backlog.num_jobs == 1, \
        "The backlog should have a size of %d, but is %d" % (1, env.job_backlog.num_jobs)

    # Schedule jobs with index 4 in the queue, this is the job with index 4 in the simulation queue
    # [25, 25] 11, finish time - 40
    __, _,  done, allocation = env.step(4)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert allocation == True, "Job must be successfully allocated" # Freeze
    # slot should now be empty as no job is in backlog
    assert env.job_backlog.num_jobs == 0, \
        "The backlog should now be empty, but is %d" % (env.job_backlog.num_jobs)

    # Assert all jobs in the queue, after the backlog has been emptied
    # Here only 1 time step has passed, when the job was not allocated successfully
    assert env.current_time == 1
    assert env.job_queue[0] == env.work_sequences[0, 6]
    assert env.job_queue[1] == env.work_sequences[0, 7]
    assert env.job_queue[2] == env.work_sequences[0, 2]
    assert env.job_queue[3] == env.work_sequences[0, 8]
    assert env.job_queue[4] == env.work_sequences[0, 9]

    env.render()

    # Add next work to the empty slots in the queue
    # Schedule job with index 3 in the queue, this is the job with index 8 in the simulation queue
    # [13, 8] 2, finish time - 6, the first vector can be put together with the first vector of job with simulation id 1
    # which ends in time step 9. This job gets allocated after job with id 5 finishes at time step 4.
    _, _,  done, allocation = env.step(3)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert allocation == True, "Job must be successfully allocated" # Freeze
    assert env.job_queue[3] is None, \
        "Job from the queue at that index should now be executed and empty, becasue the backlog is empty"

    # Schedule job with index 0 in the queue, this is the job with index 6 in the simulation queue
    # [5, 9] 5, finish time - 45, the slots before 40 are full, but the previous job has resource vector
    # [2, 13], so the machine vector [25, 25] can accomodate both of these vectors at the same time
    _, _,  done, allocation = env.step(0)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert allocation == True, "Job must be successfully allocated" # Freeze
    assert env.job_queue[0] is None, \
        "Job from backlog in queue. Expected: %s, got: %s" % (env.job_queue[3].to_string(), env.work_sequences[0, 9].to_string())

    for _ in range(10):
        _, _,  done, _ = env.step(5) # wait, here 10 time steps will pass
        assert done == False, "Simulation must not end before the whole sequence has been executed"

    assert env.current_time == 11
    env.render()
    # 7 jobs are scheduled at the same time, only 3 left at the queue
    # The jobs at:
    #            index 1: [14, 6] - 6, sim index 7
    #            index 2: [2, 5] - 60, sim index 2
    #            index 4: [5, 5] - 7, sim index 9

    # 11 time steps have passed, so jobs with SIMULATION index 0, 1, 5, 8 should have already finished, assert this
    assert env.work_sequences[0, 0].finish_time == 3, "Job %s should have finished at time step 3, but finished at %d" % \
        (env.work_sequences[0, 0].to_string(), env.work_sequences[0, 0].finish_time)
    assert env.work_sequences[0, 1].finish_time == 9, "Job %s should have finished at time step 9, but finished at %d" % \
        (env.work_sequences[0, 1].to_string(), env.work_sequences[0, 1].finish_time)
    assert env.work_sequences[0, 5].finish_time == 4, "Job %s should have finished at time step 4, but finished at %d" % \
        (env.work_sequences[0, 5].to_string(), env.work_sequences[0, 5].finish_time)
    assert env.work_sequences[0, 3].finish_time == 29, "Job %s should have finished at time step 29, but finished at %d" % \
        (env.work_sequences[0, 3].to_string(), env.work_sequences[0, 3].finish_time)
    assert env.work_sequences[0, 4].finish_time == 40, "Job %s should have finished at time step 40, but finished at %d" % \
        (env.work_sequences[0, 4].to_string(), env.work_sequences[0, 4].finish_time)
    assert env.work_sequences[0, 8].finish_time == 6, "Job %s should have finished at time step 6, but finished at %d" % \
        (env.work_sequences[0, 8].to_string(), env.work_sequences[0, 8].finish_time)
    assert env.work_sequences[0, 6].finish_time == 45, "Job %s should have finished at time step 45, but finished at %d" % \
        (env.work_sequences[0, 6].to_string(), env.work_sequences[0, 6].finish_time)

    # Run 30 steps to ensure there is space for the last 3 tasks
    for _ in range(34): # 11 + 34 = 45 (max finish time)
        _, _,  done, allocation = env.step(5)
        assert done == False, "Simulation must not end before the whole sequence has been executed"
        assert allocation == False, "This is the void action, no allocation!"

    print(env.job_queue[1].to_string())
    print(env.job_queue[4].to_string())
    #Allocate last 3 tasks at index 1 2 3
    _, _,  done, allocation = env.step(1)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert allocation == True, "Job must be successfully allocated" # Freeze

    _, _,  done, allocation = env.step(4)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert allocation == True, "Job must be successfully allocated" # Freeze

    for _ in range(9): # 9 timesteps to finish both jobs
        _, _,  done, allocation = env.step(5)
        assert done == False, "Simulation must not end before the whole sequence has been executed"
        assert allocation == False, "This is the void action, no allocation!"

    _, _,  done, allocation = env.step(2)
    assert done == False, "Simulation must not end before the whole sequence has been executed"
    assert allocation == True, "Job must be successfully allocated" # Freeze

    for _ in range(58):
        _, _,  done, _ = env.step(5)
        assert done == False, "Simulation must not end before the whole sequence has been executed"
    # Last step then all should be done 
    _, _,  done, _ = env.step(0)
    assert done == True, "Simulation must have finished"
    env.render()


if __name__ == '__main__':
    test_environment()




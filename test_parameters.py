
# fixed parameters for environment testing
class TestParameters:
    def __init__(self):
        self.num_epochs = 10000
        self.number_resources = 2
        self.max_resource_slots = 25    # max number of available resource slots per resource
        self.episode_max_length = 200   # maximum number of time steps in an episode
        # TODO: is max_resource_slots always equal to r?
        self.t = 4
        self.r = 50                     # maximum resource request for new work
        self.jobs_sequence_length = 10
        self.simulation_length = 1      # number of job sequences
        self.job_rate = 0.75
        self.time_horizon = 15 * self.t # number of observed time steps
        self.work_queue_size = 5        # maximum number of waiting jobs in the queue
        self.backlog_size = 60          # size of backlog queue

        self.delay_penalty = -1         # penalty for holding things in the current work screen
        self.hold_penalty = -1          # penalty for holding jobs in the queue
        self.dismiss_penalty = - 1      # penalty for missing a job b/c of full queue
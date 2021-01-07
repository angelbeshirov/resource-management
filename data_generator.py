import numpy as np


class DataGenerator:

    # Initializes the parameters of the data generator
    def __init__(self, parameters):
        self.number_resources = parameters.number_resources
        self.t = parameters.t
        self.r = parameters.r

        self.small_jobs_probability = 0.8
        self.big_jobs_probability = 1 - self.small_jobs_probability

        self.big_jobs_lower_time_limit = 10 * self.t 
        self.big_jobs_upper_time_limit = 15 * self.t 

        self.small_jobs_lower_time_limit = 1 * self.t 
        self.small_jobs_upper_time_limit = 3 * self.t 

        self.dominant_resource_lower_limit = 0.25 * self.r
        self.dominant_resource_upper_limit = 0.5 * self.r

        self.other_resource_lower_limit = 0.05 * self.r
        self.other_resource_upper_limit = 0.1 * self.r

        self.jobs_sequence_length = parameters.jobs_sequence_length
        self.job_rate = parameters.job_rate

    # Generates a single job
    def generate_job(self):
        # generate time requirement of the job
        if np.random.rand() < self.small_jobs_probability:  # small job
            duration = np.random.randint(self.small_jobs_lower_time_limit, self.small_jobs_upper_time_limit + 1)
        else:  # big job
            duration = np.random.randint(self.big_jobs_lower_time_limit, self.big_jobs_upper_time_limit + 1)

        resource_requirements = np.zeros(self.number_resources)

        # generate resource requirements of the job
        dominant_resource = np.random.randint(0, self.number_resources)
        for i in range(self.number_resources):
            if i == dominant_resource:
                resource_requirements[i] = np.random.randint(self.dominant_resource_lower_limit,
                                                                self.dominant_resource_upper_limit + 1)
            else:
                resource_requirements[i] = np.random.randint(self.other_resource_lower_limit, 
                                                                self.other_resource_upper_limit + 1)

        return duration, resource_requirements

    # A sequence of jobs is generated according to a Bernoulli process
    def generate_sequence(self, debug = False):
        if debug == True: 
            np.random.seed(18) # Setting the seed = same random numbers every time, can be useful for debugging
        else:
            np.random.seed() # This will generate random numbers every time it is run

        durations = np.zeros(self.jobs_sequence_length, dtype=int)
        resources_requirements = np.zeros((self.jobs_sequence_length, self.number_resources), dtype=int)

        for k in range(self.jobs_sequence_length):
            if np.random.rand() < self.job_rate:  # a new job comes for scheduling
                durations[k], resources_requirements[k, :] = self.generate_job()


        # Some of the elements will have a duration of 0 and resource requirements of 0, 
        # because of the Bernoulli process
        return durations, resources_requirements


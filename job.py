import numpy as np
from logger import Logger, LogLevel

class Job:
    """
    Class representing the a job in the system.
    """
    def __init__(self, resource_vector, length, id, logger = Logger(LogLevel['info'])):
        self.resource_vector = resource_vector
        self.length = length
        self.id = id
        self.enter_time = -1
        self.start_time = -1
        self.finish_time = -1
        self.logger = logger

    def set_start_time(self, start_time):
        """
        Sets the start time of the job, i.e. the time at which
        the job is successfulyl allocated for execution.
        """
        self.start_time = start_time
        self.finish_time = self.start_time + self.length
        self.logger.debug("Start time %d, Finish time %d" % (self.start_time, self.finish_time))

    def set_enter_time(self, time):
        """
        Sets the enter time, i.e. the time at which
        the job arrives for scheduling.
        """
        self.enter_time = time
    
    def __eq__(self, other):
        """
        Compares 2 jobs for equality.
        """
        return self.length == other.length and \
            (self.resource_vector == other.resource_vector).all()

    def __str__(self):
        return self.to_string()

    def to_string(self):
        """
        Converts a job to a string representation.
        """
        return "Jod (id: %d, length: %d, resource_vector: %s)" \
            % (self.id, self.length, np.array2string(self.resource_vector))

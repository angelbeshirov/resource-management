import numpy as np

class Job:
    def __init__(self, resource_vector, length, id):
        self.resource_vector = resource_vector
        self.length = length
        self.id = id
        self.enter_time = -1
        self.start_time = -1
        self.finish_time = -1

    def set_start_time(self, start_time):
        self.start_time = start_time
        self.finish_time = self.start_time + self.length
        print(self.finish_time)

    def set_enter_time(self, time):
        self.enter_time = time
    
    def __eq__(self, other):
        return self.length == other.length and \
            (self.resource_vector == other.resource_vector).all()

    def to_string(self):
        return "Jod (id: %d, length: %d, resource_vector: %s)" \
            % (self.id, self.length, np.array2string(self.resource_vector))

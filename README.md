# Resource Management with Deep Reinforcement Learning
## Description
Solving resource management problems is essential in most computer systems and networks nowadays. Most of these problems are solved by using specifically designed heuristics based on the workload and environment of the system. We use Deep RL to create a system which learns to manage its resources from experience. Our results show that the system can outperform some of the standard (heuristic) methods, can adapt to different environment workload and be optimized for a specific system objective.
This repository is an implementation of a resource management system using deep reinforcement learning, as described in [here](https://people.csail.mit.edu/alizadeh/papers/deeprm-hotnets16.pdf).

## Train
To start the training with the parameters specified in [parameters.py](./parameters.py) you have to run the [train.sh](./train.sh) script. The current reward used in the system is for optimizing the average slowdown of the system, which is the difference between start and completion time, normalized by the length of that job. If you want to optimize for other objectives consider changing the reward in the [environment.py](./environment.py).

## Test
To run the tests with some model run [test.sh](./test.sh) script and pass as parameter the model which you want to test. If you don't pass a model the default one will be used which is [this one](./models/best_slowdown_model.pkl). The script will evaluate the Deep RL model, the Shortest-Job-First and the Packer algorithm against the saved data in the [test.data](./test.data) file.

## Future improvements
* Add job fragmenetaion so that the cluster allows scheduling of jobs with resource requirements larger that the machine resource profile.
* Add job preemption - that is jobs should be able to be stopped and then resumed again from the point where they were stopped forward by the system
* Replace the time-dependant baseline with a value network, which estimates the average return value
* Think about how to model inter-job dependencies (shared memory, CPU cache etc.)

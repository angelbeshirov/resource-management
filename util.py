import numpy as np
from parameters import Parameters
from data_generator import DataGenerator
from job import Job

import matplotlib.pyplot as plt
import pickle

def generate_sequence_and_save(parameters):
    data_generator = DataGenerator(parameters)
    job_lengths, job_resource_vectors = data_generator.generate_sequence()

    file = open('test.data', 'wb')

    pickle.dump({"job_lengths": job_lengths,
                 "job_resource_vectors": job_resource_vectors}, file, -1)
    file.close()

    print("Saved")

def retrieve_test_data():
    file = open('test.data', 'rb')
    data = pickle.load(file)
    counter = 1
    job_lengths, job_resource_vectors = data["job_lengths"], data["job_resource_vectors"]
    job_sequence_length = len(job_lengths)
    work_sequences = np.full((1, job_sequence_length), None, dtype=object)
    for j in range(job_sequence_length):
        # check if valid Job
        if job_lengths[j] > 0:
            work_sequences[0, j] = Job(job_resource_vectors[j], \
                job_lengths[j],                                 \
                id=counter)
            counter = counter + 1
    return work_sequences

def plot(axs, episode, total_rewards, losses, avg_slowdowns, avg_completion_times, \
    min_reward, mean_reward, max_reward, std_reward):
    episode_seq = np.arange(episode+1)
    axs[0,0].clear()
    axs[0,0].plot(episode_seq, total_rewards[:episode+1], '-o')
    axs[0,0].set_title('Total reward')
    axs[0,0].set(xlabel='Episode', ylabel='Total reward')

    axs[0,1].plot(episode_seq, losses[:episode+1], '-o', color='green')
    axs[0,1].set_title('Loss')
    axs[0,1].set(xlabel='Episode', ylabel='Pseudo loss')

    axs[1,0].plot(episode_seq, avg_slowdowns[:episode+1], '-o', color='orange')
    axs[1,0].set_title('Average slowdown')
    axs[1,0].set(xlabel='Episode', ylabel='Average slowdown')

    axs[1,1].plot(episode_seq, avg_completion_times[:episode+1], '-o', color='red')
    axs[1,1].set_title('Average completion time')
    axs[1,1].set(xlabel='Episode', ylabel='Average completion time')

    axs[0,2].plot(episode_seq, mean_reward[:episode+1], '-k', label='mean reward' )
    axs[0,2].fill_between(episode_seq, 
                    mean_reward[:episode+1]-0.5*std_reward[:episode+1], 
                    mean_reward[:episode+1]+0.5*std_reward[:episode+1], 
                    color='k', 
                    alpha=0.25)

    axs[0,2].plot(episode_seq, min_reward[:episode+1], '--b' , label='min reward' )
    axs[0,2].plot(episode_seq, max_reward[:episode+1], '--r' , label='max reward' )
    axs[0,2].set_title('Min/Mean/Max Reward')
    axs[0,2].set(xlabel='Episode', ylabel='Min/Mean/Max Reward')

    plt.pause(0.05)

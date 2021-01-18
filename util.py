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

def plot_total_rewards(axs, episode_seq, total_rewards, best_reward):
    episode = episode_seq[-1]
    axs.plot(episode_seq, total_rewards[:episode+1], '-o')
    axs.set_title('Total reward')
    axs.axhline(best_reward, linestyle='--')
    axs.set(xlabel='Episode', ylabel='Total reward')

def plot_losses(axs, episode_seq, losses):
    episode = episode_seq[-1]
    axs.plot(episode_seq, losses[:episode+1], '-o', color='green')
    axs.set_title('Loss')
    axs.axhline(0, color='red')
    axs.set(xlabel='Episode', ylabel='Pseudo loss')

def plot_avg_slowdowns(axs, episode_seq, avg_slowdowns):
    axs.clear()
    episode = episode_seq[-1]
    axs.plot(episode_seq, avg_slowdowns[:episode+1], '-o', color='orange')
    axs.set_title('Average slowdown')
    axs.set(xlabel='Episode', ylabel='Average slowdown')

def plot_avg_completion_time(axs, episode_seq, avg_completion_times):
    axs.clear()
    episode = episode_seq[-1]
    axs.plot(episode_seq, avg_completion_times[:episode+1], '-o', color='red')
    axs.set_title('Average completion time')
    axs.set(xlabel='Episode', ylabel='Average completion time')

def plot_min_max_rewards(axs, episode_seq, min_reward, max_reward, mean_reward, std_reward):
    episode = episode_seq[-1]
    axs.plot(episode_seq, mean_reward[:episode+1], '-k', label='mean reward')
    axs.fill_between(episode_seq, 
                    mean_reward[:episode+1]-0.5*std_reward[:episode+1], 
                    mean_reward[:episode+1]+0.5*std_reward[:episode+1], 
                    color='k', 
                    alpha=0.25)

    axs.plot(episode_seq, min_reward[:episode+1], '--b' , label='min reward')
    axs.plot(episode_seq, max_reward[:episode+1], '--r' , label='max reward')
    axs.set_title('Min/Mean/Max Reward')
    axs.set(xlabel='Episode', ylabel='Min/Mean/Max Reward')

def plot_system_load(axs, episode_seq, avg_system_loads):
    episode = episode_seq[-1]
    axs.plot(episode_seq, avg_system_loads[:episode+1] * 100, '-o', color='red')
    axs.set_title('Average system load in %')
    axs.set(xlabel='Episode', ylabel='Average system load')

def print_job_sequence(logger, job_sequence):
    for job in job_sequence:
        if job is not None:
            logger.info(job.to_string())
        else: logger.info("None")

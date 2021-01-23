import numpy as np
import argparse
import matplotlib.pyplot as plt
import util

from environment import ResourceManagementEnv, TerminationType
from parameters import Parameters
from data_generator import DataGenerator
from neural_network import Neural_network
from job import Job
from logger import LogLevel, Logger
from trainer import Trainer
from evaluator import Evaluator
from specific_agents import PackerAgent, SJFAgent

import copy

def main():
    np.set_printoptions(precision=5)
    parameters = Parameters()

    # parameters used for starting this class from shell scripts and executing different flows with different paremeters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--train", action="store_true", help="If true the training starts.")
    parser.add_argument("-l", "--loglevel", type=str, default="info", choices=['debug', 'info'], help="Log level to be used.")
    parser.add_argument("-e", "--evaluate", action="store_true", help="If evaluate is true, all agents will be evaluated based on the system objectives.")
    parser.add_argument("-m", "--model", type=str, default="models/best_slowdown_model.pkl", help="Model which you want to evaluate.")
    args = parser.parse_args()
    logger = Logger(LogLevel[args.loglevel])

    if args.train:
        trainer = Trainer(parameters, logger)
        trainer.train()
    elif args.evaluate:
        #util.generate_sequence_and_save(parameters) # Generates a new random sequence and saves it
        test_sequence = util.retrieve_test_data()    # Retrieves the saved sequence for evaluation (save for results reproducity)
        env = ResourceManagementEnv(parameters, logger, to_render=False, termination_type=TerminationType.AllJobsDone)

        env.work_sequences = copy.deepcopy(test_sequence)
        env.simulation_length = test_sequence.shape[0]
        env.job_sequence_length = test_sequence.shape[1]
        env.seq_number = 0

        # Run the actual evaluation
        nn = Neural_network(parameters, env, logger)
        nn.load(args.model)

        packer = PackerAgent(parameters, env, logger)
        sjf = SJFAgent(parameters, env, logger)

        evaluator = Evaluator(parameters, env, logger)

        # evaluate the RL agent
        dnnsl = evaluator.evaluate_dnn(nn, deterministic=True)

        # evaluate packer
        env.work_sequences = copy.deepcopy(test_sequence)
        packersl = evaluator.evaluate_packer(packer)

        # evaluate SJF
        env.work_sequences = copy.deepcopy(test_sequence)
        sjfsl = evaluator.evaluate_sjf(sjf)

        # create barplot of the results.
        util.create_slowdown_bar([dnnsl, packersl, sjfsl])

if __name__ == '__main__':
    main()
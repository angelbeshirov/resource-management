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
from specific_agents import PackerAgent

def main():
    np.set_printoptions(precision=5)
    parameters = Parameters()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--train", action="store_true", help="If true the training starts.")
    parser.add_argument("-l", "--loglevel", type=str, default="info", choices=['debug', 'info'], help="Log level to be used.")
    parser.add_argument("-ea", "--evaluateall", action="store_true", help="If evaluateall is true, all agents will be evaluated based on the system objectives.")
    args = parser.parse_args()

    # if args.train:
        # logger = Logger(LogLevel[args.loglevel])
        # env = ResourceManagementEnv(parameters, logger, to_render=False, termination_type=TerminationType.AllJobsDone)
        # neural_net = Neural_network(parameters, env, logger)
        # neural_net.train()
    logger = Logger(LogLevel[args.loglevel])
    if args.train:
        trainer = Trainer(parameters, logger) # does the multiseq training
        trainer.train_test()
    elif args.evaluateall:
        #util.generate_seq_and_save(parameters)
        test_sequence = util.retrieve_test_data()
        env = ResourceManagementEnv(parameters, logger, to_render=False, termination_type=TerminationType.AllJobsDone)

        # Set env parameters
        env.work_sequences = test_sequence
        env.simulation_length = 1
        env.job_sequence_length = test_sequence.shape[1]

        # Run the evaluation
        nn = Neural_network(parameters, env, logger)
        nn.load('./best_model.pkl')

        packer = PackerAgent(parameters, env)

        evaluator = Evaluator(parameters, env, logger)

        evaluator.evaluate_dnn(nn, deterministic=True)
        evaluator.evaluate_packer(packer)

if __name__ == '__main__':
    main()
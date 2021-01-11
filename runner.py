import numpy as np
import argparse
import matplotlib.pyplot as plt

from environment import ResourceManagementEnv
from parameters import Parameters
from data_generator import DataGenerator
from neural_network import Neural_network
from job import Job
from logger import LogLevel, Logger



def main():
    np.set_printoptions(precision=5)
    parameters = Parameters()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--train", action="store_true", help="If true the training starts.")
    parser.add_argument("-l", "--loglevel", type=str, default="info", choices=['debug', 'info'], help="Log level to be used.")
    args = parser.parse_args()

    if args.train:
        logger = Logger(LogLevel[args.loglevel])
        env = ResourceManagementEnv(parameters, logger)
        neural_net = Neural_network(parameters, env, logger)
        neural_net.train()

if __name__ == '__main__':
    main()
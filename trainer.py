import numpy as np
from neural_network import Neural_network
from parameters import Parameters
from data_generator import DataGenerator
from environment import ResourceManagementEnv

import jax.numpy as jnp

def main():
    parameters = Parameters()
    env = ResourceManagementEnv(parameters)
    neural_net = Neural_network(parameters, env)
    env.step(0)
    state1 = env.retrieve_state()
    env.step(1)
    state2 = env.retrieve_state()
    env.step(0)
    state3 = env.retrieve_state()
    env.step(1)
    state4 = env.retrieve_state()

    #data = env.generate_work_sequences()

    # print(np.array2string(env.observe()))
    # data.reshape([1, 60, -1])
    data = np.array([state1, state2, state3, state4])
    # print(neural_net.predict(data.reshape([1, -1])))
    output = neural_net.predict(data.reshape([4, -1]))
    print(output)

    print('conservation of probability', np.sum(jnp.exp(output), axis=1))
    print("Actions to take are {}".format(jnp.argmax(output, axis=1)))

    neural_net.train()


if __name__ == '__main__':
    main()
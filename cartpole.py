# Standard libraries
import random
import numpy as np

# OpenAI gym for the CartPole environment
import gym

# tflearn for our multi-layered perceptron (MLP) model
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

class Cartpole_Agent():
    def __init__(self, game):
        self.game = game
        self.env = gym.make(game)
        self.goal_steps = 500

    def initial_population(self, score_requirement = 50, initial_games = 10000):
        '''
        Creates training data for our CartPole
        prediction model
        '''
        env = self.env
        env.reset()
        # training_data format: [OBS, MOVES]
        training_data = []
        scores = []
        accepted_scores = []

        # Perform random games and save training data
        for _ in range(initial_games):
            score = 0
            game_memory = []
            prev_observation = []
            for _ in range(self.goal_steps):
                # Perform random actions (left or right)
                action = random.randrange(0,2)
                observation, reward, done, info = env.step(action)
                if len(prev_observation) > 0 :
                    game_memory.append([prev_observation, action])
                prev_observation = observation
                score+=reward
                if done: break

            # Save only successful data
            if score >= score_requirement:
                accepted_scores.append(score)
                for data in game_memory:
                    # convert to one-hot encoding
                    if data[1] == 1:
                        output = [0,1]
                    elif data[1] == 0:
                        output = [1,0]

                    # save training data
                    training_data.append([data[0], output])

            # save overall scores
            scores.append(score)

            env.reset()

        # Save training data as a NumPy array
        training_data_save = np.array(training_data)
        np.save('population.npy',training_data_save)

        return training_data

    def load_data(self, path):
        '''
        Loads the training data for training the model
        '''
        return np.load(path)

    @staticmethod
    def neural_network_model(input_size, lr):
        '''
        A simple multi-layered perceptron model
        '''
        network = input_data(shape=[None, input_size, 1], name='input')

        network = fully_connected(network, 128, activation='relu')
        network = dropout(network, 0.8)

        network = fully_connected(network, 256, activation='relu')
        network = dropout(network, 0.8)

        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.8)

        network = fully_connected(network, 256, activation='relu')
        network = dropout(network, 0.8)

        network = fully_connected(network, 128, activation='relu')
        network = dropout(network, 0.8)

        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')
        dnn = tflearn.DNN(network, tensorboard_dir='log')

        return dnn

    def train_model(self, training_data, epochs = 5, show_metric = True, lr = 1e-3):
        '''
        Trains a multi-layered perceptron model
        for predicting the next move in CartPole
        '''
        # Pre-process training_data
        X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
        y = [i[1] for i in training_data]

        # Train DNN model
        model = self.neural_network_model(input_size = len(X[0]), lr=lr)

        model.fit({'input': X},
                  {'targets': y},
                  n_epoch=epochs,
                  snapshot_step=500,
                  show_metric=show_metric,
                  run_id='openai_learning')

        return model

    def evaluate(self, model, iterations=5):
        '''
        Renders the cartpole environment and calculates average score
        '''
        env = self.env
        scores = []
        choices = []
        iter = 1
        for game in range(iterations):
            score = 0
            game_memory = []
            prev_obs = []
            env.reset()
            # Render games based on our trained model
            for _ in range(self.goal_steps):
                env.render()
                if len(prev_obs) == 0:
                    action = random.randrange(0, 2)
                else:
                    action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

                choices.append(action)
                new_observation, reward, done, info = env.step(action)
                prev_obs = new_observation
                game_memory.append([new_observation, action])
                score += reward
                if done: break

            # Document scores
            print(f'Iteration: {iter}\nScore: {score}\n')
            iter += 1
            scores.append(score)
        env.close()
        print(f"Average Score: {int(sum(scores) / iterations)}")

# Train and evaluate Cartpole Agent
if __name__ == "__main__":
    # Create agent
    agent = Cartpole_Agent('CartPole-v1')

    # In case you have no training data yet
    training_data = agent.initial_population()

    # In case you already collect training data
    #training_data = agent.load_data('data/population.npy')

    model = agent.train_model(training_data)
    agent.evaluate(iterations=3, model=model)
import gymnasium as gym
import numpy as np
import evogym.envs
import os
import cv2
from stable_baselines3 import PPO
from evogym.utils import draw, get_uniform, get_full_connectivity, is_connected, has_actuator

def make_video(images, distances, video_dir, video_name):
    '''
    Saves a list of images as a mp4 video
    '''
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    height, width, _ = images[0].shape
    size = (width, height)
    out = cv2.VideoWriter(f'{video_dir}/{video_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 50, size)

    #Add text to video on top left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCorner = (20, 50)
    fontScale = 2
    fontColor = (0, 0, 0)
    lineType = 2

    for i in range(len(images)):
        #Print distance on top left corner, rounded to 2 decimal places
        cv2.putText(images[i], f'Distance {distances[i]:.2f}', topLeftCorner, font, fontScale, fontColor, lineType)
        out.write(images[i])

    out.release()


def random_from_shape(shape_file = None):
    '''
    Returns a randomly sampled robot of a particular shape
    '''
    if shape_file is None:
        '''Default is 8x8 letter A'''
        robot_shape = np.array([
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
        ])
    else:
        robot_shape = np.load(shape_file)

    done = False
    pd = get_uniform(4)
    pd = np.insert(pd, 0, 0.0)


    while not done:
        robot = np.zeros(robot_shape.shape)
        for i in range(robot_shape.shape[0]):
            for j in range(robot_shape.shape[1]):
                if robot_shape[i][j] == 1:
                    robot[i][j] = draw(pd)

        if is_connected(robot) and has_actuator(robot):
            done = True

    return robot, get_full_connectivity(robot)

class Organism():
    def __init__(self, body, connections):
        self.body = body
        self.connections = connections
        self.fitness = 0

        self.train_env = gym.make('Walker-v0', body=self.body, max_episode_steps=100)
        self.eval_env = gym.make('Walker-v0', body=self.body, max_episode_steps=500)

        self.model = PPO('MlpPolicy', self.train_env, verbose=0)
        self.trained = False

    def train(self, training_steps=50000, verbose=True):
        '''
        Trains ppo agent on the environment
        '''
        if not self.trained:
            self.model.learn(total_timesteps=training_steps)
            self.trained = True
            if verbose:
                print(f"Training complete after {training_steps} steps.")
        else:
            if verbose:
                print("Model already trained.")

    def evaluate(self, episodes=10, steps=500, verbose=0):
        '''
        Evaluates the organism by running several episodes and calculating average fitness.
        '''
        total_reward = 0

        for episode in range(episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            for i in range(steps):
                action, _ = self.model.predict(obs)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward

            total_reward += episode_reward
            if verbose >= 2:
                print(f"Episode {episode + 1} reward: {episode_reward}")

        self.fitness = total_reward / episodes
        if verbose >= 1:
            print(f"Average fitness over {episodes} episodes: {self.fitness}")


    def create_offspring(self, other, mutation_rate=0.1, partner_prob=0.5):
        '''
        Creates an offspring from two organisms by combining their bodies and connections
        '''
        pd = get_uniform(4)
        pd = np.insert(pd, 0, 0.0)
        new_body = self.body.copy()
        new_connections = self.connections.copy()

        # Combine the bodies, randomly selecting from either parent, and mutate
        for i in range(new_body.shape[0]):
            for j in range(new_body.shape[1]):
                if self.body[i][j] != 0:
                    if np.random.rand() < partner_prob:
                        new_body[i][j] = other.body[i][j]

                    if np.random.rand() < mutation_rate:
                        new_body[i][j] = draw(pd)


        return Organism(new_body, new_connections)
        

    def save_to_video(self, steps=500, video_dir='videos/', video_name='test'):
        images = []
        distances = []

        env = gym.make('Walker-v0', body=self.body, max_episode_steps=steps, render_mode='img')
        obs, _ = env.reset()

        for i in range(steps):
            action, _ = self.model.predict(obs)
            obs, _, _, _, _ = env.step(action)
            images.append(env.render())
            distances.append(np.mean(env.unwrapped.object_pos_at_time(env.unwrapped.get_time(), "robot")))

        env.close()
        make_video(images, distances, video_dir, video_name)

if __name__ == '__main__':

    body, connections = random_from_shape()
    organism = Organism(body, connections)
    organism.train()
    organism.evaluate()
    organism.save_to_video()
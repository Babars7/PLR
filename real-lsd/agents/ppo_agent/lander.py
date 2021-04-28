import os
import sys
import gym
import time
import torch
import torch.nn as nn
import pickle
import numpy as np
import glog as log
import subprocess
import tqdm

#import gym_unrealcv


import real_lsd

from PPOagent import PPOAgent

# TODO: change data save path

'''------------------------Hyperparameters------------------------'''
hidden_size      = 256
lr               = 3e-4
num_steps        = 20
mini_batch_size  = 5
ppo_epochs       = 4
max_frames       = 15000
threshold_reward = - 200

'''---------------------------------------------------------------'''

'''---------------------------Helper functions--------------------'''
def save_obj(obj, filename):
    dir = 'data'
    # path = os.getcwd()
    PATH = '/media/scratch1/nasib'

    if dir not in os.listdir(PATH):
        PATH =  os.path.join(PATH, dir)
        os.mkdir(PATH)
    else:
        PATH =  os.path.join(PATH, dir)

    abs_file_path = PATH + '/' + filename + '.pkl'
    with open(abs_file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    return filename

def load_obj(filename):
    dir = 'data'
    path = os.getcwd()
    assert dir in os.listdir(path)
    path = path + '/' + dir + '/'
    with open(path + filename + '.pkl', 'rb') as f:
        return pickle.load(f)

# get activation of layers in model
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# def plot(frame_idx, rewards):
#     clear_output(True)
#     plt.figure(figsize=(20,5))
#     plt.subplot(131)
#     plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
#     plt.plot(rewards)
#     plt.show()

def test_env():
    state = env.reset()
    done = False
    steps = 0
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = agent.model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        steps +=1
        total_reward += reward

    mean_reward = total_reward / steps
    return [total_reward, mean_reward]

'''---------------------------------------------------------------'''

# LOG LEVEL: Set to INFO for debugging
log.setLevel("INFO")   # WARN
log.warn

# Copy settings file to data folder
abs_path = os.path.dirname(real_lsd.__file__) + '/envs/settings/landing/cpptest.json'
cp_path  = '/media/scratch1/nasib/data/' # DATAPATH TODO
list_files = subprocess.run(["cp", abs_path, cp_path])
log.warn("The exit code was: %d" % list_files.returncode)


# Check cuda availability/set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# Initialising environment
env = gym.make('MyUnrealLand-cpptestFloorGood-DiscreteHeightFeatures-v0')
num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.n
log.info("Observation Space:", env.observation_space, "dimension of observation:", env.observation_space.shape[0])
log.info("Action Space:", env.action_space, "Number of actions:", env.action_space.n)


# Initialise agent
agent = PPOAgent(num_inputs,
                num_outputs,
                hidden_size,
                lr,
                num_steps,
                mini_batch_size,
                ppo_epochs,
                threshold_reward,
                std=0.0,
                device=device)

# print(agent.model)
# print(agent.model.actor)
# print(agent.model.actor[0])
# print(type(agent.model.actor[0]))


# set model train mode
agent.model.train()


# Register hooks for actor layer activations
for name, layer in agent.model.actor.named_modules():
    if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Linear) or isinstance(layer, nn.LogSoftmax) or isinstance(layer, nn.LeakyReLU):
        print(name, layer)
        layer.register_forward_hook(get_activation('actor_layer_{}'.format(name)))


# Define and initialise auxiliary variables
frame_idx  = 0
test_rewards = []
test_mean_rewards = []
episode_count = 0
episode_now = 0
values_at_beginning = []
prior_action = 0
activation = {}
training_data = dict()
early_stop = False


# Reset environment and load starting state into state
state = env.reset()

# Training loop
log.warn("Start training")
log.setLevel("ERROR")
pbar = tqdm.tqdm(total=max_frames)

while frame_idx < max_frames and not early_stop:
    log.warn("Frame: {}".format(frame_idx))
    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    entropy = 0

    for st in range(num_steps):
        #print("Frame ", frame_idx, "/", max_frames, "\r", end='', flush=True) # added
        pbar.update(1)
        action = None
        state = torch.FloatTensor(state).to(device)
        log.info("Model Input: {}".format(state))

        assert torch.sum(torch.isnan(state)) == 0, "State contains NANs!"
        log.info("state SIZE: {}".format(state.size()))

        dist, value = agent.model(state)
        log.info("Forward pass distribution: {}, forward pass value: {}".format(dist, value))

        if (episode_now != episode_count):
            values_at_beginning.append(value)
        episode_now = episode_count

        # Output all layer activations to logstream
        for i in range(5):
            log.warn("actor layer {} activation: {}".format(i, activation['actor_layer_'+str(i)]))

        state = state.unsqueeze(1)
        state = torch.transpose(state, 0, 1)
        log.info("state post unsqueeze and transpose SIZE: {}".format(state.size()))

        value = value.unsqueeze(1)
        log.info("Value: {}".format(value))

        action = dist.sample()
        log.warn("Sampled Action: {}".format(action))
        log.info("Action TYPE: {}, SHAPE: {}".format(type(action), action.shape))

        action_not_same = not (action == prior_action)
        log.warn("Sampled Action is not same as prior action: {}".format(action_not_same))
        prior_action = action

        next_state, reward, done, _ = env.step(action.cpu().numpy())
        log.warn("Step REWARD: {} DONE: {}".format(reward, done))

        log_prob = dist.log_prob(action)
        log.info("Step LOG_PROB: {}".format(log_prob))
        log.info("LOG_PROB TYPE: {}, SHAPE: {}".format(type(log_prob), log_prob.shape))

        entropy += dist.entropy().mean()

        # Append data to arrays
        log_prob = log_prob.unsqueeze(0)
        log_prob = log_prob.unsqueeze(1)
        log_prob = log_prob.to(device)
        log_probs.append(log_prob)

        values.append(value)

        interim = torch.FloatTensor([np.float(reward)])
        interim = interim.unsqueeze(1)
        reward  = interim.to(device)
        rewards.append(reward)

        mask = float(1-done)
        mask = torch.FloatTensor([mask])
        mask = mask.unsqueeze(1)
        masks.append(mask.to(device))

        states.append(state)

        action = action.unsqueeze(0)
        action = action.unsqueeze(1)
        action = action.to(device)
        actions.append(action)

        # collect data on training progress
        if frame_idx % 2048 == 0:
            test_ep_total_rewards = []
            test_ep_mean_rewards = []
            for _ in range(10):
                test_res = test_env()
                test_ep_total_rewards.append(test_res[0])
                test_ep_mean_rewards.append(test_res[1])

            test_mean_reward = np.mean(test_ep_mean_rewards)
            test_reward = np.mean(test_ep_total_rewards)
            log.warn("Test results -> Total reward: {}, Mean reward: {}".format(test_reward, test_mean_reward))

            test_mean_rewards.append(test_mean_reward)
            test_rewards.append(test_reward)
            # plot(frame_idx, test_rewards)
            # if test_reward > threshold_reward: early_stop = True

        # next state logic
        if done:
            episode_count += 1
            log.warn("Resetting.")
            state = env.reset()
        else:
            state = next_state

        frame_idx += 1

        ### END for loop

    next_state = torch.FloatTensor(next_state).to(device)

    assert torch.sum(torch.isnan(next_state)) == 0

    _, next_value = agent.model(next_state)

    next_state = next_state.unsqueeze(1)
    next_state = torch.transpose(next_state, 0, 1)

    # compute returns for advantage function
    returns = agent.compute_gae(next_value,
                                rewards,
                                masks,
                                values)

    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    advantage = returns - values

    # log.info("Returns before CAT: {}".format(len(returns)))
    log.info("Returns   SIZE after CAT: {}".format(returns.size()))
    log.info("log probs SIZE after CAT: {}".format(log_probs.size()))
    log.info("Values    SIZE after CAT: {}".format(values.size()))
    log.info("States    SIZE after CAT: {}".format(states.size()))
    log.info("Actions   SIZE after CAT: {}".format(actions.size()))
    log.info("Advantage SIZE after CAT: {}".format(advantage.size()))

    log.warn("Calling PPO update.")
    agent.ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

    ### END while loop

pbar.close()

# Save data collected during training
training_data['test_rewards'] = test_rewards
training_data['test_mean_rewards'] = test_mean_rewards
training_data['values_at_beginning'] = values_at_beginning
_ = save_obj(training_data, 'training_data')

# Delete training data once data is saved! Free up Memory
del training_data
del test_rewards
del test_mean_rewards
del values_at_beginning

#log.setLevel("WARN") # added

# Save model
log.warn("Saving the model.")
agent.save_model()

log.warn("Training completed.")
print("Training completed")

'''------------------- Testing the policy after training --------------------'''
print("Start testing")
# Testing parameters
num_tests = 1
episodes_per_test = 50
successful_episodes = 0

log.warn("Setting model to eval, setup for testing.")
agent.model.eval()
pbar2 = tqdm.tqdm(total=episodes_per_test * num_tests)

with torch.no_grad():
    for test in range(num_tests):
        episode_count = 0
        num_test_episodes = episodes_per_test

        episodes = {}

        state = env.reset()

        while episode_count < num_test_episodes:

            pbar2.update(1)
            done = False
            episode = {}

            # poses     = [start_pose]
            states    = [state]
            dists     = []
            values    = []
            actions   = []
            rewards   = []
            log_probs = []
            traj      = []

            while not done:
                action = 0
                states.append(state)

                state = torch.FloatTensor(state).to(device)
                dist, value = agent.model(state)

                action = dist.sample()
                log.info("action type: {}".format(action))
                log_prob = dist.log_prob(action)

                next_state, reward, done, info = env.step(action.cpu().numpy())

                # poses.append(info['Pose'])
                dists.append(dist)
                values.append(value.cpu().numpy())
                actions.append(action.cpu().numpy())
                log_probs.append(log_prob.cpu().numpy())
                rewards.append(reward)

                # next state logic
                if done:
                    if info['Success']:
                        successful_episodes += 1
                    traj = info['Trajectory']
                    state = env.reset()
                    episode_count += 1
                else:
                    state = next_state

            # episode['poses']    = poses
            episode['states']   = states
            episode['dists']    = dists
            episode['values']   = values
            episode['actions']  = actions
            episode['rewards']  = rewards
            episode['log_probs'] = log_probs
            episode['trajectory']= traj

            key = 'episode_{}'.format(episode_count)
            episodes[key] = episode
        
        pbar2.close()
        filename = time.strftime("%Y%m%d_%H%M%S") + '{}'.format(test)
        log.warn("About to save the test data.")
        file = save_obj(episodes, filename)
        del episodes
        log.warn("Successes out of {}: {}".format(num_tests*episodes_per_test, successful_episodes))
        print("Successes out of {}: {}".format(num_tests*episodes_per_test, successful_episodes))
        # print(load_obj(file))

log.warn("Done Testing.")

env.close()

sys.exit('Training and testing completed.')

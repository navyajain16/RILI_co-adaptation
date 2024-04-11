import argparse
import datetime
import os.path
import gym
import gym_rili
import numpy as np
from algos.rili import RILI
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
import random

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="rili-circle-v0")
parser.add_argument('--resume', default="None")
parser.add_argument('--change_partner', type=float, default=0.99)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--save_name', default='run')
parser.add_argument('--start_eps', type=int, default=300)
parser.add_argument('--num_eps', type=int, default=2000)
parser.add_argument('--dir_name', default='experiment1')
args = parser.parse_args()


# Environment
env = gym.make(args.env_name)
env.set_params(change_partner=args.change_partner)

# Agent
agent1 = RILI(env.action_space, env.observation_space.shape[0], env._max_episode_steps)
agent2 = RILI(env.action_space, env.observation_space.shape[0], env._max_episode_steps)
agent3 = RILI(env.action_space, env.observation_space.shape[0], env._max_episode_steps)

# Tensorboard
folder = "runs/" + 'rili' + "/"
writer = SummaryWriter(folder + '{}_{}'.format(args.save_name, datetime.datetime.now().strftime("%m-%d_%H-%M")))

# Memory
memory_agent1 = ReplayMemory(capacity=args.num_eps, interaction_length=env._max_episode_steps)
memory_agent2 = ReplayMemory(capacity=args.num_eps, interaction_length=env._max_episode_steps)
memory_agent3 = ReplayMemory(capacity=args.num_eps, interaction_length=env._max_episode_steps)

# Resume Training
if args.resume != "None":
    agent.load_model(args.resume)
    memory.load_buffer(args.resume)
    args.start_eps = 0

z_prev_agent1 = np.zeros(10)
z_agent1 = np.zeros(10)


z_prev_agent2 = np.zeros(10)
z_agent2 = np.zeros(10)

z_prev_agent3 = np.zeros(10)
z_agent3 = np.zeros(10)

# Main loop
reward1_chk = 0
reward2_chk = 0
reward3_chk = 0
models_list = []

for i_episode in range(1, args.num_eps+1):

    if len(memory_agent1) > 4:
        z_agent1 = agent1.predict_latent(
                        memory_agent1.get_steps(memory_agent1.position - 4),
                        memory_agent1.get_steps(memory_agent1.position - 3),
                        memory_agent1.get_steps(memory_agent1.position - 2),
                        memory_agent1.get_steps(memory_agent1.position - 1))

        z_agent2 = agent2.predict_latent(
                        memory_agent2.get_steps(memory_agent2.position - 4),
                        memory_agent2.get_steps(memory_agent2.position - 3),
                        memory_agent2.get_steps(memory_agent2.position - 2),
                        memory_agent2.get_steps(memory_agent2.position - 1))
                    
        z_agent3 = agent3.predict_latent(
                        memory_agent3.get_steps(memory_agent3.position - 4),
                        memory_agent3.get_steps(memory_agent3.position - 3),
                        memory_agent3.get_steps(memory_agent3.position - 2),
                        memory_agent3.get_steps(memory_agent3.position - 1))


    episode_reward1 = 0
    episode_reward2 = 0
    episode_reward3 = 0 
    episode_steps = 0
    done = False
    state1, state2, state3 = env.reset()
    

    while not done:

        if i_episode < args.start_eps:
            action_agent1 = env.action_space.sample()
            action_agent2 = env.action_space.sample()
            action_agent3 = env.action_space.sample()
        else:
            action_agent1 = agent1.select_action(state1, z_agent1)
            action_agent2 = agent2.select_action(state2, z_agent2)
            action_agent3 = agent3.select_action(state3, z_agent3)


        if len(memory_agent1) > args.batch_size:
            critic_1_loss1, critic_2_loss1, policy_loss1, ae_loss1, curr_loss1, next_loss1, kl_loss1 = agent1.update_parameters(memory_agent1, args.batch_size)
            critic_1_loss2, critic_2_loss2, policy_loss2, ae_loss2, curr_loss2, next_loss2, kl_loss2 = agent2.update_parameters(memory_agent2, args.batch_size)
            critic_1_loss3, critic_2_loss3, policy_loss3, ae_loss3, curr_loss3, next_loss3, kl_loss3 = agent3.update_parameters(memory_agent3, args.batch_size)
            # writer.add_scalar('autoencoder/ae_loss', ae_loss, agent.updates)
            # writer.add_scalar('autoencoder/z_curr_loss', curr_loss, agent.updates)
            # writer.add_scalar('autoencoder/z_next_loss', next_loss, agent.updates)
            # writer.add_scalar('autoencoder/kl_loss', kl_loss, agent.updates)
            # writer.add_scalar('SAC/critic_1', critic_1_loss, agent.updates)
            # writer.add_scalar('SAC/critic_2', critic_2_loss, agent.updates)
            # writer.add_scalar('SAC/policy', policy_loss, agent.updates)

        next_states, rewards, done, _ = env.step([action_agent1, action_agent2, action_agent3])
        reward1, reward2, reward3 = rewards[0], rewards[1], rewards[2]
        next_state1, next_state2, next_state3 = next_states[0], next_states[1], next_states[2]
        
        episode_steps += 1
        episode_reward1 += reward1
        episode_reward2 += reward2
        episode_reward3 += reward3

        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        memory_agent1.push_timestep(state1, action_agent1, reward1, next_state1, mask)
        memory_agent2.push_timestep(state2, action_agent2, reward2, next_state2, mask)
        memory_agent3.push_timestep(state3, action_agent3, reward3, next_state3, mask)
        state1 = next_state1
        state2 = next_state2
        state3 = next_state3

    z_prev_agent1 = np.copy(z_agent1)
    z_prev_agent2 = np.copy(z_agent2)
    z_prev_agent3 = np.copy(z_agent3)

    memory_agent1.push_interaction()
    memory_agent2.push_interaction()
    memory_agent3.push_interaction()
    writer.add_scalar('reward/episode_reward', episode_reward1, i_episode)
    print("Episode: {}, partner: {}, reward: {}".format(i_episode, env.partner, round(episode_reward1, 2)))

    if max(episode_reward1,episode_reward2,episode_reward3) == episode_reward2:
        reward2_chk += 100
    elif max(episode_reward1,episode_reward2,episode_reward3) == episode_reward3:
        reward3_chk += 100
    else:
        reward1_chk += 100
    

    # print(f"chk1:{reward1_chk}")
    # print(f"chk2:{reward2_chk}")
    # print(f"chk3:{reward3_chk}")
    if i_episode % 50 == 0:
        model_name = args.save_name + '_' + str(i_episode)
        if max(reward1_chk,reward2_chk,reward3_chk) == reward1_chk:
            print("agent1")
            agent1.save_model_witdr(model_name, args.dir_name)
            models_list.append(model_name)
            random_idx = np.random.randint(len(models_list))
            savemodel_name = models_list[random_idx]

            agent2.load_model(f"{args.dir_name}/{savemodel_name}")
            agent3.load_model(f"{args.dir_name}/{savemodel_name}")
        elif max(reward1_chk,reward2_chk,reward3_chk) == reward3_chk:
            print("agent3")
            agent3.save_model_witdr(model_name, args.dir_name)
            models_list.append(model_name)
            random_idx = np.random.randint(len(models_list))
            savemodel_name = models_list[random_idx]

            agent2.load_model(f"{args.dir_name}/{savemodel_name}")
            agent1.load_model(f"{args.dir_name}/{savemodel_name}")
        else:
            print("agent2")
            agent2.save_model_witdr(args.save_name+'_'+str(i_episode), args.dir_name)
            models_list.append(model_name)
            random_idx = np.random.randint(len(models_list))
            savemodel_name = models_list[random_idx]
            agent1.load_model(f"{args.dir_name}/{savemodel_name}")
            agent3.load_model(f"{args.dir_name}/{savemodel_name}")
        
        reward1_chk = 0
        reward2_chk = 0
        reward3_chk = 0
        
        #agent2.save_model(args.save_model + '_' + str(i_episode))
        #memory.save_buffer(args.save_name + '_' + str(i_episode))

    if len(models_list)>10:
        models_list = models_list[-7:]
    print(models_list)
# agent.save_model(args.save_name + '_' + str(i_episode))
# memory.save_buffer(args.save_name + '_' + str(i_episode))

#! python3

import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

class PolicyGradient(nn.Module):
    def __init__(self, state_size, action_size, lr_actor=1e-3, lr_critic=1e-3, mode='REINFORCE', n=128, gamma=0.99, device='cuda'):
        super(PolicyGradient, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.mode = mode
        self.n = n
        self.gamma = gamma

        self.device = device

        hidden_layer_size = 256

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size),
            nn.Softmax()
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1)
        )

        # initialize networks, optimizers, move networks to device
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)


    def forward(self, state):
        return(self.actor(state), self.critic(state))

    @torch.inference_mode
    def get_action(self, state, stochastic, cuda=1):
        # if stochastic, sample using the action probabilities, else get the argmax
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION
        state = torch.tensor(state).to(self.device) if cuda else torch.tensor(state)
        probs = self.actor(state).detach().cpu().numpy()
        if stochastic:
            act = np.random.choice(a=len(probs), p=probs)
        else:
            act = np.argmax(probs)
        return act


    def calculate_n_step_bootstrap(self, rewards_tensor, values):
        # calculate n step bootstrap
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION
        pass


    def train(self, states, actions, rewards):
        # train the agent using states, actions, and rewards
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION
        pass
    
    def calc_gt(self, rwds):
        sum = 0
        for i,rwd in enumerate(rwds):
            sum += self.gamma**i*rwd
        return sum 

    # N-step A2C 
    def run(self, env, max_steps, num_episodes, cuda=1):
        self.critic = self.critic.to(self.device) if cuda else self.critic
        self.actor = self.actor.to(self.device) if cuda else self.actor
        def te(x, cuda=cuda): return torch.tensor(x) if not cuda else torch.tensor(x).to(self.device)
        total_rewards = []
        N,gamma = self.n, self.gamma
        st = time.time()
        for ep in range(num_episodes):
            steps = 0
            s, _ = env.reset()
            buffer = []
            while steps < max_steps:
                a = self.get_action(s, True)
                s_t, rwd, done, info, _, = env.step(a)
                buffer.append((s,a,rwd))
                s = s_t
                if done: break
                steps+=1
            T = len(buffer)
            # super messy calculate rewards backwards
            Gs = [0] 
            Gts = []
            Vts = []
            Ps = []
            Ps_idxs = []
            v_ends = []
            for t in range(T-1, -1, -1):
                if t+N<T:
                    v_ends.append(te(buffer[t][0]))
                # t=T-1, then R
                G = buffer[t][2] + gamma*Gs[-1]
                Gt = te(G) #+ te(gamma**N)*v_end
                Vt = te(buffer[t][0])
                Gs.append(G)
                Gts.append(Gt)
                Vts.append(Vt)
                Ps.append(te(buffer[t][0]))
                Ps_idxs.append(buffer[t][1])
            assert len(Gts) ==  T
            # run all nets in batch 
            v_ends = self.critic(torch.stack(v_ends)) if v_ends else []
            Gts = torch.stack(Gts).view(-1, 1) 
            # inefficient add v_ends but oh well
            for t in range(len(Gts)):
                if t+N<T:
                    Gts[t] += v_ends[t] 
                else:
                    break
            Vts = self.critic(torch.stack(Vts))
            Ps = torch.log(self.actor(torch.stack(Ps)))
            Ps_idxs = torch.tensor(Ps_idxs).to(Ps.device)
            Ps_idxs = Ps_idxs.unsqueeze(-1)
            Ps = torch.gather(Ps, 1, Ps_idxs).squeeze(-1)
            
            L_act = ((Gts.detach()-Vts.detach())*Ps).sum() / -T
            L_crit = ((Gts-Vts)**2).sum() / T

            self.optimizer_critic.zero_grad()  # Clear existing gradients
            L_crit.backward()  # Backpropagate the critic loss
            self.optimizer_critic.step()  # Update critic network weights

            self.optimizer_actor.zero_grad()  # Clear existing gradients
            L_act.backward()  # Backpropagate the actor loss
            self.optimizer_actor.step()  # Update actor network weights
            if ep % 100 == 0:
                e = time.time()
                print(f'-- episode {ep} --')
                print(f'{e-st:7.2f}')
                st = time.time()
                # 20 IID test eps
                mean_rwds = []
                for _ in range(20):
                    test_steps = 0
                    s,_ = env.reset()
                    test_rwds = []
                    while test_steps < max_steps:
                        a = self.get_action(s, True)
                        s_t, rwd, done, info, _ = env.step(a)
                        test_rwds.append(rwd)
                        s = s_t
                        if done: break
                        test_steps+=1                    
                    rwd = sum(test_rwds)
                    mean_rwds.append(rwd)
                mean_rwd = sum(mean_rwds) / len(mean_rwds)
                total_rewards.append(mean_rwd)
                print(f'mean rwd per 20 trials: {mean_rwd}')
                    
        print(len(total_rewards))
        return(total_rewards)


def graph_agents(graph_name, agents, env, max_steps, num_episodes):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * graph_every for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Total Reward')
    fig.savefig(f'./graphs/{graph_name}.png')
    plt.close(fig)
    print(f'Finished: {graph_name}')


def parse_args():
    mode_choices = ['REINFORCE', 'REINFORCE_WITH_BASELINE', 'A2C']

    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--mode', type=str, default='REINFORCE', choices=mode_choices, help='Mode to run the agent in')
    parser.add_argument('--n', type=int, default=100, help='The n to use for n step A2C')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=3500, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    return parser.parse_args()

def main():
    import time
    args = parse_args()

    env = gym.make(args.env_name)
    state_sz = env.observation_space.shape[0]
    action_sz = env.action_space.n
    avg_ep_rwds = []
    s = time.time()
    for i in range(args.num_runs):
        print(f'--- run {i} ---')
        PG = PolicyGradient(state_sz, action_sz, lr_actor=1e-3, lr_critic=1e-3, n=args.n, gamma=0.99, device='cpu')
        PG.device = 'cpu'
        rwds = PG.run(env, args.max_steps, args.num_episodes)
        avg_ep_rwds.append(rwds)
    e = time.time()
    print((e-s)/60)

    avg_ep_rwds = np.array(avg_ep_rwds)
    #assert avg_ep_rwds.shape == (args.num_runs, args.num_episodes // 100)  # Adjusted according to your example

    # Calculate mean, max, and min undiscounted return across trials
    mean_rwds = np.mean(avg_ep_rwds, axis=0)
    max_rwds = np.max(avg_ep_rwds, axis=0)
    min_rwds = np.min(avg_ep_rwds, axis=0)

    # Plotting
    episodes = np.arange(0, args.num_episodes, 100)  # Assuming rewards are recorded every 100 episodes
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean_rwds, label='Mean Undiscounted Return')
    plt.fill_between(episodes, min_rwds, max_rwds, color='gray', alpha=0.2, label='Min/Max Undiscounted Return')
    plt.xlabel('Number of Training Episodes')
    plt.ylabel('Mean Undiscounted Return')
    plt.title(f'Performance over Training Episodes, N={args.n}')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_over_training_episodes.png')
    plt.show()

if __name__ == '__main__':
    main()

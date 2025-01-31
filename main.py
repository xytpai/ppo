import torch
import matplotlib.pyplot as plt
import env
import model
import numpy as np
import torch.optim as optim


def main(height=9, width=9, num_eat=20):
    agent = model.MLP(height, width)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    # while True:
    overall_reward_list = []
    for episode in range(40000):
        playground = env.PlayGround(height=height, width=width, num_eat=num_eat)
        playground.set_random()
        action_logprob_list = []
        state_list = []
        reward_list = []

        for step in range(50):
            state = playground.get_space()
            action_probs = agent(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            state_list.append(state)
            action_logprob_list.append(log_prob)
            reward = playground.interact(model.decode_action(action))
            reward_list.append(reward)
        
        discounted_rewards = []
        cumulative_reward = 0
        gamma = 0.9
        for r in reversed(reward_list):
            cumulative_reward = r + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-6)

        loss = 0
        for log_prob, reward in zip(action_logprob_list, discounted_rewards):
            loss -= log_prob * reward
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        overall_reward_list.append(np.sum(reward_list))
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {np.sum(reward_list)}, loss: {loss.item()}")
    plt.plot(overall_reward_list)
    plt.show()

    playground = env.PlayGround(height=height, width=width, num_eat=num_eat)
    playground.set_random()
    playground.print()
    for step in range(100):
        with torch.no_grad():
            ch = input('continue:')
            state = playground.get_space()
            action_probs = agent(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            playground.interact(model.decode_action(action))
            playground.print()


if __name__ == '__main__':
    main()

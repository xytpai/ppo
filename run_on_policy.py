import torch
import env
import model
import matplotlib.pyplot as plt


def main(
        batch_size=4,
        height=12,
        width=12,
        num_eat=20,
        nstep=50,
        nepisode=10000,
        gamma=0.9,
        lr=1e-3):
    
    agent = model.PolicyModel(height, width)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    overall_rewards_list = []
    for episode in range(nepisode):
        
        playground = env.PlayGround(batch_size, height, width, num_eat)
        playground.set_random()

        log_probs = []
        rewards = []

        for step in range(nstep):
            state = playground.get_space()
            action_probs = agent(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            reward = playground.interact(model.decode_action(action))
            log_probs.append(log_prob)
            rewards.append(reward)
        
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.stack(discounted_rewards, dim=-1) # b, step
        discounted_rewards = (discounted_rewards - discounted_rewards.mean(dim=1, keepdim=True)
                                ) / (discounted_rewards.std(dim=1, keepdim=True) + 1e-6)
        log_probs = torch.stack(log_probs, dim=-1)
        rewards = torch.stack(rewards, dim=-1)

        loss = - (log_probs * discounted_rewards).sum(dim=1).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

         # b, step
        overall_rewards = rewards.sum(dim=1).mean().item()
        overall_rewards_list.append(overall_rewards)
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {overall_rewards}, Loss: {loss}")
    plt.plot(overall_rewards_list)
    plt.show()

    playground = env.PlayGround(1, height, width, num_eat)
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

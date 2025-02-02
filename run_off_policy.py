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
        lr=1e-3,
        clip_epsilon=0.2):
    
    agent = model.PolicyModel(height, width)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    overall_reward_list = []
    for episode in range(nepisode):
        
        playground = env.PlayGround(batch_size, height, width, num_eat)
        playground.set_random()

        observe_actions = []
        observe_log_probs = []
        observe_states = []
        observe_rewards = []

        with torch.no_grad():
            for step in range(nstep):
                state = playground.get_space()
                action_probs = agent(state)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                rewards = playground.interact(model.decode_action(action))
                observe_actions.append(action.clone())
                observe_log_probs.append(log_prob.clone())
                observe_states.append(state.clone())
                observe_rewards.append(rewards.clone())
            
            discounted_rewards = []
            cumulative_reward = 0
            for r in reversed(observe_rewards):
                cumulative_reward = r + gamma * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)
            discounted_rewards = torch.stack(discounted_rewards, dim=-1) # b, step
            discounted_rewards = (discounted_rewards - discounted_rewards.mean(dim=1, keepdim=True)
                                ) / (discounted_rewards.std(dim=1, keepdim=True) + 1e-6)
            observe_log_probs = torch.stack(observe_log_probs, dim=-1) # b, step
            observe_rewards = torch.stack(observe_rewards, dim=-1) # b, step

        actual_log_probs = []
        entropy_loss = 0
        for state, action in zip(observe_states, observe_actions):
            action_probs = agent(state)
            dist = torch.distributions.Categorical(action_probs)
            entropy_loss += dist.entropy().mean()
            log_prob = dist.log_prob(action)
            actual_log_probs.append(log_prob)
        actual_log_probs = torch.stack(actual_log_probs, dim=-1) # b, step

        ratio = torch.exp(actual_log_probs - observe_log_probs.detach())
        loss = - torch.min(
            ratio * discounted_rewards, 
            torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon
        ) * discounted_rewards).sum(dim=1).mean() - 0.01 * entropy_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

         # b, step
        observe_overall_reward = observe_rewards.sum(dim=1).mean().item()
        overall_reward_list.append(observe_overall_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {observe_overall_reward}, Loss: {loss}")
    plt.plot(overall_reward_list)
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

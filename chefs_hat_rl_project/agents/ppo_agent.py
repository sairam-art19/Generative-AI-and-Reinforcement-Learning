import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from agents.networks import ActorCritic

class PPOAgent:
    def __init__(self, state_dim, action_dim,
                 lr=3e-4, gamma=0.99, eps_clip=0.2):

        self.gamma = gamma
        self.eps_clip = eps_clip

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.memory = []

    def select_action(self, state):
        state = torch.FloatTensor(np.array(state))

        logits, value = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()

        self.memory.append({
            "state": state,
            "action": action,
            "logprob": dist.log_prob(action),
            "value": value
        })

        return action.item()

    def store_reward(self, reward):
        self.memory[-1]["reward"] = reward

    def update(self):

        states = torch.stack([m["state"] for m in self.memory])
        actions = torch.stack([m["action"] for m in self.memory])
        old_logprobs = torch.stack([m["logprob"] for m in self.memory])

        # Compute returns
        returns = []
        discounted = 0

        for step in reversed(self.memory):
            discounted = step["reward"] + self.gamma * discounted
            returns.insert(0, discounted)

        returns = torch.tensor(returns, dtype=torch.float32)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Forward pass
        logits, values = self.policy(states)
        dist = Categorical(logits=logits)

        logprobs = dist.log_prob(actions)
        ratios = torch.exp(logprobs - old_logprobs.detach())

        advantages = returns - values.squeeze().detach()

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios,
                            1 - self.eps_clip,
                            1 + self.eps_clip) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = torch.nn.functional.mse_loss(
            values.squeeze(),
            returns
        )

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = []

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
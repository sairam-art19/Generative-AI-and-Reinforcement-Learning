# Chef's Hat Reinforcement Learning Project

## Sparse / Delayed Reward Variant

### Student ID: 16943218

------------------------------------------------------------------------

# 1. Introduction

This project implements and evaluates a Reinforcement Learning (RL)
agent for the Chef's Hat Gym environment. The primary focus corresponds
to the assigned variant:

**Sparse / Delayed Reward Variant (ID mod 7 = 3)**

The goal of this variant is to investigate the challenges introduced by
delayed rewards and analyse whether reward shaping techniques improve
learning stability and convergence speed.

The agent is implemented using Proximal Policy Optimisation (PPO) with
an actor--critic neural network architecture.

------------------------------------------------------------------------

# 2. Project Objectives

-   Integrate and interact correctly with the Chef's Hat Gym
    environment.
-   Implement a PPO-based reinforcement learning agent.
-   Investigate learning under sparse, delayed reward conditions.
-   Compare sparse reward and shaped reward strategies.
-   Log training performance automatically.
-   Save models and results for reproducibility.

------------------------------------------------------------------------

# 3. Algorithm Selection

## Proximal Policy Optimisation (PPO)

PPO was selected because:

-   It provides stable policy updates.
-   It performs well in stochastic environments.
-   It handles discrete action spaces efficiently.
-   It is widely used in modern reinforcement learning research.

### Architecture

-   Actor--Critic network
-   Two hidden layers (256 units each)
-   ReLU activation
-   Categorical action distribution
-   Clipped surrogate objective
-   Mean Squared Error value loss

------------------------------------------------------------------------

# 4. Reward Strategy (Variant Focus)

Two reward configurations are implemented:

## 4.1 Sparse Reward (Baseline)

-   Only final match reward is used.
-   Represents pure delayed credit assignment.

## 4.2 Reward Shaping

-   Small positive rewards for favourable sub-events.
-   Small penalties for unfavourable actions.
-   Designed to improve credit assignment and reduce variance.

------------------------------------------------------------------------

# 5. Project Structure

chefs-hat-rl-project/ │ ├── agents/ ├── training/ ├── utils/ ├──
experiments/ ├── results/ │ ├── model/ │ ├── csv/ │ └── plot/ ├──
ChefsHatGYM-main/ ├── README.md └── requirements.txt

------------------------------------------------------------------------

# 6. Installation

## Create Virtual Environment

python -m venv .venv

Activate (Windows):

.venv`\Scripts`{=tex}`\activate`{=tex}

## Install Dependencies

pip install -r requirements.txt

If needed:

pip install numpy==1.24.4

------------------------------------------------------------------------

# 7. Running Experiments

From project root:

python -m experiments.run_sparse_baseline

Training will begin automatically.

------------------------------------------------------------------------

# 8. Automatic Output Generation

After training, the following structure is created automatically:

results/ │ ├── model/ │ └── ppo_model.pt ├── csv/ │ └──
training_results.csv └── plot/ └── training_plot.png

------------------------------------------------------------------------

# 9. Evaluation Metrics

-   Episode reward progression
-   Learning curve trends
-   Convergence speed
-   Training stability

------------------------------------------------------------------------

# 10. Limitations

-   PPO without GAE
-   Single-seed experiments
-   Limited hyperparameter tuning
-   Reward shaping may introduce bias

------------------------------------------------------------------------

# 11. Conclusion

This project demonstrates successful PPO implementation and empirical
investigation of sparse versus shaped reward strategies in a delayed
reward multi-agent environment.

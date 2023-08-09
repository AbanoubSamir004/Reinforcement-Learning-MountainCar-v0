# Reinforcement Learning for MountainCar-v0


## Project Overview

This repository contains the implementation of various Reinforcement Learning algorithms to tackle the MountainCar-v0 environment from OpenAI Gym.

## Implemented Algorithms

1. **Monte Carlo Method**: We implemented the Monte Carlo algorithm, which involves learning from episodes of interactions with the environment. By sampling trajectories and updating the value function based on the cumulative rewards obtained, the algorithm aims to approximate the optimal policy.

2. **Q-Learning**: Q-Learning is a model-free algorithm that iteratively updates the Q-values of state-action pairs. Our implementation focused on learning the optimal action-value function and making decisions based on the highest Q-values.

3. **SARSA**: Another model-free algorithm, SARSA, involves updating the Q-values while taking into account the next action that would be chosen under the current policy. This approach aims to strike a balance between exploration and exploitation.

## Results and Observations

After thorough implementation and testing, we found that the Q-Learning algorithm outperformed the other methods for solving the MountainCar-v0 environment. The following observations were made:

- **Faster Convergence**: Q-Learning demonstrated faster convergence compared to the Monte Carlo and SARSA methods. This means that the Q-Learning algorithm reached a near-optimal policy more quickly, requiring fewer iterations.

- **Higher Average Rewards**: Q-Learning achieved higher average rewards in comparison to the other algorithms. This implies that the car was able to reach the flag more consistently and efficiently while maximizing the rewards obtained along the way.

## Results
**MonteCarlo Game**

![MonteCarlo_Game](https://github.com/AbanoubSamir004/Reinforcement-Learning-MountainCar-v0/assets/60902991/9dee8335-d03b-4253-b3ca-8fa328a4b589)

**Q-Learning Game**

![Q-Learning_Game](https://github.com/AbanoubSamir004/Reinforcement-Learning-MountainCar-v0/assets/60902991/e4f21285-a296-418a-bbfe-b96b118830f0)

**Sarsa Game**

![Sarsa_Game](https://github.com/AbanoubSamir004/Reinforcement-Learning-MountainCar-v0/assets/60902991/0ec7cf96-95ad-47d6-8696-9e6c7d5b9172)

**Average Rewards Comparison**

![Rewards__Algorithms_comparison_mountaincar](https://github.com/AbanoubSamir004/Reinforcement-Learning-MountainCar-v0/assets/60902991/8ea5e143-ad9c-4915-80da-393de01a1b94)

## Conclusion

We successfully implemented and compared Monte Carlo, Q-Learning, and SARSA algorithms for solving the MountainCar-v0 environment. Through meticulous testing, we observed that Q-Learning outshone the other methods in terms of convergence speed and average rewards.

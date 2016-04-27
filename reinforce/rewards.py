import numpy as np

class RewardParser:
  def __init__(self, observations, dimensions):
    self.observations = observations
    self.state_count = dimensions['state_count']

  def rewards(self):
    print('COMPUTING REWARDS')
    total_state_rewards = np.zeros(self.state_count)
    total_state_visits = np.zeros(self.state_count)

    for observation in self.observations:
      visits = float(len(observation['state_transitions']))
      reward_per_visit = observation['reward'] / visits

      for state_transition in observation['state_transitions']:
        state = state_transition['state']
        total_state_rewards[state] += reward_per_visit
        total_state_visits[state] += 1

    average_state_rewards = total_state_rewards / total_state_visits
    average_state_rewards = np.nan_to_num(average_state_rewards)

    return average_state_rewards
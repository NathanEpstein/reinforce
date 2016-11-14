import numpy as np

class PolicyParser:
  def __init__(self, dimensions):
    self.state_count = dimensions['state_count']
    self.action_count = dimensions['action_count']

  def policy(self, P, rewards):
    print('COMPUTING POLICY')

    best_policy = np.zeros(self.state_count)
    state_values = np.zeros(self.state_count)

    GAMMA = 0.9
    ITERATIONS = 125
    for i in range(ITERATIONS):
      print ("iteration: {0} / {1}".format(i + 1, ITERATIONS))

      for state in range(0, self.state_count):
        state_value = -float('Inf')

        for action in range(0, self.action_count):
          action_value = 0

          for state_ in range(0, self.state_count):
            action_value += (P[state][action][state_] * state_values[state_] * GAMMA)

          if (action_value >= state_value):
            state_value = action_value
            best_policy[state] = action

        state_values[state] = rewards[state] + state_value

    return best_policy
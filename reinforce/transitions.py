import numpy as np

class TransitionParser:
  def __init__(self, observations, dimensions):
    self.observations = observations
    self.state_count = dimensions['state_count']
    self.action_count = dimensions['action_count']

  def transition_probabilities(self):
    print('COMPUTING TRANSITIONS')
    transition_count = self._count_transitions()
    return self._parse_probabilities(transition_count)

  def _count_transitions(self):
    transition_count = np.zeros((self.state_count, self.action_count, self.state_count))

    for observation in self.observations:
      for state_transition in observation['state_transitions']:
        state = state_transition['state']
        action = state_transition['action']
        state_ = state_transition['state_']

        transition_count[state][action][state_] += 1

    return transition_count

  def _parse_probabilities(self, transition_count):
    P = np.zeros((self.state_count, self.action_count, self.state_count))

    for state in range(0, self.state_count):
      for action in range(0, self.action_count):

        total_transitions = float(sum(transition_count[state][action]))

        if (total_transitions > 0):
          P[state][action] = transition_count[state][action] / total_transitions
        else:
          P[state][action] = 1.0 / self.state_count

    return P
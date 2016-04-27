class StateActionEncoder:
  def __init__(self, observations):
    self.observations = observations
    self._parse_states_and_actions()

  def parse_dimensions(self):
    return {
      'state_count': len(self.int_to_state),
      'action_count': len(self.int_to_action)
    }

  def observations_to_int(self):
    for observation in self.observations:
      for transition in observation['state_transitions']:
        transition['state'] = self.state_to_int[transition['state']]
        transition['state_'] = self.state_to_int[transition['state_']]
        transition['action'] = self.action_to_int[transition['action']]

  def parse_encoded_policy(self, encoded_policy):
    policy = {}
    for index, encoded_action in enumerate(encoded_policy):
      state = self.int_to_state[index]
      action = self.int_to_action[int(encoded_action)]
      policy[state] = action

    return policy

  def _parse_states_and_actions(self):
    state_dict, action_dict = {}, {}
    state_array, action_array = [], []
    state_index, action_index = 0, 0

    for observation in self.observations:
      for transition in observation['state_transitions']:
        state = transition['state']
        action = transition['action']

        if state not in state_dict.keys():
          state_dict[state] = state_index
          state_array.append(state)
          state_index += 1

        if action not in action_dict.keys():
          action_dict[action] = action_index
          action_array.append(action)
          action_index += 1

    self.state_to_int = state_dict
    self.action_to_int = action_dict
    self.int_to_state = state_array
    self.int_to_action = action_array


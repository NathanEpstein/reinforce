from encoding import StateActionEncoder
from rewards import RewardParser
from transitions import TransitionParser
from policy import PolicyParser

class MarkovAgent:
  def __init__(self, observations):
    # encode observation data as int values
    self.state_action_encoder = StateActionEncoder(observations)
    self.state_action_encoder.observations_to_int()
    dimensions = self.state_action_encoder.parse_dimensions()

    # create reward, transition, and policy parsers
    self.reward_parser = RewardParser(observations, dimensions)
    self.transition_parser = TransitionParser(observations, dimensions)
    self.policy_parser = PolicyParser(dimensions)

  def learn(self):
    R = self.reward_parser.rewards()
    P = self.transition_parser.transition_probabilities()

    # learn int-encoded policy and convert to readable dictionary
    encoded_policy = self.policy_parser.policy(P, R)
    self.policy = self.state_action_encoder.parse_encoded_policy(encoded_policy)

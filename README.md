# reinforce

<img src="./MDP.png">

A 'plug and play' reinforcement learning library in Python.

Infers a Markov Decision Process from data and solves for the optimal policy.

Implementation based on Andrew Ng's <a href="http://cs229.stanford.edu/notes/cs229-notes12.pdf">notes.</a>

More information related to this project can be found <a href="https://github.com/NathanEpstein/pydata-reinforce">here.</a>

## Example Usage

```python

observations = [
  { 'state_transitions': [
      { 'state': 'low', 'action': 'climb', 'state_': 'mid' },
      { 'state': 'mid', 'action': 'climb', 'state_': 'high' },
      { 'state': 'high', 'action': 'sink', 'state_': 'mid' },
      { 'state': 'mid', 'action': 'sink', 'state_': 'low' },
      { 'state': 'low', 'action': 'sink', 'state_': 'bottom' }
    ],
    'reward': 0
  },
  { 'state_transitions': [
      { 'state': 'low', 'action': 'climb', 'state_': 'mid' },
      { 'state': 'mid', 'action': 'climb', 'state_': 'high' },
      { 'state': 'high', 'action': 'climb', 'state_': 'top' },
    ],
    'reward': 0
  }
]

trap_states = [
  {
    'state_transitions': [
      { 'state': 'bottom', 'action': 'sink', 'state_': 'bottom' },
      { 'state': 'bottom', 'action': 'climb', 'state_': 'bottom' }
    ],
    'reward': 0
  },
  {
    'state_transitions': [
      { 'state': 'top', 'action': 'sink', 'state_': 'top' },
      { 'state': 'top', 'action': 'climb', 'state_': 'top' },
    ],
    'reward': 1
  },
]

from learn import MarkovAgent
mark = MarkovAgent(observations + trap_states)
mark.learn()

print(mark.policy)
# {'high': 'climb', 'top': 'sink', 'bottom': 'sink', 'low': 'climb', 'mid': 'climb'}
# NOTE: policy in top and bottom states is chosen randomly (doesn't affect state)

```

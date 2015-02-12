from __future__ import division

#create maps from states<->ints, actions<->ints
#convert single reward to step rewards if applicable
def parse(*args):
  # obs and R, 1 reward per observation
  if len(args) > 1:
    obs = args[0]
    R = args[1]
  # only obs, 1 reward per step in each observation
  else:
    obs = args[0]
    actMap = []
    stateMap = []
    for o in range(0,len(obs)):
      for t in range(0,len(obs[o])):
        state = obs[o][t][0]
        action = obs[o][t][1]
        # reward = obs[o][t][2]
        if (state not in stateMap):
          stateMap.append(state)
        obs[o][t][0] = stateMap.index(state)
        if (action not in actMap):
          actMap.append(action)
        obs[o][t][1] = actMap.index(action)
    return [stateMap,actMap]

from __future__ import division

#create maps from states<->ints, actions<->ints
#convert single reward to step rewards if applicable
def parse(*args):
  obs = args[0]
  # obs and R, 1 reward per observation
  # convert obs to include rewards each step
  if len(args) > 1:
    R = args[1]
    for o in range(0,len(obs)):
      totalReward = R[o]
      stepReward = totalReward/len(obs[o])
      for t in range (0,len(obs[o])):
        obs[o][t].append(stepReward)

  #create maps from actions and states to integers
  actMap = []
  stateMap = []
  for o in range(0,len(obs)):
    for t in range(0,len(obs[o])):
      state = obs[o][t][0]
      action = obs[o][t][1]
      if (state not in stateMap):
        stateMap.append(state)
      obs[o][t][0] = stateMap.index(state)
      if (action not in actMap):
        actMap.append(action)
      obs[o][t][1] = actMap.index(action)
  return [stateMap,actMap,obs]
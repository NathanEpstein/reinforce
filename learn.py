import parse as par
import mdpmodel as mod
import mdpsolve as sol

#make results class

# takes 3d list of observations & reward list(if step-wise rewards not included in observations)
# [obs],[obs,gamma],[obs,gamma,R]
def learn(*args):
  obs_ = args[0]
  gamma = 1
  if (len(args) > 1):
    gamma = args[1]
  if (len(args) > 2):
    R = args[2]
    parsed = par.parse(obs_,R)
  else:
    parsed = par.parse(obs_)
  #parsed = [stateMap,actionMap,observations]

  stateMap = parsed[0]
  actMap = parsed[1]
  obs = parsed[2]
  model = mod.model(len(stateMap),len(actMap),obs)

  P = model[0]
  R = model[1]
  policy = sol.policy(P,gamma,R)

  #map integer policy and action back to
  strat = {}
  for i in range(0,len(policy)):
    strat[stateMap[i]] = actMap[policy[i]]

  #return strategy
  return strat

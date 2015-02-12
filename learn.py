from __future__ import division

#create maps from states<->ints, actions<->ints
#convert single reward to step rewards if applicable
def learn(*args):
  # obs and R, 1 reward per observation
  if len(args) > 1:
    obs = args[0]
    R = args[1]
  # only obs, 1 reward per step in each observation
  else:
    obs = args[0]

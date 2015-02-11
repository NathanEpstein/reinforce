def model(s,a,obs):
  #initialize counters for reward and probability transition matrix
  pcount = [[[0]*s]*a]*s
  rcount = [[0,0]]*s

  # count state and reward observations
  for o in range (0,len(obs)):
    for t in range (0,len(obs[o])):
      #REWARD TRACKER:
      state = obs[o][t][0]
      state_ = obs[o][t+1][0]
      action = obs[o][t][1]
      reward = obs[o][t][2]

      #increment cumulative reward for observed state
      rcount[state][0] += reward
      #increment state visits count
      rcount[state][1] += 1

      #PROBABILITY TRANSITION TRACKER
      #increment count of transitions, C_sa[s']
      pcount[state][action][state_] += 1

  # compute R[s]
  R = [0]*s
  for s in range(0,s):
    R[s] = rcount[s][0]/rcount[s][1]

  # compute P_sa[s']





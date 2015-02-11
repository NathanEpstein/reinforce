def checkConverge(new,old):
  totalDif = 0
  totalOld = 0
  for i in range(0,len(old)):
    totalDif += abs(new[i] - old[i])
    totalOld += abs(old[i])
  return (totalDif < 0.001*totalOld)


#S(states) and A(actions) are implicitly integers implicitly defined by P
#(P[state][action][state_] = p(state->state_|action)
def policy(P,gamma,R):
  pol = [0]*len(P)
  V = [0] * len(P)
  converged = False
  while not (converged):
    V_ = V[:] #track previous iteration for comparison
    #iterate over each state
    for s in range(0,len(P)):
      futureVal = -float('Inf')
      #iterate over each action
      for a in range(0,len(P[s])):
        arg = 0
        val = 0
        #iterate over each destination state
        for s_ in range(0,len(P[s][a])):
          val += (gamma*(P[s][a][s_] * V[s_]))
        if (val > futureVal):
          futureVal = val
          pol[s] = a
      V[s] = R[s] + futureVal
    converged = checkConverge(V, V_)
  return pol



#S(states) and A(actions) are implicitly integers implicitly defined by P
#(P[state][action][state_] = p(state->state_|action)
def policy(P,gamma,R):
  pol = [0]*len(P)
  V = [0] * len(P)
  #update 7-9 later to be convergence instead of just many loops
  count = 0
  while (count < 1000):
    count += 1
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
  return pol


def main():
  P = [[[0.25,0.25,0.5],[0,1,0],[0,0,1]],[[0.25,0.25,0.5],[0,0,1],[0,1,0]],[[0.25,0.25,0.5],[0,0,1],[0,1,0]]]
  gamma = 0.95
  R = [1,10,1]

  x = policy(P,gamma,R)
  print(x)


main()

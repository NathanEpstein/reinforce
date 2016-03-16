#reinforce

A 'plug and play' reinforcement learning library in Python.

Infers a Markov Decision Process from data and solves for the optimal policy.

Implementation based on Andrew Ng's <a href="http://cs229.stanford.edu/notes/cs229-notes12.pdf">notes.</a>

##Motivation

<a href="https://github.com/scikit-learn/scikit-learn">scikit-learn</a> provides excellent tools for supervised and unsupervised learning but explicitly does not deal with reinforcement learning.

reinforce is intended to compliment the functionality of scikit-learn and together form a more complete machine learning toolkit.

##Install

`pip install reinforce`

##Usage

```python
import learn as l

l.learn(obs)
# or
l.learn(obs,gamma)
# or
l.learn(obs,gamma,R)

```
###Output
```python
import learn as l

model = l.learn(obs,gamma,R)
```

`model` is a dictionary which contains the estimated optimal action for each state.

###Inputs

####obs
obs is a 3-dimensional list. Each element of obs is a 2-d list of time-steps. Each time-step is a list of the form [state, action, reward] if no R is specified, or [state,action] if R is specified. See examples for more detail.

```python
obsA = [[stateA1,actionA1,rewardA1],[stateA2,actionA2,rewardA2],...]
obsB = [[stateB1,actionB1,rewardB1],[stateB2,actionB2,rewardB2],...]

obs = [obsA,obsB]
```

####gamma
A value specifying the discount factor for future rewards. In the range (0,1]

```python
gamma = 0.95
```

####R
If rewards are ommitted in obs, R is a list of length = len(obs) specifying the reward for each observation. See examples for more detail.

```python
obsA = [[stateA1,actionA1],[stateA2,actionA2],...]
obsB = [[stateB1,actionB1],[stateB2,actionB2],...]

obs = [obsA,obsB]
R = [rewardA,rewardB]
```
##Examples
<img src="example.png">

###Example1
```python
import learn as l

def main():
  obs1 = [["A","F",0],["A","L",0],["Prize","F",1]]
  obs2 = [["C","R",0],["D","F",0],["B","B",0],["D","L",0]]
  obs3 = [["C","F",0],["A","R",0],["B","L",0],["A","L",0],["Prize","L",1]]

  obs = [obs1,obs2,obs3]
  gamma = 0.95 #slight discount to rewards farther in the future

  model = l.learn(obs,gamma)
  # or try it without gamma
  # model = l.learn(obs)

  print ("From these three paths, the learned strategy is: ")
  print (model)

  #note that many transition probabilities are estimated as uniform because there isn't yet data
main()

From these three paths, the learned strategy is:
# {'A': 'L', 'C': 'F', 'B': 'L', 'Prize': 'F', 'D': 'L'}
```

###Example2

```python
import learn as l

def main():
  obs1 = [["A","F"],["A","L"],["Prize","F"]]
  obs2 = [["C","R"],["D","F"],["B","B"],["D","L"]]
  obs3 = [["C","F"],["A","R"],["B","L"],["A","L"],["Prize","L"]]

  obs = [obs1,obs2,obs3]
  gamma = 1 #no discount
  rewards = [1,0,1]

  model = l.learn(obs,gamma,rewards)

  print ("From these three paths, the learned strategy is: ")
  print (model)

  #note that many transition probabilities are estimated as uniform because there isn't yet data
main()

# From these three paths, the learned strategy is:
# {'A': 'R', 'C': 'F', 'B': 'L', 'Prize': 'F', 'D': 'L'}
```

###Practical Note

It is worth mentioning that the algorithm will learn (and the strategy will improve) **much faster if rewards for each step can be included** (as opposed to a reward for each observation). This is only for the rare case in which the user can choose between the two types of data - in practice it is more likely that only one reward per observation will be available.

This can be seen in the above examples - the model in example 1 is more effective than that in example 2 (clearly, going left in state A is preferable to going right in state A).


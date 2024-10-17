# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            newVal = self.values.copy()

            for state in self.mdp.getStates():
                actions = self.mdp.getPossibleActions(state)
                if not actions or self.mdp.isTerminal(state):
                    newVal[state] = 0  
                    continue
                qValues = []
                for action in actions:
                    qValues.append(self.computeQValueFromValues(state, action))
                
                newVal[state] = max(qValues)

            self.values = newVal  


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qVal = 0
        states = self.mdp.getTransitionStatesAndProbs(state, action)
        for transitionState in states:
            nextState = transitionState[0]
            prob = transitionState[1]

            reward = self.mdp.getReward(state, action, nextState)

            qVal += prob * (reward + self.discount * self.values[nextState])

        return qVal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        
        policies = util.Counter()

        for action in actions:
            policies[action] = self.getQValue(state, action)

        if not policies:
            return None
        return policies.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
    
        for iter in range(self.iterations):
            index = iter%len(states)
            state = states[index]

            if not self.mdp.isTerminal(state):
                maxVal = float('-inf')
                
                for action in self.mdp.getPossibleActions(state):
                    currVal = 0
                    #calculate q values for all potential states
                    for newState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        currReward = self.mdp.getReward(state, action, newState)
                        
                        currVal += prob * (currReward + self.discount * self.getValue(newState))
                    maxVal = max(currVal, maxVal)
                self.values[state] = maxVal

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        priorityQ = util.PriorityQueue()
        predecessors = {}
        for s in self.mdp.getStates():
            predecessors[s] = set() 
        
        for s in self.mdp.getStates():
            #for each non-terminal state s
            if not self.mdp.isTerminal(s):
                #predecessors of a state s as all states that have a nonzero probability of reaching s by taking some action a
                for action in self.mdp.getPossibleActions(s):
                    transitions = self.mdp.getTransitionStatesAndProbs(s, action)
                    for transition in transitions:
                        #nonzero probability 
                        if transition[1] > 0:
                            predecessors[transition[0]].add(s)

                #find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s 
                maxQ = float('-inf')
                for action in self.mdp.getPossibleActions(s):
                    maxQ = max(maxQ, self.computeQValueFromValues(s, action) )
                
                #push s into the priority queue with priority -diff 
                diff = abs(self.values[s] - maxQ)
                priorityQ.push(s, -diff)

        for i in range(self.iterations):
            #if the priority queue is empty, then terminate
            if priorityQ.isEmpty():
                break

            #pop a state s off the priority queue.
            s = priorityQ.pop()

            #update the value of s (if it is not a terminal state) in self.values
            if not self.mdp.isTerminal(s):
                maxQ = float('-inf')
                for action in self.mdp.getPossibleActions(s):
                    maxQ = max(maxQ, self.computeQValueFromValues(s, action))
        
                self.values[s] = maxQ

            #for each predecessor p of s
            for p in predecessors[s]:
                #find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p 
                maxQ = float('-inf')
                for action in self.mdp.getPossibleActions(p):
                    maxQ = max(maxQ, self.computeQValueFromValues(p, action) )
                #if diff > theta, push p into the priority queue with priority -diff 
                diff = abs(self.values[p] - maxQ)
                if diff > self.theta:
                    priorityQ.update(p, -diff)

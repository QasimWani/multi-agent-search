# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #the farther the ghost from pacman, the higher the score should be.
        #the closer the food from pacman, the higher the score should be.
        #evaluationFunction = current_score * ∑(distance_to_ghost/(total_distance_to_food^2)) + 0.1 (bias)

        dist_ghost = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        food_distance = sum([manhattanDistance(newPos, food) for food in newFood]) if newFood else 1e-1       
        food_ghost = sum([x/(food_distance**2) for x in dist_ghost]) + 1e-1

        return successorGameState.getScore() * food_ghost

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        _, action = self.max_value(gameState, 1, self.index) #run MAXIMIZER (set depth to 1 because `value` will give depth + 1)
        return action

    def value(self, state, depth, agentIndex):
        if(state.isWin() or state.isLose()): #check for terminal state
            #this fixes the bug of pacman being stuck
            return self.evaluationFunction(state), None

        next_agent = (agentIndex + 1) % state.getNumAgents() #cyclic index for next agent

        if(next_agent == self.index): #next_agent = pacman
            if(self.depth == depth):  #terminal state, if next agent == pacman and depth = game depth
                return self.evaluationFunction(state), None

            return self.max_value(state, depth + 1, next_agent) #run maximizer, every max node iterates depth by 1.

        return self.min_value(state, depth, next_agent) #min state, i.e. ghost(s), depth remains the same

    def max_value(self, state, depth, agentIndex):
        v, max_action = float('-inf'), None #represent smallest maximum value
        actions = state.getLegalActions(self.index) #get actions

        for action in actions:
            successor = state.generateSuccessor(agentIndex, action) #get next states
            next_value = self.value(successor, depth, agentIndex)[0]
            if (next_value > v): #store the maximum action
                max_action = action
            v = max(v, next_value)
        return v, max_action #should return maximum possible value

    def min_value(self, state, depth, agentIndex):
        v, minimum_action = float('inf'), None #represent largest smallest value
        actions = state.getLegalActions(agentIndex) #get action for a specific ghost

        for action in actions:
            successor = state.generateSuccessor(agentIndex, action) #get next states
            next_value = self.value(successor, depth, agentIndex)[0]
            if (v > next_value): #store the minimum action
                minimum_action = action
            v = min(v, next_value)
        return v, minimum_action #should return minimum possible value along with minimum action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        prune = float('-inf'), float('inf') #alpha-beta initialization
        
        #run MAXIMIZER (set depth to 1 because `value` will give depth + 1)
        _, action = self.max_value(gameState, 1, self.index, prune)
        return action


    def value(self, state, depth, agentIndex, prune):

        if(state.isWin() or state.isLose()): #check for terminal state
            #this fixes the bug of pacman being stuck
            return self.evaluationFunction(state), None
        
        next_agent = (agentIndex + 1) % state.getNumAgents() #cyclic index for next agent

        if(next_agent == self.index): #pacman is next agent?
            if self.depth == depth: #terminal state when next_agent = pacman and depth = game_depth
                return self.evaluationFunction(state), None #perform static evaluation

            return self.max_value(state, depth + 1, next_agent, prune) #use MAXIMIZER
        return self.min_value(state, depth, next_agent, prune) #use MINIMIZER

    def max_value(self, state, depth, agentIndex, prune):
        alpha, beta = prune
        v, max_action = float('-inf'), None #represent smallest maximum value
        actions = state.getLegalActions(self.index) #get actions

        for action in actions:
            successor = state.generateSuccessor(agentIndex, action) #get next states
            next_value = self.value(successor, depth, agentIndex, (alpha, beta))[0]
            if (next_value > v): #store the maximum action
                max_action = action

            v = max(v, next_value)

            if(v > beta):
                return v, max_action

            alpha = max(alpha, v)

        return v, max_action #should return maximum possible value

    def min_value(self, state, depth, agentIndex, prune):
        alpha, beta = prune
        v, minimum_action = float('inf'), None #represent largest smallest value
        actions = state.getLegalActions(agentIndex) #get action for a specific ghost

        for action in actions:
            successor = state.generateSuccessor(agentIndex, action) #get next states
            next_value = self.value(successor, depth, agentIndex, (alpha, beta))[0]
            if (v > next_value): #store the minimum action
                minimum_action = action
            
            v = min(v, next_value)
            
            if(v < alpha):
                return v, minimum_action
            
            beta = min(beta, v)
            
        return v, minimum_action #should return minimum possible value along with minimum action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        _, action = self.max_value(gameState, 1, 0)
        return action

    def value(self, state, depth, agentIndex):
        if(state.isWin() or state.isLose()): #check for terminal state
            #this fixes the bug of pacman being stuck
            return self.evaluationFunction(state), None

        next_agent = (agentIndex + 1) % state.getNumAgents() #cyclic index for next agent

        if(next_agent == self.index): #next_agent = pacman
            if(self.depth == depth):  #terminal state, if next agent == pacman and depth = game depth
                return self.evaluationFunction(state), None

            return self.max_value(state, depth + 1, next_agent) #run maximizer, every max node iterates depth by 1.

        return self.min_value(state, depth, next_agent) #min state, i.e. ghost(s), depth remains the same
    
    def max_value(self, state, depth, agentIndex):
        v, max_action = float('-inf'), None #represent smallest maximum value
        actions = state.getLegalActions(self.index) #get actions
        for action in actions:
            successor = state.generateSuccessor(agentIndex, action) #get next states
            next_value = self.value(successor, depth, agentIndex)[0]
            if (next_value > v): #store the maximum action
                max_action = action
            v = max(v, next_value)

        return v, max_action #should return maximum possible value

    def min_value(self, state, depth, agentIndex):
        v, minimum_action = float('inf'), None #represent largest smallest value
        actions = state.getLegalActions(agentIndex) #get action for a specific ghost
        uniform_expectation = 1/len(actions) #since expectimax in this project follows uniform dist, each probability is equal.
        expected_value = 0 #expected value 

        for action in actions:
            successor = state.generateSuccessor(agentIndex, action) #get next states
            next_value = self.value(successor, depth, agentIndex)[0]
            if (v > expected_value): #store the minimum action
                minimum_action = action

            expected_value +=  uniform_expectation * next_value #calculate the expectation

        return expected_value, minimum_action #should return minimum possible value along with minimum action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    * The evaluation function designed rewards the pacman agent inversely proportional to
      the square of manhattan distance to the nearest food and rewards the agent proportional
      to the manhattan distance to the nearest ghost.

    * This can be expressed by the following formula: 
        evaluationFunction = current_score * ∑(distance_to_ghost/(total_distance_to_food^2)) + 0.1 (bias)

    * In this, bias is chosen to prevent from 0 score which implies that the pacman is at a goal state which
    may not always be the case.

    * In layman terms, what this means is the following:
        - The farther the ghost from pacman, the higher the score.
        - The closer the food from pacman, the higher the score.
    
    * Reason for using squared proportion:
        This was used based on pure trial and error. After using different approaches,
        squared heuristic proved to be the most sensical because pacman should try his
        best to live than eat the nearest food. This is because the game doesn't end if
        pacman is unable to eat food, but ends if it comes under contact with ghost (without sudo mode).
        - This shows that the pacman should value proxmity to the ghost more than proximity to the nearest food-pellet.

    # Note: The logic of this code is reused from problem 1 of this assignment, because it's a powerful heuristic.

    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()

    #the farther the ghost from pacman, the higher the score should be.
    #the closer the food from pacman, the higher the score should be.
    #evaluationFunction = current_score * ∑(distance_to_ghost/(total_distance_to_food^2)) + 0.1 (bias)

    dist_ghost = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
    food_distance = sum([manhattanDistance(newPos, food) for food in newFood]) if newFood else 1e-1       
    food_ghost = sum([x/(food_distance**2) for x in dist_ghost]) + 1e-1

    return currentGameState.getScore() * food_ghost

# Abbreviation
better = betterEvaluationFunction

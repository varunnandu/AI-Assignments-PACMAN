# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        available_food = successorGameState.getNumFood()
        if  available_food > 0:
            distance_from_food = [manhattanDistance(newPos, food) for food in newFood.asList()]
            possible_ghost_position = []
            number_of_agents = successorGameState.getNumAgents()
            for i in range(1, number_of_agents):
                gx, gy = successorGameState.getGhostPosition(i)
                for x in range(-1, 2):
                    for y in range (-2, 2):
                        possible_ghost_position.append((gx - x, gy - y))
            if newPos in possible_ghost_position:
                return None
            else:
                return float(1) /float(available_food) + float(0.00001) / float(min(distance_from_food))

        return successorGameState.getScore()



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
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        node_action = actions[0]
        node_score = -float("inf")
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            score = self.minimum(next_state, 0, 1)
            if score > node_score:
                node_action = action
                node_score = score
        return node_action

    def minimum(self, gameState, depth, agent):

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        node_score = float("inf")
        actions = gameState.getLegalActions(agent)
        for action in actions:
            next_state = gameState.generateSuccessor(agent, action)
            if agent == (gameState.getNumAgents() - 1):
                score = self.maximum(next_state, depth + 1)
            else:
                score = self.minimum(next_state, depth, agent + 1)
            if score < node_score:
                node_score = score

        return node_score

    def maximum(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(0)
        node_score = -float("inf")
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            score = self.minimum(next_state, depth, 1)
            if score > node_score:
                node_score = score
        return node_score

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        node_action = actions[0]
        node_score = -float("inf")
        alpha = -float("inf")
        beta = float ("inf")
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            score = self.minimum(next_state, 0, 1, alpha, beta)
            if score > node_score:
                node_action = action
                node_score = score
            if alpha < node_score:
                alpha = node_score
        return node_action

    def minimum(self, gameState, depth, agent, alpha, beta):

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        node_score = float("inf")
        actions = gameState.getLegalActions(agent)
        for action in actions:
            next_state = gameState.generateSuccessor(agent, action)
            if agent == (gameState.getNumAgents() - 1):
                score = self.maximum(next_state, depth + 1, alpha, beta)
            else:
                score = self.minimum(next_state, depth, agent + 1, alpha, beta)
            if score < node_score:
                node_score = score
            if beta > node_score:
                beta = node_score
            if node_score < alpha:
                return node_score

        return node_score

    def maximum(self, gameState, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(0)
        node_score = -float("inf")
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            score = self.minimum(next_state, depth, 1, alpha, beta)
            if score > node_score:
                node_score = score
            if alpha < node_score:
                alpha = node_score
            if node_score > beta:
                return node_score
        return node_score


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
        actions = gameState.getLegalActions(0)
        node_action = actions[0]
        node_score = -float("inf")
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            score = self.minimum(next_state, 0, 1)
            if score > node_score:
                node_action = action
                node_score = score
        return node_action

    def minimum(self, gameState, depth, agent):

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        node_score = float("inf")
        scores = []
        actions = gameState.getLegalActions(agent)
        for action in actions:
            next_state = gameState.generateSuccessor(agent, action)
            if agent == (gameState.getNumAgents() - 1):
                score = self.maximum(next_state, depth + 1)
            else:
                score = self.minimum(next_state, depth, agent + 1)
            if score < node_score:
                node_score = score
            scores.append(score)
        average_score = sum(scores)/len(actions)

        return average_score

    def maximum(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(0)
        node_score = -float("inf")
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            score = self.minimum(next_state, depth, 1)
            if score > node_score:
                node_score = score
        return node_score

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <If the pacman is within 5 units of the ghosts location, it will runaway from the ghost
      using twice the speed and it will move towards power pallete with the highest priority. But if the pacman is more
      than 5 units away from the ghost, it will just move towrds power palletes with highest priority and food palettes with slightly
      lower priority>
    """
    "*** YOUR CODE HERE ***"
    food = currentGameState.getFood()
    food_position_list = food.asList()
    pacman = currentGameState.getPacmanPosition()
    score = scoreEvaluationFunction(currentGameState)
    capsules = currentGameState.getCapsules()
    agent = currentGameState.getNumAgents() - 1

    closest_food = float("inf")
    minimum_distance_to_ghost = float("inf")



    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return - float("inf")

    for p in food_position_list:
        distance = manhattanDistance(p, pacman)
        if distance < closest_food:
            closest_food = distance

    for i in range(1, agent+1):
        dist = manhattanDistance(pacman, currentGameState.getGhostPosition(i))
        minimum_distance_to_ghost = min(minimum_distance_to_ghost, dist)



    if minimum_distance_to_ghost < 5:
        score += minimum_distance_to_ghost * 2
        score -= 4 * len(capsules)


    else:
        score -= closest_food * 1.5
        score -= 4 * len(capsules)

        return score


    return score

# Abbreviation
better = betterEvaluationFunction


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
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    return max([(self.getQ(gameState, action), action) for action in gameState.getLegalActions(self.index)], key = lambda x: x[0])[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    def policy(player):
      return max if player == self.index else min

    def branch(state, player, depth):
      if player != state.getNumAgents()-1:
        return (player + 1, depth)
      else:
        return (0, depth-1)

    def V(state, player, d, action = None):
      if state.isWin() or state.isLose() or len(state.getLegalActions()) == 0:
        return state.getScore(), Directions.STOP
      elif d == 0:
        return self.evaluationFunction(state), Directions.STOP
      else:
        if action:
          return V(state.generateSuccessor(player, action),*branch(state, player, d))[0], action
        else:
          return policy(player)([(V(state.generateSuccessor(player, action), *branch(state, player, d))[0], action) for action in state.getLegalActions(player)], key = lambda x: x[0])
  
    return V(gameState, self.index, self.depth, action)[0]
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    return max([(self.getQ(gameState, action), action) for action in gameState.getLegalActions(self.index)], key = lambda x: x[0])[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER

    def policy(player):
      return max if player == self.index else min

    def branch(state, player, depth):
      if player != state.getNumAgents()-1:
        return (player + 1, depth)
      else:
        return (0, depth-1)

    def V(state, player, d, action = None):
      if state.isWin() or state.isLose() or len(state.getLegalActions()) == 0:
        return state.getScore(), Directions.STOP
      elif d == 0:
        return self.evaluationFunction(state), Directions.STOP
      else:
        if action:
          return V(state.generateSuccessor(player, action),*branch(state, player, d))[0], action
        else:
          if player == self.index:
            return max([(V(state.generateSuccessor(player, action), *branch(state, player, d))[0], action) for action in state.getLegalActions(player)], key = lambda x: x[0])
          else:
            actions = state.getLegalActions(player)
            prob = 1/len(actions)
            return sum([prob * V(state.generateSuccessor(player, action), *branch(state, player, d))[0] for action in actions]), random.choice(actions)
  
    return V(gameState, self.index, self.depth, action)[0]

    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    return max([(self.getQ(gameState, action), action) for action in gameState.getLegalActions(self.index)], key = lambda x: x[0])[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def branch(state, player, depth):
      if player != state.getNumAgents()-1:
        return (player + 1, depth)
      else:
        return (0, depth-1)

    def V(state, player, d, action = None):
      if state.isWin() or state.isLose() or len(state.getLegalActions()) == 0:
        return state.getScore(), Directions.STOP
      elif d == 0:
        return self.evaluationFunction(state), Directions.STOP
      else:
        if action:
          return V(state.generateSuccessor(player, action),*branch(state, player, d))[0], action
        else:
          if player == self.index:
            return max([(V(state.generateSuccessor(player, action), *branch(state, player, d))[0], action) for action in state.getLegalActions(player)], key = lambda x: x[0])
          else:
            actions = state.getLegalActions(player)
            prob = 1/len(actions)*0.5
            return sum([(prob + (0.5 if action == Directions.STOP else 0)) * V(state.generateSuccessor(player, action), *branch(state, player, d))[0] for action in actions]), random.choice(actions)
  
    return V(gameState, self.index, self.depth, action)[0]
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    return max([(self.getQ(gameState, action), action) for action in gameState.getLegalActions(self.index)], key = lambda x: x[0])[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def branch(state, player, depth):
      if player != state.getNumAgents()-1:
        return (player + 1, depth)
      else:
        return (0, depth-1)

    def V(state, player, d, action = None):
      if state.isWin() or state.isLose() or len(state.getLegalActions()) == 0:
        return state.getScore(), Directions.STOP
      elif d == 0:
        return self.evaluationFunction(state), Directions.STOP
      else:
        if action:
          return V(state.generateSuccessor(player, action),*branch(state, player, d))[0], action
        else:
          if player == self.index:
            return max([(V(state.generateSuccessor(player, action), *branch(state, player, d))[0], action) for action in state.getLegalActions(player)], key = lambda x: x[0])
          elif player%2 == 1:
            return min([(V(state.generateSuccessor(player, action), *branch(state, player, d))[0], action) for action in state.getLegalActions(player)], key = lambda x: x[0])
          else:
            actions = state.getLegalActions(player)
            prob = 1/len(actions)
            return sum([prob * V(state.generateSuccessor(player, action), *branch(state, player, d))[0] for action in actions]), random.choice(actions)
  
    return V(gameState, self.index, self.depth, action)[0]
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    return max([(self.getQ(gameState, action), action) for action in gameState.getLegalActions(self.index)], key = lambda x: x[0])[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def branch(state, player, depth):
      if player != state.getNumAgents()-1:
        return (player + 1, depth)
      else:
        return (0, depth-1)

    def V(state, player, d, alpha, beta, action = None):
      if state.isWin() or state.isLose() or len(state.getLegalActions()) == 0:
        return state.getScore(), Directions.STOP
      elif d == 0:
        return self.evaluationFunction(state), Directions.STOP
      else:
        if action:
          return V(state.generateSuccessor(player, action),*branch(state, player, d), alpha, beta)[0], action
        else:
          if player == self.index:
            v_hold = float("-inf"), Directions.STOP
            for action in state.getLegalActions(player):
              v_hold = max(v_hold, V(state.generateSuccessor(player, action), *branch(state, player, d), alpha, beta), key = lambda x: x[0])
              alpha = max(alpha, v_hold[0])
              if alpha >= beta:
                break
            return v_hold
          elif player%2 == 1:
            v_hold = float("inf"), Directions.STOP
            for action in state.getLegalActions(player):
              v_hold = min(v_hold, V(state.generateSuccessor(player, action), *branch(state, player, d), alpha, beta), key = lambda x: x[0])
              beta = min(alpha, v_hold[0])
              if alpha >= beta:
                break
            return v_hold
          else:
            actions = state.getLegalActions(player)
            prob = 1/len(actions)
            return sum([prob * V(state.generateSuccessor(player, action), *branch(state, player, d), alpha, beta)[0] for action in actions]), random.choice(actions)
    return V(gameState, self.index, self.depth, float("-inf"), float("inf"), action)[0]
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER
  weight = []
  feature = []

  score_current_state = currentGameState.getScore()
  pacman_pos          = currentGameState.getPacmanPosition()
  food_n              = currentGameState.getNumFood()
  food_pos_list       = currentGameState.getFood().asList()
  capsule_pos_list    = currentGameState.getCapsules()
  ghost_state_list    = currentGameState.getGhostStates()

  dist_food = []
  for food_pos in food_pos_list:
    dist_food.append(manhattanDistance(food_pos, pacman_pos))
  
  dist_capsule = []
  for capsule_pos in capsule_pos_list:
    dist_capsule.append(manhattanDistance(capsule_pos, pacman_pos))

  dist_unscared_ghost = []
  dist_scared_ghost = []
  for ghost in ghost_state_list:
    ghost_pos = ghost.getPosition()
    if ghost.scaredTimer > 0:
      dist_scared_ghost.append(manhattanDistance(ghost_pos, pacman_pos)/ghost.scaredTimer)
    else:
      dist_unscared_ghost.append(manhattanDistance(ghost_pos, pacman_pos))

  ##
  weight.append(1)
  feature.append(score_current_state)

  ##
  weight.append(-0.1)
  feature.append(food_n)
  
  ##
  weight.append(3/food_n)
  feature.append(1/min(dist_food) if len(dist_food)>0 else 0)
  
  ##
  weight.append(5.5 if len(dist_scared_ghost)==0 else 0)
  feature.append(1/min(dist_capsule) if len(dist_capsule)>0 else 0)

  ##
  weight.append(15)
  feature.append(1/min(dist_scared_ghost) if len(dist_scared_ghost)>0 else 0)
  weight.append(1.1)
  feature.append(min(dist_unscared_ghost) if len(dist_unscared_ghost)>0 else 0)

  return sum([w*v for w,v in zip(weight,feature)])
  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  return 'MinimaxAgent'
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction

import util, math, random
from collections import defaultdict
from util import ValueIteration


############################################################
# Problem 2a: BlackjackMDP


class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        super().__init__()

        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_ANSWER (our solution is 44 lines of code, but don't worry if you deviate from this)
        def update(deck, idx, total):
            deck_ = list(deck)
            deck_[idx] -= 1
            return total + self.cardValues[idx], (tuple(deck_) if sum(deck_) else None)
        p = list()
        total_in_hand, is_peek, deck = state
        if deck is None:
            pass
        else:
            deck_sum = sum(deck)
            if action == "Quit" or deck_sum == 0:
                p.append(((total_in_hand, None, None), 1.0, total_in_hand))
            elif action == "Take":
                if is_peek:
                    total_update, deck_ = update(deck, is_peek, total_in_hand)
                    if total_update > self.threshold:
                        p.append(((total_update, None, None), 1, 0))
                    else:
                        p.append(((total_update, None, deck_), 1, (0 if deck_ else total_update)))
                else:
                    for idx, cnt in enumerate(deck):
                        if cnt>0:
                            total_update, deck_ = update(deck, idx, total_in_hand)
                            prob = cnt / deck_sum
                            if total_update > self.threshold:
                                p.append(((total_update, None, None), prob, 0))
                            else:
                                p.append(((total_update, None, deck_), prob, (0 if deck_ else total_update)))
            elif action == "Peek":
                if is_peek:
                    pass
                else:
                    for idx, cnt in enumerate(deck):
                        if cnt>0:
                            p.append(((total_in_hand, idx, deck), cnt/deck_sum, -self.peekCost))
            elif action == "Quit":
                p.append(((total_in_hand, None, None), 1.0, total_in_hand))
        return p
        # END_YOUR_ANSWER

    def discount(self):
        return 1


############################################################
# Problem 3a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class Qlearning(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with episode=[..., state, action,
    # reward, newState], which you should use to update
    # |self.weights|. You should update |self.weights| using
    # self.getStepSize(); use self.getQ() to compute the current
    # estimate of the parameters. Also, you should assume that
    # V_opt(newState)=0 when isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        state, action, reward, newState = episode[-4:]

        if isLast(state):
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        coef = self.getStepSize() * (self.getQ(state, action) - (reward + self.discount * max([self.getQ(newState,action_) for action_ in self.actions(newState)])))
        approx = self.featureExtractor(state, action)
        for k,v in approx:
            self.weights[k] -= coef*v
        # END_YOUR_ANSWER


############################################################
# Problem 3b: Q SARSA

class SARSA(Qlearning):
    # We will call this function with episode=[..., state, action,
    # reward, newState, newAction, newReward, newNewState], which you
    # should use to update |self.weights|. You should
    # update |self.weights| using self.getStepSize(); use self.getQ()
    # to compute the current estimate of the parameters. Also, you
    # should assume that Q_pi(newState, newAction)=0 when when
    # isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        assert (len(episode) - 1) % 3 == 0
        if len(episode) >= 7:
            state, action, reward, newState, newAction = episode[-7: -2]
        else:
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        coef = self.getStepSize() * (self.getQ(state, action) - (reward + self.discount * self.getQ(newState, newAction)))
        approx = self.featureExtractor(state, action)
        for k,v in approx:
            self.weights[k] -= coef*v
        # END_YOUR_ANSWER

# Return a singleton list containing indicator feature (if exist featurevalue = 1)
# for the (state, action) pair.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 3c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs
# (see identityFeatureExtractor() above for an example).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card type and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card type is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).
#       Example: if the deck is (3, 4, 0, 2), you should have four features (one for each card type).
#       And the first feature key will be (0, 3, action)
#       Only add these features if the deck != None

def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
    features = [((total, action), 1)]
    if counts:
        t = tuple(int(bool(v)) for v in counts)
        features.append(((t, action), 1))
        features += [((idx, cnt, action), 1) for idx, cnt in enumerate(counts)]
    return features
    # END_YOUR_ANSWER

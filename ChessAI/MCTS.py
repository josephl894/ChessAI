import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Vs = {}

        self.Qsa = {}
        self.Nsa = {}
        self.Ps = {}
        self.Ns = {}

        # this is the only member variable you'll have to use. It'll be used in select()
        self.visited = set()  # all "state" positions we have seen so far

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def gameEnded(self, canonicalBoard):
        """
        This function determines if the current board position is the end of the game.

        Returns:
            gameReward: a value that returns 0 if the game hasn't ended, 1 if the player won, -1 if the player lost
        """

        gameReward = self.game.getGameEnded(canonicalBoard, 1)
        return gameReward

    def predict(self, state, canonicalBoard):
        """
        A wrapper to perform predictions and necessary policy masking for the code to work.
        The key idea is to call this function to return an initial policy vector and value from the neural network
        instead of needing a rollout

        Returns:
            r: the reward given by the neural network
        """
        self.Ps[state], val = self.nnet.predict(canonicalBoard)
        valids = self.game.getValidMoves(canonicalBoard, 1)
        self.Ps[state] = self.Ps[state] * valids
        sum_Ps_s = np.sum(self.Ps[state])
        if sum_Ps_s > 0:
            self.Ps[state] /= sum_Ps_s
        else:
            log.error("All valid moves were masked, doing a workaround.")
            self.Ps[state] = self.Ps[state] + valids
            self.Ps[state] /= np.sum(self.Ps[state])

        self.Vs[state] = valids
        self.Ns[state] = 0
        return val

    def getValidActions(self, state):
        """
        Generates the valid actions from the avialable actions. Actions are given as a list of integers.
        The integers represent which spot in the board to place an Othello disc. 
        To see a (x, y) representation of an action, you can do "x, y = (int(action/self.game.n), action%self.game.n)"

        Returns:
            validActions: all valid actions you can take in terms of a list of integers
        """

        validActions = []
        for action in range(self.game.getActionSize()):
            if self.Vs[state][action]:
                validActions.append(action)
        return validActions

    def nextState(self, canonicalBoard, action):
        """
        Gets the next board state given the action

        Returns:
            nextBoard: the next board state given the action
        """

        nextState, nextPlayer = self.game.getNextState(
            canonicalBoard, 1, action)
        nextState = self.game.getCanonicalForm(nextState, nextPlayer)
        return nextState

    def getConfidenceVal(self, state, action):
        if (state, action) not in self.Qsa:
            self.Qsa[(state, action)] = 0
            self.Nsa[(state, action)] = 0

        u = self.Qsa[(state, action)] + self.args.cpuct * self.Ps[state][action] * math.sqrt(self.Ns[state]) / (
            1 + self.Nsa[(state, action)])

        return u

    def updateValues(self, r, state, action):
        self.Qsa[(state, action)] = (self.Nsa[(state, action)] *
                                     self.Qsa[(state, action)] + r) / (self.Nsa[(state, action)] + 1)
        self.Nsa[(state, action)] += 1
        self.Ns[state] += 1

    def expand(self, state):
        self.visited.add(state)

    def select(self, state, board):
        # TODO: your implementation goes here
        r = self.gameEnded(board)
        if r is not 0:
            return None, None, None, -r
        elif state not in self.visited:
            self.expand(state)
            r = self.simulate(state, board)
            return None, None, None, -r
        u = float('-inf')
        action = None
        for nextAction in self.getValidActions(state):
            next_u = self.getConfidenceVal(state, nextAction)
            if next_u > u:
                u = next_u
                action = nextAction

        # End implementation
        board = self.nextState(board, action)
        state = self.game.stringRepresentation(board)
        return state, board, action, None

    def backpropagate(self, seq):
        # TODO: your implementation goes here
        r = 0
        for s, action, nextReward in reversed(seq):
            if nextReward is not None:
                r = nextReward
            else:
                self.updateValues(r, s, action)
                r = -r

    def simulate(self, state, board):
        # TODO: your implementation goes here
        r = self.predict(state, board)
        return r

    def search(self, initial_board):
        """
        This function performs MCTS. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return a
        reward r for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the reward of the current
        state. This is done since r is in [-1,1] and if r is the value of a
        state for the current player, then its value is -r for the other player.

        Returns:
            b0: the initial board state of the othello board
        """
        initial_state = self.game.stringRepresentation(initial_board)

        for _ in range(self.args.numMCTSSims):
            # TODO: your implementation goes here
            state = initial_state
            board = initial_board
            seq = []
            reward = None

            while reward is None:
                nextState, nextBoard, action, nextReward = self.select(
                    state, board)
                tup = (state, action, nextReward)
                seq.append(tup)
                state = nextState
                board = nextBoard
                reward = nextReward
            self.backpropagate(seq)
        return initial_board

import numpy as np
import random
import sys


class State:
    """The State class represents a grid state at a certain position on the gameboard.
    """

    def __init__(self, position, type, reward, is_terminal_state):
        self.position = position
        self.type = type
        self.reward = reward
        self.terminal = is_terminal_state

    def get_actions(self):
        """Get the possible actions you can take at a certain State.
        This is the same for every state as you can always attempt to move in any direction.
        """
        return ["north", "south", "east", "west"]


class Grid:
    """Represents the gameboard as a grid, where each grid position corresponds to a State class object. """

    def __init__(self, rows, cols, states, start_pos, empty_reward):
        self.num_rows = rows
        self.num_cols = cols
        self.states = states
        self.start = start_pos
        self.empty_reward = empty_reward
        self.visited = np.zeros(states.shape)

    def visit(self, state):
        """ Marks a state as visited (to keep track of consumed vs unconsumed dots)"""
        position = state.position
        self.visited[position] = 1

    def reset(self):
        """Resets grid information specific to a single episode.
        In this case, marks all states as not visited.
        """
        self.visited = np.zeros(self.visited.shape)

    def get_reward(self, state):
        """ Returns the reward for a given state. Also marks the state as visited if it is a dot state.
        """
        if (state.type != 'd'):
            return state.reward
        elif (self.visited[state.position] == 0):
            self.visit(state)  # mark dot as visited for future purposes
            return state.reward
        else:
            return self.empty_reward  # dot already consumed


class RLAgent:
    """The Pacman agent that you will be working on. Contains a grid class object that it will use for training.
    """

    def __init__(self, grid, discount_factor, learning_rate, episodes):
        self.grid = grid
        self.df = discount_factor
        self.lr = learning_rate
        self.eps = episodes
        # TODO
        # Initialize relevant variables.
        # Current state should be a state class object, representing the agent's current state.
        # The q and n dictionaries should map from a tuple of (state_position, action) = ((row, col), action) to a q-value or n-count respectively.
        self.current_state = grid.states[grid.start[0]][grid.start[1]]
        self.q = {}
        self.n = {}
        for row in range(self.grid.num_rows):
            for col in range(self.grid.num_cols):
                self.q[((row, col), 'north')] = 0
                self.q[((row, col), 'south')] = 0
                self.q[((row, col), 'west')] = 0
                self.q[((row, col), 'east')] = 0
                self.n[((row, col), 'north')] = 0
                self.n[((row, col), 'south')] = 0
                self.n[((row, col), 'west')] = 0
                self.n[((row, col), 'east')] = 0
        # END TODO

    def execute_action(self, action):
        """ Attempts to execute an action (from its current state), returns new state and resulting reward.
        Note: This does not update the current state.

        :param action: A string corresponding to a valid action.
        :return: a tuple containing the new state and resulting reward in that order (new_state, new_reward).
        """

        rand = random.random()
        row_dir = 0
        col_dir = 0
        if (action == "north"):
            row_dir = -1
        elif (action == "south"):
            row_dir = 1
        elif (action == "east"):
            col_dir = 1
        else:
            col_dir = -1

        if rand < 0.9:
            # action is successful
            new_pos = (
                self.current_state.position[0] + row_dir, self.current_state.position[1] + col_dir)
            if (new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= self.grid.num_rows or new_pos[1] >= self.grid.num_cols):
                # if action hits boundary walls
                return self.current_state, self.grid.get_reward(self.current_state)
            else:
                new_state = self.grid.states[new_pos]

            if (new_state.type == "w"):
                # if action hits a wall
                return self.current_state, self.grid.get_reward(self.current_state)
            else:
                return new_state, self.grid.get_reward(new_state)
        else:
            # action is unsuccessful -- end up going in opposite direction
            row_dir = -1 * row_dir
            col_dir = -1 * col_dir
            new_pos = (
                self.current_state.position[0] + row_dir, self.current_state.position[1] + col_dir)
            if (new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= self.grid.num_rows or new_pos[1] >= self.grid.num_cols):
                # if action hits boundary walls
                return self.current_state, self.grid.get_reward(self.current_state)
            else:
                new_state = self.grid.states[new_pos]

            if (new_state.type == "w"):
                # if action hits a wall
                return self.current_state, self.grid.get_reward(self.current_state)
            else:
                return new_state, self.grid.get_reward(new_state)

    def select_action(self, state):
        """Selects an action to perform for the given state based on the current q-values.
        Uses epsilon-greedy action selection as described below.

        :return: a string corresponding to the selected action.
        """

        actions = state.get_actions()
        max_q = float('-inf')
        best_action = None
        for action in actions:
            if (self.q[(state.position, action)] > max_q):
                max_q = self.q[(state.position, action)]
                best_action = action

        # in train time:
        # use epsilon-greedy action selection
        # chooses best action with probability 1-epsilon
        # otherwise, explores a random action (with probability epsilon)
        epsilon = 0.3
        rand = random.random()
        if rand < epsilon:
            rand_a = random.randint(0, len(actions)-1)
            return actions[rand_a]
        else:
            return best_action

    def q_learning(self):
        """ Perform the q_learning RL algorithm, as described in lecture.
        Use the hyperparameters stored in this RLAgent class instance.

        Make sure to utilize self.current_state to keep track of your current state as some of the given functions utilize this value.

        :return: the dictionary of q-values (q-table) after training has finished.

        You need to finish implementing this function.
        """

        for i in range(self.eps):
            # this resets the grid (i.e. resets consumed dots to be unconsumed).
            self.grid.reset()
            # Note that this does not change any other member variables inside RLAgent which may or may not also need to be reset.
            # TODO
            self.current_state = self.grid.states[self.grid.start[0]][self.grid.start[1]]
            while self.current_state.terminal is not True:
                a = self.select_action(self.current_state)
                new_s, reward = self.execute_action(a)
                max_val = max(self.q[(new_s.position, 'north')], self.q[(new_s.position, 'south')],
                              self.q[(new_s.position, 'west')], self.q[(new_s.position, 'east')])
                self.n[(self.current_state.position, a)] += 1
                self.q[(self.current_state.position, a)] += (self.lr * (reward +
                                                       (self.df * max_val) - self.q[(self.current_state.position, a)]))
                self.current_state = new_s
            # END TODO
        return self.q

    def SARSA(self):
        """ Perform the SARSA RL algorithm, as described in lecture.
        Use the hyperparameters stored in this RLAgent class instance.
        :return: the dictionary of q-values (q-table) after training has finished.

        You need to finish implementing this function.
        """

        for i in range(self.eps):
            self.grid.reset()
            # TODO
            self.current_state = self.grid.states[self.grid.start[0]][self.grid.start[1]]
            a = self.select_action(self.current_state)
            while self.current_state.terminal is not True:
                new_s, reward = self.execute_action(a)
                new_a = self.select_action(new_s)
                self.n[(self.current_state.position, a)] += 1
                self.q[(self.current_state.position, a)] += (self.lr * (reward +
                                                       (self.df * self.q[(new_s.position, new_a)]) - self.q[(self.current_state.position, a)]))
                self.current_state = new_s
                a = new_a
            # END TODO
        return self.q

    def get_str_rep(self):
        """Get string representation of current q-values and n-counts."""
        output = ""
        for i in range(self.grid.num_rows):
            for j in range(self.grid.num_cols):
                for a in ["north", "south", "east", "west"]:
                    if (self.grid.states[i, j].type != "w" and not self.grid.states[i, j].terminal):
                        output = output + "Q((%i, %i), %s) = %f" % (i, j, a, self.q[(
                            (i, j), a)]) + "\t" + "N((%i, %i), %s) = %i" % (i, j, a, self.n[((i, j), a)]) + "\n"
        return output

    def print_results(self):
        """Prints the q-value and n-count dictionaries for the relevant states."""
        str_rep = self.get_str_rep()
        print(str_rep)


def parse_file(filename):
    """Parses the given text file into a RLAgent Class object."""
    f = open(filename, 'r')
    line = f.readline()
    hyperparameters = line.split()
    discount_factor = float(hyperparameters[0])
    learning_rate = float(hyperparameters[1])
    num_episodes = int(hyperparameters[2])
    line = f.readline()
    size = line.split()
    rows = int(size[0])
    cols = int(size[1])
    line = f.readline()
    rewards = line.split()
    empty_reward = float(rewards[0])
    dot_reward = float(rewards[1])
    states = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        line = f.readline()
        entries = line.split()
        for j in range(cols):
            state_type = entries[j]
            if (state_type == "." or state_type == 'w'):
                states[i, j] = State((i, j), state_type, empty_reward, False)
            elif (state_type == 's'):
                states[i, j] = State((i, j), state_type, empty_reward, False)
                start_loc = (i, j)
            elif (state_type == "d"):
                states[i, j] = State((i, j), state_type, dot_reward, False)
            elif (state_type == "g"):
                states[i, j] = State((i, j), state_type, -1, True)
            else:
                states[i, j] = State((i, j), state_type, 1, True)
    g = Grid(rows, cols, states, start_loc, empty_reward)
    agent = RLAgent(g, discount_factor, learning_rate, num_episodes)
    return agent


if __name__ == "__main__":
    random.seed(492)
    file = sys.argv[1]
    agent = parse_file(file)
    # agent.q_learning()  # switch to agent.SARSA() if you want to test SARSA instead
    agent.SARSA()
    agent.print_results()

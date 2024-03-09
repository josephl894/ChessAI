import numpy as np
import random 
import sys
import activeRL
import time

def visualize_grid(agent):
    output = ""
    for row in range(agent.grid.num_rows):
        for col in range(agent.grid.num_cols):
            if((row, col) == agent.current_state.position):
                output = output + "p "
            else:
                output = output + agent.grid.states[row, col].type + " "
        output = output + "\n"
    print(output)

def visualize_run(agent):
    trained_q = agent.q
    agent.grid.reset() 
    agent.current_state = agent.grid.states[agent.grid.start]
    visualize_grid(agent)
    time.sleep(3)
    while(True):
        best_q = float('-inf')
        best_action = ""
        for action in ["north", "south", "east", "west"]:
            if(agent.q[(agent.current_state.position, action)] > best_q):
                best_q = agent.q[(agent.current_state.position, action)]
                best_action = action

        new_state, new_reward = agent.execute_action(best_action)
        print(new_state.position)
        print(new_reward)
        agent.current_state = new_state

        visualize_grid(agent)
        time.sleep(3)

        if new_state.terminal:
            break


if __name__ == "__main__":
    random.seed(492)
    file = sys.argv[1]
    agent = activeRL.parse_file(file)
    #agent.q_learning() ### switch to agent.SARSA() if you want to visualize SARSA instead
    agent.SARSA()
    visualize_run(agent)
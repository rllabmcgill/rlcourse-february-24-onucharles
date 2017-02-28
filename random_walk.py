import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import pickle

N_STATES = 5

WALK_LEFT = 0
WALK_RIGHT = 1


states = np.arange(N_STATES)
actual_state_values = (states + 1) / (N_STATES + 1)

def walk(state, action):
    nextState = -1
    isEnd = False
    curReward = 0

    if WALK_LEFT == action:
        nextState = state - 1
        if nextState == -1:
            isEnd = True
    elif WALK_RIGHT == action:
        nextState = state + 1
        if nextState == N_STATES:
            isEnd = True
            curReward = 1
            nextState = -1

    return nextState, isEnd, curReward

def generateEpisode():
    episode = []

    start_state = np.floor(N_STATES / 2)
    curState = start_state
    while True:
        action = np.random.binomial(1, 0.5)
        nextState, isEnd, curReward = walk(curState, action)
        episode.append((int(curState), curReward, int(nextState)))
        if isEnd == True:
            break
        curState = nextState

    return episode

#monte carlo prediction
def mcPredictEveryVisit(alpha, n_episodes, n_iteration):

    rmse = np.zeros((n_episodes, n_iteration))

    for k in range(n_iteration):
        state_values = np.zeros((N_STATES)) + 0.5
        for j in range(n_episodes):
            episode = generateEpisode()

            #this is every-visit MC
            (state, lastReward, nextState) = episode[-1]
            for (state, reward, nextState) in episode:
                state_values[state] += alpha * (lastReward - state_values[state])

            rmse[j, k] = sqrt(mean_squared_error(state_values, actual_state_values))

    return np.mean(rmse, axis=1) #, np.mean(state_values, axis=1)

def mcPredictFirstVisit(alpha, n_episodes, n_iteration):

    rmse = np.zeros((n_episodes, n_iteration))

    for k in range(n_iteration):
        state_values = np.zeros((N_STATES)) + 0.5
        for j in range(n_episodes):
            episode = generateEpisode()

            #this is first visit MC
            cum_returns = np.zeros(N_STATES)
            is_state_visited = np.zeros(N_STATES, dtype=bool)   #takes a value of 0 or 1 in each cell
            for i in np.arange(len(episode)):
                (state, reward, nextState) = episode[i]
                is_state_visited[state] = 1
                cum_returns[is_state_visited] += reward

            #state_values[:,k] = state_values[:,k] + alpha * (cum_returns - state_values[:,k])
            state_values = state_values + alpha * (cum_returns - state_values)
            rmse[j, k] = sqrt(mean_squared_error(state_values, actual_state_values))

    return np.mean(rmse, axis=1) #, np.mean(state_values, axis=1)


#td(0) for prediction
def tdPredict(alpha, n_episodes, n_iteration):

    rmse = np.zeros((n_episodes, n_iteration))

    for k in range(n_iteration):
        state_values = np.zeros(N_STATES) + 0.5
        for i in range(n_episodes):
            episode = generateEpisode()
            for j in range(len(episode)):
                (state, reward, nextState) = episode[j]
                if nextState == -1:
                    state_values[state] = state_values[state] + alpha * (reward - state_values[state])
                    break
                state_values[state] = state_values[state] + alpha * (reward + 1 * state_values[nextState] - state_values[state])

            rmse[i, k] = sqrt(mean_squared_error(state_values, actual_state_values))
    return np.mean(rmse, axis=1)

def run_experiments():
    n_episodes = 100
    n_iterations = 100

    #1. Everyvisit vs first visit monte carlo
    alpha = 0.01
    mc_ev_rmse = mcPredictEveryVisit(alpha, n_episodes, n_iterations)
    mc_fv_rmse = mcPredictFirstVisit(alpha, n_episodes, n_iterations)
    plt.plot(np.arange(n_episodes), mc_ev_rmse, '-', label='MC - Every visit')
    plt.plot(np.arange(n_episodes), mc_fv_rmse, '-', label='MC - First visit')
    plt.ylabel('root mean square error')
    plt.xlabel('episodes')
    plt.legend(loc='upper right')
    plt.show()

    #2. monte carlo: varying alpha
    alphas = [0.01, 0.02, 0.04, 0.08]
    for alpha in alphas:
        mc_rmse = mcPredictEveryVisit(alpha, n_episodes, n_iterations)
        plt.plot(np.arange(n_episodes), mc_rmse, '-', label='MC-' + str(alpha))
    plt.ylabel('root mean square error')
    plt.xlabel('episodes')
    plt.legend(loc='upper right')
    plt.show()

    #3. td(0): varying alpha
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
    for alpha in alphas:
        td_rmse = tdPredict(alpha, n_episodes, n_iterations)
        plt.plot(np.arange(n_episodes), td_rmse, '-', label='TD-' + str(alpha))
    plt.ylabel('root mean square error')
    plt.xlabel('episodes')
    plt.legend(loc='upper right')
    plt.show()

    #4. td(0) vs monte carlo
    alphas = [0.05, 0.1]
    for alpha in alphas:
        td_rmse = tdPredict(alpha, n_episodes, n_iterations)
        plt.plot(np.arange(n_episodes), td_rmse, '-', label='TD-' + str(alpha))

    alphas = [0.01, 0.02]
    for alpha in alphas:
        mc_rmse = mcPredictEveryVisit(alpha, n_episodes, n_iterations)
        plt.plot(np.arange(n_episodes), mc_rmse, '--', label='MC-' + str(alpha))

    plt.ylabel('root mean square error')
    plt.xlabel('episodes')
    plt.legend(loc='upper right')
    plt.show()

run_experiments()
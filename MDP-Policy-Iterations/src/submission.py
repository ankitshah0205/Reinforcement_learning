### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

np.set_printoptions(precision=3)

def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = None
    ############################
    ### START CODE HERE ###
    #Bellman backup is value of expected immediate reward plus the discounted future value of next state 
    # let's calculate the expected immediate reward first 
    immediate_reward_for_bellman_backup = R[state, action]
    # Now, let's calculate the expected future discounted value of next state using numpy sum function 
    # Let's use numpy recursive function.
    # Gamma is discount factor
    discounted_expected_future_value = np.sum(T[state, action, :]*V)
    backup_val = immediate_reward_for_bellman_backup+gamma*discounted_expected_future_value

    ### END CODE HERE ###
    ############################

    return backup_val

def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, _ = R.shape
    value_function = np.zeros(num_states)
    ############################
    ### START CODE HERE ###
    #Policy evluation refers to computing value function for given policy.
    #We will start with initial value of value function usijg numpy zeroes function
    #Then we will repeatedly apply bellman backup for given policy
    #And finally, we will repeat until values converge
    while True:
        convergence_delta = 0
        new_value_function = np.zeros(num_states)
        
        for intermediate_state in range(num_states):
            a = policy[intermediate_state]
            # Bellman expectation backup under the given policy
            new_value_function[intermediate_state] = R[intermediate_state, a] + gamma * np.sum(T[intermediate_state, a, :] * value_function)
        
        # Now check the convergence difference
        convergence_delta = np.max(np.abs(new_value_function - value_function))
        value_function = new_value_function
        
        if convergence_delta < tol:
            break
    ### END CODE HERE ###
    ############################
    return value_function


def policy_improvement(R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = np.zeros(num_states, dtype=int)
    ############################
    ### START CODE HERE ###
    # policy improvement is the process of making policy better using it's value function 
    # we will first look at all possible actions in each state
    # then we will choose the action with the highest expected return according to value function
    for intermediate_state in range(num_states):
        q_values = np.zeros(num_actions)
        for intermediate_action in range(num_actions):
            q_values[intermediate_action] = R[intermediate_state, intermediate_action] + gamma * np.sum(T[intermediate_state, intermediate_action, :] * V_policy)
        new_policy[intermediate_state] = np.argmax(q_values)
    ### END CODE HERE ###
    ############################
    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, _ = R.shape
    V_policy = None

    # Policy iteration is to alternate between policy evluation and policy improvement until policy keeps changing.
    # When policy stops changing, that means that we have found optimal policy.
    # Hence, we will keep calling policy_evluation and policy_improvement functions above until policy changes.
    policy = np.zeros(num_states, dtype=int)
    while True:
        # Lets evluate policy using the function we implemented earlier
        V_policy = policy_evaluation(policy, R, T, gamma, tol)

        # Let's improve the policy using the function we implemented earlier
        new_intermediate_policy = policy_improvement(R, T, V_policy, gamma)

        # Check if policy value is changing or not
        if np.array_equal(new_intermediate_policy, policy):
            break

        policy = new_intermediate_policy

    ### END CODE HERE ###
    ############################
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    ### START CODE HERE ###
    # value iteration skips full evaluation, it repeatedly applies optimality backups instead of fully evaluating policy. 
    # Value Iteration is often faster for large state spaces since it avoids solving full systems of equations.
    while True:
        # Hold the values for this iteration in this array
        new_value_function = np.zeros(num_states)

        # First lets apply Bellman optimality backup repeatedly
        for intermediate_state in range(num_states):
            # Store value of each possible action for this intermediate state 
            q_values = np.zeros(num_actions)
            for intermediate_action in range(num_actions):
                # Implement Q-value = immediate reward + (discounted factor*future reward) 
                q_values[intermediate_action] = R[intermediate_state, intermediate_action] + gamma * np.sum(T[intermediate_state, intermediate_action, :] * value_function)
            new_value_function[intermediate_state] = np.max(q_values)

        # Now, lets do convergence check
        difference = np.max(np.abs(new_value_function - value_function))
        value_function = new_value_function

        #check if we have stopped converging
        if difference < tol:
            break

    # Now extract greedy policy from final value function above
    for intermediate_state in range(num_states):
        #Store the values for this iteration in this array 
        q_values = np.zeros(num_actions)
        for intermediate_action in range(num_actions):
            #Implement Q-Value = immedite_reqard + (discounted factor*future reard) 
            q_values[intermediate_action] = R[intermediate_state, intermediate_action] + gamma * np.sum(T[intermediate_state, intermediate_action, :] * value_function)
        #Now, we select action that gives highest return
        policy[intermediate_state] = np.argmax(q_values)
    ### END CODE HERE ###
    ############################
    return value_function, policy


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = 'WEAK'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.99

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([['L', 'R'][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([['L', 'R'][a] for a in policy_vi])

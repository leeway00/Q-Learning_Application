"""
Simple MDP Environment for Testing Q-Learning Agent
Simulates a medical decision-making scenario (simplified)
"""

import numpy as np


class SimpleMedicalMDP:
    """
    Simplified medical decision-making environment
    
    This is a simple gridworld-style MDP that simulates medical interventions.
    States represent patient conditions, actions represent treatments.
    """
    
    def __init__(self, n_states=50, n_actions=5, seed=None):
        """
        Initialize the MDP environment
        
        Args:
            n_states: Number of discrete states
            n_actions: Number of discrete actions (treatments)
            seed: Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.current_state = 0
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create transition probabilities and rewards
        # P[s, a, s'] = probability of transitioning from s to s' with action a
        self.P = np.random.dirichlet(np.ones(n_states), size=(n_states, n_actions))
        
        # R[s, a] = expected reward for taking action a in state s
        # Simulate that some states are "bad" (low rewards) and need good actions
        self.R = np.random.randn(n_states, n_actions) * 0.5
        
        # Create a "good" region (states with higher rewards)
        good_states = np.random.choice(n_states, size=n_states // 5, replace=False)
        for s in good_states:
            optimal_action = np.random.randint(n_actions)
            self.R[s, optimal_action] += 2.0  # Bonus for optimal action in good state
        
        # Create absorbing "terminal" states (representing recovery or adverse outcome)
        self.terminal_states = set(np.random.choice(n_states, size=n_states // 10, replace=False))
        
    def reset(self):
        """
        Reset environment to initial state
        
        Returns:
            Initial state
        """
        # Start from a random non-terminal state
        while True:
            self.current_state = np.random.randint(self.n_states)
            if self.current_state not in self.terminal_states:
                break
        return self.current_state
    
    def step(self, action):
        """
        Take an action in the environment
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        state = self.current_state
        
        # Get reward
        reward = self.R[state, action] + np.random.randn() * 0.1  # Add noise
        
        # Transition to next state
        next_state = np.random.choice(self.n_states, p=self.P[state, action])
        
        # Check if terminal
        done = next_state in self.terminal_states
        
        # Add terminal reward
        if done:
            # Positive terminal reward for reaching good terminal states
            if np.random.random() < 0.7:  # 70% chance of good outcome
                reward += 5.0
            else:
                reward -= 2.0
        
        self.current_state = next_state
        
        info = {
            'prev_state': state,
            'action': action,
            'is_terminal': done
        }
        
        return next_state, reward, done, info
    
    def get_optimal_policy(self):
        """
        Compute the optimal policy using value iteration (for comparison)
        
        Returns:
            Optimal policy array
        """
        # Simple value iteration
        V = np.zeros(self.n_states)
        gamma = 0.99
        
        for _ in range(100):  # Fixed number of iterations
            V_new = np.zeros(self.n_states)
            for s in range(self.n_states):
                if s in self.terminal_states:
                    continue
                # Bellman optimality equation
                Q_values = [self.R[s, a] + gamma * np.dot(self.P[s, a], V) 
                           for a in range(self.n_actions)]
                V_new[s] = max(Q_values)
            V = V_new
        
        # Extract policy
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            if s in self.terminal_states:
                continue
            Q_values = [self.R[s, a] + gamma * np.dot(self.P[s, a], V) 
                       for a in range(self.n_actions)]
            policy[s] = np.argmax(Q_values)
        
        return policy

"""
Q-Learning Agent for AI Clinician with LD2Z Scheduler
Implements reinforcement learning for medical decision making
"""

import numpy as np
from ld2z_scheduler import LD2ZScheduler


class QLearningAgent:
    """
    Q-Learning agent using LD2Z learning rate scheduler
    
    This agent learns optimal policies for sequential decision making,
    using the LD2Z learning rate schedule for improved convergence.
    """
    
    def __init__(self, n_states, n_actions, gamma=0.99, 
                 initial_lr=1.0, use_ld2z=True):
        """
        Initialize Q-Learning agent
        
        Args:
            n_states: Number of discrete states
            n_actions: Number of discrete actions
            gamma: Discount factor for future rewards
            initial_lr: Initial learning rate
            use_ld2z: Whether to use LD2Z scheduler (True) or constant learning rate (False)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.use_ld2z = use_ld2z
        
        # Initialize Q-table
        self.Q = np.zeros((n_states, n_actions))
        
        # Track visit counts for each state-action pair (for LD2Z)
        self.visit_counts = np.zeros((n_states, n_actions), dtype=int)
        
        # Initialize scheduler
        if use_ld2z:
            self.scheduler = LD2ZScheduler(initial_lr=initial_lr)
        else:
            self.constant_lr = initial_lr
            
        # Statistics tracking
        self.total_steps = 0
        self.episode_rewards = []
        
    def select_action(self, state, epsilon=0.1):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            epsilon: Exploration probability
            
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action
            return np.argmax(self.Q[state, :])
    
    def update(self, state, action, reward, next_state, done=False):
        """
        Update Q-values using Q-learning update rule with LD2Z scheduler
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Get learning rate
        if self.use_ld2z:
            self.visit_counts[state, action] += 1
            alpha = self.scheduler.get_learning_rate(self.visit_counts[state, action])
        else:
            alpha = self.constant_lr
        
        # Q-learning update
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state, :])
        
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += alpha * td_error
        
        # Update statistics
        self.total_steps += 1
        if self.use_ld2z:
            self.scheduler.step()
        
        return td_error
    
    def get_policy(self):
        """
        Get the greedy policy (best action for each state)
        
        Returns:
            Array of best actions for each state
        """
        return np.argmax(self.Q, axis=1)
    
    def get_value_function(self):
        """
        Get the value function (max Q-value for each state)
        
        Returns:
            Array of values for each state
        """
        return np.max(self.Q, axis=1)

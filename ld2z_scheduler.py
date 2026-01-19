"""
LD2Z Learning Rate Scheduler for Q-Learning
Based on: "Sharp asymptotic theory for Q-learning with ld2z learning rate"
"""

import numpy as np


class LD2ZScheduler:
    """
    Learning rate scheduler using LD2Z (learning rate proportional to 1/t^{2/3})
    
    The LD2Z scheduler provides a learning rate that decays as 1/t^{2/3},
    which has been shown to provide optimal convergence properties for Q-learning.
    """
    
    def __init__(self, initial_lr=1.0, exponent=2/3):
        """
        Initialize LD2Z scheduler
        
        Args:
            initial_lr: Initial learning rate multiplier
            exponent: Decay exponent (default 2/3 for LD2Z)
        """
        self.initial_lr = initial_lr
        self.exponent = exponent
        self.step_count = 0
        
    def get_learning_rate(self, state_action_count=None):
        """
        Get current learning rate
        
        Args:
            state_action_count: Number of times this state-action pair has been visited
                              If None, uses global step count
        
        Returns:
            Current learning rate
        """
        if state_action_count is None:
            state_action_count = self.step_count
        
        # Avoid division by zero
        if state_action_count == 0:
            return self.initial_lr
            
        return self.initial_lr / (state_action_count ** self.exponent)
    
    def step(self):
        """Increment the step counter"""
        self.step_count += 1
        
    def reset(self):
        """Reset the scheduler"""
        self.step_count = 0

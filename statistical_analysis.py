"""
Statistical Analysis Tools for Q-Learning with LD2Z Scheduler
Analyzes convergence properties and performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt


class StatisticalAnalyzer:
    """
    Analyzer for statistical properties of Q-learning with LD2Z scheduler
    """
    
    def __init__(self):
        """Initialize the statistical analyzer"""
        self.history = {
            'q_values': [],
            'td_errors': [],
            'rewards': [],
            'learning_rates': [],
            'value_differences': []
        }
        
    def record_step(self, q_table, td_error, reward, learning_rate):
        """
        Record statistics for a training step
        
        Args:
            q_table: Current Q-table
            td_error: TD error from update
            reward: Reward received
            learning_rate: Current learning rate
        """
        self.history['q_values'].append(np.mean(np.abs(q_table)))
        self.history['td_errors'].append(abs(td_error))
        self.history['rewards'].append(reward)
        self.history['learning_rates'].append(learning_rate)
        
        if len(self.history['q_values']) > 1:
            value_diff = abs(self.history['q_values'][-1] - self.history['q_values'][-2])
            self.history['value_differences'].append(value_diff)
        else:
            self.history['value_differences'].append(0)
    
    def compute_convergence_metrics(self):
        """
        Compute convergence metrics
        
        Returns:
            Dictionary containing convergence statistics
        """
        metrics = {}
        
        if len(self.history['td_errors']) > 100:
            # Compute moving average of TD errors (convergence indicator)
            window = min(100, len(self.history['td_errors']) // 10)
            recent_td_errors = self.history['td_errors'][-window:]
            metrics['mean_recent_td_error'] = np.mean(recent_td_errors)
            metrics['std_recent_td_error'] = np.std(recent_td_errors)
            
            # Compute trend in value differences
            recent_value_diffs = self.history['value_differences'][-window:]
            metrics['mean_value_change'] = np.mean(recent_value_diffs)
            
            # Overall statistics
            metrics['total_steps'] = len(self.history['td_errors'])
            metrics['mean_reward'] = np.mean(self.history['rewards'])
            metrics['std_reward'] = np.std(self.history['rewards'])
            
            # Convergence rate estimation (based on TD error decay)
            if len(self.history['td_errors']) > 500:
                early_td = np.mean(self.history['td_errors'][100:200])
                late_td = np.mean(self.history['td_errors'][-100:])
                metrics['td_error_reduction'] = early_td - late_td
                metrics['convergence_ratio'] = late_td / early_td if early_td > 0 else 0
        
        return metrics
    
    def plot_convergence(self, save_path=None):
        """
        Plot convergence metrics
        
        Args:
            save_path: Path to save the figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot TD errors
        axes[0, 0].plot(self.history['td_errors'], alpha=0.6)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('TD Error')
        axes[0, 0].set_title('TD Error over Time')
        axes[0, 0].set_yscale('log')
        
        # Plot learning rates
        axes[0, 1].plot(self.history['learning_rates'])
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('LD2Z Learning Rate Schedule')
        axes[0, 1].set_yscale('log')
        
        # Plot cumulative rewards (moving average)
        if len(self.history['rewards']) > 50:
            window = min(50, len(self.history['rewards']) // 10)
            rewards_ma = np.convolve(self.history['rewards'], 
                                    np.ones(window)/window, mode='valid')
            axes[1, 0].plot(rewards_ma)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Reward (Moving Average)')
            axes[1, 0].set_title('Reward over Time')
        
        # Plot value function changes
        axes[1, 1].plot(self.history['value_differences'], alpha=0.6)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Value Function Change')
        axes[1, 1].set_title('Value Function Convergence')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_report(self):
        """
        Generate a text report of statistical properties
        
        Returns:
            String containing the report
        """
        metrics = self.compute_convergence_metrics()
        
        report = "=" * 60 + "\n"
        report += "Q-Learning with LD2Z Scheduler - Statistical Analysis\n"
        report += "=" * 60 + "\n\n"
        
        if metrics:
            report += f"Total Training Steps: {metrics.get('total_steps', 0)}\n\n"
            
            report += "Convergence Metrics:\n"
            report += f"  Mean Recent TD Error: {metrics.get('mean_recent_td_error', 0):.6f}\n"
            report += f"  Std Recent TD Error: {metrics.get('std_recent_td_error', 0):.6f}\n"
            report += f"  Mean Value Change: {metrics.get('mean_value_change', 0):.6f}\n"
            
            if 'convergence_ratio' in metrics:
                report += f"  Convergence Ratio: {metrics['convergence_ratio']:.4f}\n"
                report += f"  TD Error Reduction: {metrics['td_error_reduction']:.6f}\n"
            
            report += f"\nReward Statistics:\n"
            report += f"  Mean Reward: {metrics.get('mean_reward', 0):.4f}\n"
            report += f"  Std Reward: {metrics.get('std_reward', 0):.4f}\n"
        else:
            report += "Insufficient data for analysis (need > 100 steps)\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report

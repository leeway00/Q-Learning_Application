"""
Main Training Script for AI Clinician with LD2Z Scheduler
"""

import numpy as np
import matplotlib.pyplot as plt
from q_learning_agent import QLearningAgent
from environment import SimpleMedicalMDP
from statistical_analysis import StatisticalAnalyzer


def train_agent(env, agent, analyzer, n_episodes=1000, max_steps_per_episode=100, 
                epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    """
    Train Q-learning agent with LD2Z scheduler
    
    Args:
        env: Environment to train in
        agent: Q-learning agent
        analyzer: Statistical analyzer
        n_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Decay rate for exploration
        
    Returns:
        Trained agent and analyzer with results
    """
    epsilon = epsilon_start
    episode_rewards = []
    
    print(f"Training Q-Learning Agent with {'LD2Z' if agent.use_ld2z else 'Constant'} Scheduler")
    print(f"Episodes: {n_episodes}, Max Steps: {max_steps_per_episode}")
    print("-" * 60)
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(state, epsilon)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update Q-values
            td_error = agent.update(state, action, reward, next_state, done)
            
            # Record statistics
            if agent.use_ld2z:
                lr = agent.scheduler.get_learning_rate(
                    agent.visit_counts[state, action]
                )
            else:
                lr = agent.constant_lr
            analyzer.record_step(agent.Q, td_error, reward, lr)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Decay exploration rate
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{n_episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Epsilon: {epsilon:.3f}")
    
    print("-" * 60)
    print("Training completed!")
    
    return agent, analyzer, episode_rewards


def compare_schedulers(env, n_states, n_actions, n_episodes=1000, seed=42):
    """
    Compare LD2Z scheduler with constant learning rate
    
    Args:
        env: Environment
        n_states: Number of states
        n_actions: Number of actions
        n_episodes: Number of training episodes
        seed: Random seed
        
    Returns:
        Dictionary with results from both schedulers
    """
    np.random.seed(seed)
    
    print("\n" + "=" * 60)
    print("Comparing LD2Z Scheduler vs Constant Learning Rate")
    print("=" * 60 + "\n")
    
    # Train with LD2Z scheduler
    print("\n1. Training with LD2Z Scheduler")
    agent_ld2z = QLearningAgent(n_states, n_actions, use_ld2z=True, initial_lr=1.0)
    analyzer_ld2z = StatisticalAnalyzer()
    agent_ld2z, analyzer_ld2z, rewards_ld2z = train_agent(
        env, agent_ld2z, analyzer_ld2z, n_episodes=n_episodes
    )
    
    # Reset environment
    env.reset()
    
    # Train with constant learning rate
    print("\n2. Training with Constant Learning Rate")
    agent_const = QLearningAgent(n_states, n_actions, use_ld2z=False, initial_lr=0.1)
    analyzer_const = StatisticalAnalyzer()
    agent_const, analyzer_const, rewards_const = train_agent(
        env, agent_const, analyzer_const, n_episodes=n_episodes
    )
    
    # Compare results
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    
    print("\nLD2Z Scheduler:")
    print(analyzer_ld2z.generate_report())
    
    print("\nConstant Learning Rate:")
    print(analyzer_const.generate_report())
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot rewards
    window = 50
    rewards_ld2z_ma = np.convolve(rewards_ld2z, np.ones(window)/window, mode='valid')
    rewards_const_ma = np.convolve(rewards_const, np.ones(window)/window, mode='valid')
    
    axes[0].plot(rewards_ld2z_ma, label='LD2Z', alpha=0.8)
    axes[0].plot(rewards_const_ma, label='Constant LR', alpha=0.8)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward (Moving Average)')
    axes[0].set_title('Training Rewards Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot TD errors
    axes[1].plot(analyzer_ld2z.history['td_errors'], label='LD2Z', alpha=0.6)
    axes[1].plot(analyzer_const.history['td_errors'], label='Constant LR', alpha=0.6)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('TD Error')
    axes[1].set_title('TD Error Comparison')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scheduler_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved as 'scheduler_comparison.png'")
    
    return {
        'ld2z': {
            'agent': agent_ld2z,
            'analyzer': analyzer_ld2z,
            'rewards': rewards_ld2z
        },
        'constant': {
            'agent': agent_const,
            'analyzer': analyzer_const,
            'rewards': rewards_const
        }
    }


def main():
    """Main function"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create environment
    print("Creating medical decision-making environment...")
    n_states = 50
    n_actions = 5
    env = SimpleMedicalMDP(n_states=n_states, n_actions=n_actions, seed=42)
    
    # Compare schedulers
    results = compare_schedulers(env, n_states, n_actions, n_episodes=1000)
    
    # Generate detailed plots
    print("\nGenerating convergence plots...")
    
    fig1 = results['ld2z']['analyzer'].plot_convergence('ld2z_convergence.png')
    print("LD2Z convergence plot saved as 'ld2z_convergence.png'")
    
    fig2 = results['constant']['analyzer'].plot_convergence('constant_lr_convergence.png')
    print("Constant LR convergence plot saved as 'constant_lr_convergence.png'")
    
    print("\n" + "=" * 60)
    print("Training and analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

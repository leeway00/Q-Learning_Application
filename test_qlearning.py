"""
Tests for Q-Learning Application with LD2Z Scheduler
"""

import numpy as np
from ld2z_scheduler import LD2ZScheduler
from q_learning_agent import QLearningAgent
from environment import SimpleMedicalMDP
from statistical_analysis import StatisticalAnalyzer


def test_ld2z_scheduler():
    """Test LD2Z scheduler basic functionality"""
    print("Testing LD2Z Scheduler...")
    
    scheduler = LD2ZScheduler(initial_lr=1.0, exponent=2/3)
    
    # Test initial learning rate
    assert scheduler.get_learning_rate(1) == 1.0, "Initial LR should be 1.0"
    
    # Test decay
    lr_10 = scheduler.get_learning_rate(10)
    lr_100 = scheduler.get_learning_rate(100)
    lr_1000 = scheduler.get_learning_rate(1000)
    
    assert lr_10 > lr_100 > lr_1000, "LR should decrease with time"
    
    # Test exponent effect (2/3 should be between constant and 1/t)
    # For t=100: 1/t^(2/3) ≈ 0.0464, 1/t = 0.01
    assert 0.04 < lr_100 < 0.05, f"LR at t=100 should be ~0.046, got {lr_100}"
    
    print("  ✓ LD2Z scheduler tests passed")


def test_q_learning_agent():
    """Test Q-learning agent functionality"""
    print("Testing Q-Learning Agent...")
    
    n_states, n_actions = 10, 3
    agent = QLearningAgent(n_states, n_actions, use_ld2z=True)
    
    # Test initialization
    assert agent.Q.shape == (n_states, n_actions), "Q-table shape incorrect"
    assert agent.visit_counts.shape == (n_states, n_actions), "Visit counts shape incorrect"
    assert np.all(agent.Q == 0), "Q-table should be initialized to zero"
    
    # Test action selection
    state = 0
    action = agent.select_action(state, epsilon=0.0)  # Greedy
    assert 0 <= action < n_actions, "Action should be valid"
    
    # Test update
    reward = 1.0
    next_state = 1
    td_error = agent.update(state, action, reward, next_state, done=False)
    
    assert agent.Q[state, action] != 0, "Q-value should be updated"
    assert agent.visit_counts[state, action] == 1, "Visit count should increment"
    assert agent.total_steps == 1, "Step count should increment"
    
    print("  ✓ Q-learning agent tests passed")


def test_environment():
    """Test medical MDP environment"""
    print("Testing Medical MDP Environment...")
    
    n_states, n_actions = 20, 4
    env = SimpleMedicalMDP(n_states, n_actions, seed=42)
    
    # Test initialization
    assert env.n_states == n_states, "State count incorrect"
    assert env.n_actions == n_actions, "Action count incorrect"
    assert env.P.shape == (n_states, n_actions, n_states), "Transition matrix shape incorrect"
    assert env.R.shape == (n_states, n_actions), "Reward matrix shape incorrect"
    
    # Test transition probabilities sum to 1
    for s in range(n_states):
        for a in range(n_actions):
            prob_sum = np.sum(env.P[s, a, :])
            assert np.isclose(prob_sum, 1.0), f"Transition probs should sum to 1, got {prob_sum}"
    
    # Test reset
    state = env.reset()
    assert 0 <= state < n_states, "Initial state should be valid"
    assert state not in env.terminal_states, "Initial state should not be terminal"
    
    # Test step
    next_state, reward, done, info = env.step(0)
    assert 0 <= next_state < n_states, "Next state should be valid"
    assert isinstance(reward, (int, float)), "Reward should be numeric"
    assert isinstance(done, bool), "Done should be boolean"
    
    print("  ✓ Environment tests passed")


def test_statistical_analyzer():
    """Test statistical analyzer"""
    print("Testing Statistical Analyzer...")
    
    analyzer = StatisticalAnalyzer()
    
    # Record some data
    q_table = np.random.randn(10, 3)
    for i in range(200):
        td_error = np.random.randn() * 0.5
        reward = np.random.randn()
        lr = 1.0 / (i + 1) ** (2/3)
        analyzer.record_step(q_table, td_error, reward, lr)
    
    # Test metrics computation
    metrics = analyzer.compute_convergence_metrics()
    assert 'mean_recent_td_error' in metrics, "Should compute TD error metrics"
    assert 'mean_reward' in metrics, "Should compute reward metrics"
    assert metrics['total_steps'] == 200, "Step count incorrect"
    
    # Test report generation
    report = analyzer.generate_report()
    assert isinstance(report, str), "Report should be a string"
    assert len(report) > 0, "Report should not be empty"
    
    print("  ✓ Statistical analyzer tests passed")


def test_integration():
    """Test full integration"""
    print("Testing Integration...")
    
    # Setup
    np.random.seed(42)
    n_states, n_actions = 10, 3
    env = SimpleMedicalMDP(n_states, n_actions, seed=42)
    agent = QLearningAgent(n_states, n_actions, use_ld2z=True)
    analyzer = StatisticalAnalyzer()
    
    # Run a few episodes
    n_episodes = 10
    for ep in range(n_episodes):
        state = env.reset()
        for step in range(20):
            action = agent.select_action(state, epsilon=0.1)
            next_state, reward, done, info = env.step(action)
            td_error = agent.update(state, action, reward, next_state, done)
            
            lr = agent.scheduler.get_learning_rate(agent.visit_counts[state, action])
            analyzer.record_step(agent.Q, td_error, reward, lr)
            
            state = next_state
            if done:
                break
    
    # Verify training occurred
    assert agent.total_steps > 0, "Should have taken steps"
    assert np.any(agent.Q != 0), "Q-table should be updated"
    assert len(analyzer.history['rewards']) > 0, "Should have recorded rewards"
    
    print("  ✓ Integration tests passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Running Tests for Q-Learning Application")
    print("=" * 60 + "\n")
    
    try:
        test_ld2z_scheduler()
        test_q_learning_agent()
        test_environment()
        test_statistical_analyzer()
        test_integration()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60 + "\n")
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

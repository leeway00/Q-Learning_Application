# Q-Learning Application: AI Clinician with LD2Z Scheduler

This repository implements a Q-learning approach for medical decision-making (inspired by the AI Clinician project) using the LD2Z learning rate scheduler for optimal convergence properties.

## Overview

This project combines:
- **AI Clinician**: Reinforcement learning approach for optimal treatment strategies in intensive care (based on [matthieukomorowski/AI_Clinician](https://github.com/matthieukomorowski/AI_Clinician))
- **LD2Z Scheduler**: Learning rate schedule proportional to 1/t^(2/3) from [leeway00/Q-learning_forked](https://github.com/leeway00/Q-learning_forked)
- **Statistical Analysis**: Tools to analyze convergence properties and performance metrics

## Features

- ✅ Q-learning agent with LD2Z learning rate scheduler
- ✅ Simplified medical decision-making MDP environment
- ✅ Comprehensive statistical analysis tools
- ✅ Convergence visualization and comparison
- ✅ Interactive Jupyter notebook for exploration

## Installation

```bash
# Clone the repository
git clone https://github.com/leeway00/Q-Learning_Application.git
cd Q-Learning_Application

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run the Training Script

```bash
python train.py
```

This will:
- Train a Q-learning agent with LD2Z scheduler
- Train a baseline agent with constant learning rate
- Compare their performance
- Generate convergence plots
- Output statistical analysis

### 2. Use the Jupyter Notebook

```bash
jupyter notebook AI_Clinician_LD2Z.ipynb
```

The notebook provides:
- Interactive visualization of the LD2Z scheduler
- Step-by-step training process
- Detailed statistical analysis
- Policy and value function visualization

## Project Structure

```
Q-Learning_Application/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── ld2z_scheduler.py            # LD2Z learning rate scheduler
├── q_learning_agent.py          # Q-learning agent implementation
├── environment.py               # Simplified medical MDP environment
├── statistical_analysis.py      # Statistical analysis tools
├── train.py                     # Main training script
└── AI_Clinician_LD2Z.ipynb     # Interactive Jupyter notebook
```

## Key Components

### LD2Z Scheduler

The LD2Z scheduler implements a learning rate that decays as 1/t^(2/3):

```python
from ld2z_scheduler import LD2ZScheduler

scheduler = LD2ZScheduler(initial_lr=1.0, exponent=2/3)
lr = scheduler.get_learning_rate(step)
```

**Properties:**
- Slower decay than standard 1/t schedule
- Optimal convergence guarantees for Q-learning
- Balances exploration and exploitation

### Q-Learning Agent

```python
from q_learning_agent import QLearningAgent

agent = QLearningAgent(
    n_states=50,
    n_actions=5,
    gamma=0.99,
    initial_lr=1.0,
    use_ld2z=True
)
```

### Statistical Analysis

```python
from statistical_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()
# Record training statistics
analyzer.record_step(q_table, td_error, reward, learning_rate)

# Generate report
print(analyzer.generate_report())

# Plot convergence
analyzer.plot_convergence('convergence.png')
```

## Results

The implementation provides:

1. **Convergence Analysis**: Track TD errors, value function changes, and rewards over time
2. **Scheduler Comparison**: Compare LD2Z with constant learning rate baselines
3. **Policy Quality**: Evaluate learned policies and value functions
4. **Statistical Properties**: Analyze convergence speed, stability, and performance

### Example Output

```
Q-Learning with LD2Z Scheduler - Statistical Analysis
============================================================

Total Training Steps: 50000

Convergence Metrics:
  Mean Recent TD Error: 0.045632
  Std Recent TD Error: 0.023451
  Mean Value Change: 0.000234
  Convergence Ratio: 0.1234
  TD Error Reduction: 1.234567

Reward Statistics:
  Mean Reward: 2.3456
  Std Reward: 1.2345
```

## Background

### AI Clinician

The AI Clinician project (Komorowski et al., Nature Medicine 2018) uses reinforcement learning to optimize treatment strategies for sepsis patients in intensive care. Key aspects:
- Learns optimal policies for IV fluid and vasopressor administration
- Uses Markov Decision Process (MDP) framework
- Trained on MIMIC-III dataset

### LD2Z Scheduler

The LD2Z learning rate schedule comes from "Sharp asymptotic theory for Q-learning with ld2z learning rate and its generalization". Key properties:
- Learning rate: α(t) = α₀ / t^(2/3)
- Optimal convergence rate for Q-learning
- Better balance between bias and variance compared to 1/t schedule

## Statistical Properties Analyzed

This implementation analyzes:

1. **Convergence Speed**: How quickly the algorithm converges
2. **TD Error Dynamics**: Evolution of temporal difference errors
3. **Value Function Stability**: Changes in estimated values over time
4. **Reward Performance**: Cumulative and average rewards
5. **Policy Quality**: Comparison with optimal policies
6. **Learning Rate Impact**: Effect of different schedules on convergence

## Extensions

Potential extensions:
- Integration with real medical datasets (MIMIC-III, eICU)
- More complex state/action spaces
- Deep Q-learning with LD2Z scheduler
- Multi-agent scenarios
- Safety constraints for medical applications

## References

1. Komorowski, M., Celi, L.A., Badawi, O. et al. "The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care." Nature Medicine 24, 1716–1720 (2018). https://doi.org/10.1038/s41591-018-0213-5

2. "Sharp asymptotic theory for Q-learning with ld2z learning rate and its generalization" - Implementation at https://github.com/leeway00/Q-learning_forked

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.

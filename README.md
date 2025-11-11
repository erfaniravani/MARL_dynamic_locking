# MARL Dynamic Locking

This repository contains the implementation and experiments for  
**"Practical Reinforcement Learning-Based Dynamic Defense Against Side Channel Attacks in Last Level Caches."**  

The framework builds upon **[AutoCAT](https://github.com/example/AutoCAT)** and **MACTA** to model cache behavior and train multi-agent reinforcement learning (MARL) agents for dynamic cache-line locking defense.

---

## 📁 Code Structure

### 🔹 Cache Modeling
- `AutoCAT/src/cache.py`  
- `AutoCAT/src/cache_simulator.py`  
  > Implements a 16-way set-associative cache simulator used for the experiments.

---

### 🔹 Model Definitions
- `AutoCAT/src/rlmeta/cache_ppo_transformer_model.py`  
- `AutoCAT/src/rlmeta/cache_ppo_transformer_model_defender.py`  
- `AutoCAT/src/rlmeta/cache_ppo_transformer_model_student.py`  
  > Define the PPO Transformer models used for attacker, defender, and student (distillation) agents.

---

### 🔹 Agents
- `AutoCAT/src/rlmeta/macta/agent/`  
  > Contains agent implementations:  
  - `PPO_agent` — RL-based agent.  
  - `spec_agent` — benign (non-attacking) agent.  
  - `prime_probe`, `evict_reload` — static attack agents.

---

### 🔹 Configuration Files
- `AutoCAT/src/rlmeta/macta/config/`  
  - `macta.yaml` — main configuration for MARL training.  
- `AutoCAT/src/rlmeta/macta/config/env_config/`  
  > Environment modeling configuration files.  
- `AutoCAT/src/rlmeta/macta/config/model_config/`  
  > Model training configurations for both attack and defense models.

---

### 🔹 Environments
- `AutoCAT/src/rlmeta/macta/env/`  
  > Contains RL environments.  
  - `cache_attacker_defender_env.py` — primary environment used for the MARL training experiments.

---

### 🔹 Model Interfaces
- `AutoCAT/src/rlmeta/macta/model/`  
  > Implements model interfaces and pooling mechanisms.

---

### 🔹 Training Scripts
- `AutoCAT/src/rlmeta/macta/train/`  
  > Training entry points:  
  - `train_autocat.py` — trains a single attacker model.  
  - `train_macta_defender.py` — trains the MARL-based dynamic locking defender.  
  - `train_student.py` — performs knowledge distillation from defender to student model.

---

## ⚙️ Dependencies
- Python 3.9+  
- PyTorch  
- Gymnasium  
- NumPy, SciPy  

(See environment YAML or requirements file for details.)

---

## 🚀 Usage
Example (training MARL defender):
```bash
python AutoCAT/src/rlmeta/macta/train/train_macta_defender.py --config AutoCAT/src/rlmeta/macta/config/macta.yaml

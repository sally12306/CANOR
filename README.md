# CANOR

This repository contains the codes for our paper titled "CANOR: Conflict-Aware Novelty-Weighted Offline-to-Online Reinforcement Learning for Robot Crowd Navigation". 



### Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```
   
### Overview
This repository is organized in two parts:
* crowd_nav/ folder contains configurations and policies used in the simulator.
* crowd_sim/ folder contains the simulation environment.

### Getting Started
The complete training pipeline for our model consists of three main stages: data collection, offline pre-training, and online fine-tuning, which should be executed inside the crowd_nav/ folder. Please execute them in the following order.
1. Data collection
```
python trajectory.py
```
2. Offline pre-training
```
python canor.py
```
3. Online fine-tuning
```
python online_finetune_canor.py
```


### Credits
This repository contains the code for the following papers:

- [CORL: Research-oriented Deep Offline Reinforcement Learning Library](https://github.com/tinkoff-ai/CORL).
- [ Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning](https://github.com/nakamotoo/Cal-QL).

### Contact
If you have any questions or find any bugs, please feel free to open an issue or pull request.





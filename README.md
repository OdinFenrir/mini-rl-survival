Mini RL Survival (Kid-Friendly) — Q-learning that ACTUALLY Learns
===============================================================

This is a tiny Reinforcement Learning project you can understand.

You get:
- a small grid world (env.py)
- a tabular Q-learning agent (qlearn.py)
- a trainer you can tweak (train.py)

Goal
----
Survive longer by:
- avoiding hazards (X)
- collecting food (F) to regain energy

Files
-----
- env.py     : environment
- qlearn.py  : Q-learning (save/load included)
- train.py   : train + eval + play modes (argparse)

Quick start (Windows PowerShell)
--------------------------------
1) Train:

    python .\train.py --episodes 2000 --eval-every 200 --eval-episodes 50 --save qtable.pkl

2) Watch the learned agent play (greedy policy):

    python .\train.py --play --load qtable.pkl --max-steps 200 --sleep 0.05

Easy experiments
----------------
- Make it learn slower but better:
    --eps-decay 80000

- Make survival harder:
    --hazards 14 --energy-step 2

- Make food more valuable:
    --energy-food 25

Best tip
--------
If you want to *see learning*:
- train for 5k+ episodes
- then watch --play
- compare it to random behavior (delete qtable.pkl or don't --load)

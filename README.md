Mini RL Agent (Kid-Friendly) — Learn to Survive + Get Food
========================================================

What this is
------------
A tiny *tabular* Reinforcement Learning project you can actually understand.
No deep learning, no big libraries.

The world:
- A 10x10 grid.
- Your agent starts with ENERGY.
- Each step costs a bit of energy.
- There is FOOD on the grid. Reaching it gives reward and refills energy.
- There are a few HAZARDS. Stepping on them ends the episode.
- The agent learns using Q-learning.

Goal:
- Survive as long as possible AND keep collecting food.

How to run
----------
1) Open a terminal in this folder
2) Run:

    python train.py

You’ll see:
- episode stats
- a "greedy" evaluation run every so often
- a tiny ASCII render of the grid during evaluation

Files
-----
- env.py     : the environment (grid world)
- qlearn.py  : Q-learning agent (epsilon-greedy)
- train.py   : training loop + periodic evaluation

Easy tweaks (try these!)
------------------------
In train.py:
- Increase episodes (more learning): EPISODES
- Change exploration: EPS_START / EPS_END
- Change learning rate: ALPHA
- Change discount: GAMMA

In env.py:
- Grid size
- Step energy cost
- How much energy food gives
- Number of hazards

What to look for
----------------
At first the agent dies quickly.
After some episodes it learns to:
- avoid hazards
- head toward food
- keep energy up (survive longer)

Have fun iterating.

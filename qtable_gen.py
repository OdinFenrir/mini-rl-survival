import pickle, random
q = {}
# small random qtable for 10x10 grid (keeps file small)
for ax in range(10):
    for ay in range(10):
        for fx in range(10):
            for fy in range(10):
                if random.random() < 0.006:
                    state = (ax, ay, fx, fy, 10)
                    q[state] = [random.random() for _ in range(4)]
with open("qtable.pkl","wb") as f:
    pickle.dump(q, f)
print("Wrote qtable.pkl (states=%d)" % len(q))

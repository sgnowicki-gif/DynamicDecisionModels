import numpy as np

# Transition matrices from (b), rows = current state 1-8, cols = next state 1-8
P = {
    'N': np.array([
        [0.8, 0.1, 0,   0.1, 0,   0,   0,   0],
        [0.1, 0.8, 0.1, 0,   0,   0,   0,   0],
        [1/6, 1/6, 0,   1/6, 1/6, 1/6, 1/6, 0],
        [0.7, 0,   0,   0.2, 0,   0.1, 0,   0],
        [0,   0,   0.7, 0,   0.2, 0,   0,   0.1],
        [0,   0,   0,   0.7, 0,   0.2, 0.1, 0],
        [0,   0,   0,   0,   0,   0.1, 0.8, 0.1],
        [1/6, 1/6, 0,   1/6, 1/6, 1/6, 1/6, 0],
    ]),
    'E': np.array([
        [0.2, 0.7, 0,   0.1, 0,   0,   0,   0],
        [0.1, 0.2, 0.7, 0,   0,   0,   0,   0],
        [1/6, 1/6, 0,   1/6, 1/6, 1/6, 1/6, 0],
        [0.1, 0,   0,   0.8, 0,   0.1, 0,   0],
        [0,   0,   0.1, 0,   0.8, 0,   0,   0.1],
        [0,   0,   0,   0.1, 0,   0.2, 0.7, 0],
        [0,   0,   0,   0,   0,   0.1, 0.2, 0.7],
        [1/6, 1/6, 0,   1/6, 1/6, 1/6, 1/6, 0],
    ]),
    'S': np.array([
        [0.2, 0.1, 0,   0.7, 0,   0,   0,   0],
        [0.1, 0.8, 0.1, 0,   0,   0,   0,   0],
        [1/6, 1/6, 0,   1/6, 1/6, 1/6, 1/6, 0],
        [0.1, 0,   0,   0.2, 0,   0.7, 0,   0],
        [0,   0,   0.1, 0,   0.2, 0,   0,   0.7],
        [0,   0,   0,   0.1, 0,   0.8, 0.1, 0],
        [0,   0,   0,   0,   0,   0.1, 0.8, 0.1],
        [1/6, 1/6, 0,   1/6, 1/6, 1/6, 1/6, 0],
    ]),
    'W': np.array([
        [0.8, 0.1, 0,   0.1, 0,   0,   0,   0],
        [0.7, 0.2, 0.1, 0,   0,   0,   0,   0],
        [1/6, 1/6, 0,   1/6, 1/6, 1/6, 1/6, 0],
        [0.1, 0,   0,   0.8, 0,   0.1, 0,   0],
        [0,   0,   0.1, 0,   0.8, 0,   0,   0.1],
        [0,   0,   0,   0.1, 0,   0.8, 0.1, 0],
        [0,   0,   0,   0,   0,   0.7, 0.2, 0.1],
        [1/6, 1/6, 0,   1/6, 1/6, 1/6, 1/6, 0],
    ]),
}

# Emission matrix from (c), rows = state 1-8, cols = East, West, Both, None, Good, Bad
B = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1],
], dtype=float)

# Initial belief
b0 = np.array([1/6, 1/6, 0, 1/6, 1/6, 1/6, 1/6, 0], dtype=float)

gamma = 0.95

#Reward array given what state you end up in
R = np.array([-0.04, -0.04, 1.0, -0.04, -0.04, -0.04, -0.04, -1.0])  # states 1-8


def belief_update(b, a, o):
    pred = b @ P[a]            # distribution over next state
    post = B[:, o] * pred      # weight by observation likelihood
    return post / post.sum()   # normalize

def V(b, steps):
    if steps == 0:
        return 0.0             # check against the problem's horizon convention
    best = -np.inf
    for a in P:
        immediate = float((b @ P[a]) @ R)      # expected reward under b (fill from the problem's reward definition: +1 at 3, -1 at 8, -0.04 elsewhere)
        future = 0.0
        for o in range(6):
            pred = b @ P[a]
            p_o = float(B[:, o] @ pred)   # P(observation o | b, a)
            if p_o > 0:
                future += p_o * V(belief_update(b, a, o), steps - 1)
        best = max(best, immediate + gamma * future)
    return best

def Q_value(b, a, steps):
    pred = b @ P[a]
    immediate = float(pred @ R)

    future = 0.0
    for o in range(6):
        p_o = float(B[:, o] @ pred)
        if p_o > 0:
            future += p_o * V(belief_update(b, a, o), steps - 1)

    return immediate + gamma * future

def build_tree(b, steps):
    if steps == 0:
        return None

    # best action at this belief, same Q calculation as in V
    a_best = max(P, key=lambda a: Q_value(b, a, steps))

    node = {'belief': b, 'action': a_best, 'children': {}}

    if steps > 1:
        pred = b @ P[a_best]
        for o in range(6):
            p_o = float(B[:, o] @ pred)
            if p_o > 0:  # prune impossible observations
                b_next = belief_update(b, a_best, o)
                node['children'][o] = build_tree(b_next, steps - 1)

    return node

tree = build_tree(b0, 3)

obs_names = ["East", "West", "Both", "None", "Good", "Bad"]

def print_tree(node, indent=0, label="root"):
    if node is None:
        return
    print("  " * indent + f"{label}: action={node['action']}, belief={np.round(node['belief'], 3)}")
    for o, child in node['children'].items():
        print_tree(child, indent + 1, f"obs={obs_names[o]}")

print_tree(tree)

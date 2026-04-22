"""
Multi-hop tasks with genuine computational depth for K-scaling.

Task 1: Pointer chasing — "A→B, B→C, C→D, D→E. Follow from A." Answer: "E"
  Depth scales with chain length. K=1 can resolve 1 hop, K=N resolves N hops.

Task 2: Variable substitution — "a=5, b=a+3, c=b*2, d=c-a. What is d?"
  Each variable depends on previous ones. Depth = number of variables.

Task 3: Text-encoded larger grids (16×16)
  Proven to K-scale at 8×8. Does it scale more at 16×16?

All tasks generate (prompt, answer) pairs as text for the solver+decoder pipeline.
"""
import random
import string


def generate_pointer_chase(depth=4, n_nodes=8):
    """Generate a pointer chasing problem with given chain depth."""
    n_nodes = max(n_nodes, depth + 3)  # ensure enough nodes for chain + distractors
    names = random.sample(string.ascii_uppercase[:min(n_nodes, 26)], min(n_nodes, 26))
    # Build chain: names[0] → names[1] → ... → names[depth]
    chain = names[:depth + 1]
    random.shuffle(chain[1:-1])  # shuffle middle to make it non-trivial

    # Create edges (including distractors)
    edges = []
    for i in range(depth):
        edges.append(f"{chain[i]} points to {chain[i+1]}")
    # Add distractor edges from unused nodes
    for name in names[depth+1:]:
        target = random.choice(names[:depth+1])
        edges.append(f"{name} points to {target}")
    random.shuffle(edges)

    prompt = "Links: " + ". ".join(edges) + f".\nFollowing the chain from {chain[0]}, what is the final destination?"
    answer = chain[-1]
    return prompt, answer, depth


def generate_variable_sub(depth=4):
    """Generate variable substitution with given depth."""
    vars_used = []
    assignments = []
    # First variable: literal
    var = chr(ord('a'))
    val = random.randint(1, 9)
    assignments.append(f"{var} = {val}")
    vars_used.append((var, val))

    for i in range(1, depth):
        new_var = chr(ord('a') + i)
        # Reference a previous variable
        ref_var, ref_val = random.choice(vars_used)
        op = random.choice(['+', '-', '*'])
        operand = random.randint(1, 5)
        if op == '+':
            new_val = ref_val + operand
            expr = f"{ref_var} + {operand}"
        elif op == '-':
            new_val = ref_val - operand
            expr = f"{ref_var} - {operand}"
        else:
            new_val = ref_val * operand
            expr = f"{ref_var} * {operand}"
        assignments.append(f"{new_var} = {expr}")
        vars_used.append((new_var, new_val))

    target_var, target_val = vars_used[-1]
    prompt = "Given: " + ", ".join(assignments) + f".\nWhat is the value of {target_var}?"
    answer = str(target_val)
    return prompt, answer, depth


def generate_text_grid(n=16):
    """Generate text-encoded grid reachability (larger grids)."""
    grid = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if random.random() < 0.3 and (i, j) != (0, 0):
                grid[i][j] = 1

    # BFS reachability
    reach = [[0]*n for _ in range(n)]
    reach[0][0] = 1
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(n):
                if reach[i][j] == 1 and grid[i][j] == 0:
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < n and 0 <= nj < n and reach[ni][nj] == 0 and grid[ni][nj] == 0:
                            reach[ni][nj] = 1
                            changed = True

    grid_str = '/'.join(''.join('#' if c else '.' for c in row) for row in grid)
    reach_str = '/'.join(''.join(str(c) for c in row) for row in reach)
    prompt = f"Grid {n}x{n}: {grid_str}\nReachable from (0,0): "
    answer = reach_str
    return prompt, answer, n


def generate_mixed_batch(batch_size, min_depth=2, max_depth=8, grid_size=16):
    """Generate a mixed batch of multi-hop tasks."""
    prompts = []
    answers = []
    depths = []
    for _ in range(batch_size):
        task = random.choice(['pointer', 'variable', 'grid'])
        depth = random.randint(min_depth, max_depth)
        if task == 'pointer':
            p, a, d = generate_pointer_chase(depth=depth, n_nodes=max(depth+2, 8))
        elif task == 'variable':
            p, a, d = generate_variable_sub(depth=depth)
        else:
            p, a, d = generate_text_grid(n=grid_size)
        prompts.append(p)
        answers.append(a)
        depths.append(d)
    return prompts, answers, depths


if __name__ == '__main__':
    # Demo
    print("=== Pointer Chasing (depth=4) ===")
    p, a, d = generate_pointer_chase(depth=4)
    print(f"Prompt: {p}")
    print(f"Answer: {a}\n")

    print("=== Variable Substitution (depth=5) ===")
    p, a, d = generate_variable_sub(depth=5)
    print(f"Prompt: {p}")
    print(f"Answer: {a}\n")

    print("=== Text Grid 8x8 ===")
    p, a, d = generate_text_grid(n=8)
    print(f"Prompt: {p[:100]}...")
    print(f"Answer: {a[:50]}...")


def generate_proof_chain(depth=4):
    """Generate a logical proof chain requiring multi-hop deduction.
    'If A then B. If B then C. If C then D. Given A is true. Is D true?'
    """
    props = random.sample([
        'raining', 'cold', 'windy', 'cloudy', 'snowing', 'foggy',
        'sunny', 'warm', 'humid', 'stormy', 'dark', 'bright',
        'freezing', 'dry', 'calm', 'hazy', 'clear', 'overcast'
    ], min(depth + 2, 18))

    chain = props[:depth + 1]
    rules = []
    for i in range(depth):
        rules.append(f"If it is {chain[i]} then it is {chain[i+1]}")
    # Add distractors
    for p in props[depth+1:]:
        target = random.choice(chain)
        rules.append(f"If it is {p} then it is {target}")
    random.shuffle(rules)

    prompt = "Rules: " + ". ".join(rules) + f".\nGiven: It is {chain[0]}.\nQuestion: Is it {chain[-1]}?"
    answer = "Yes"
    return prompt, answer, depth

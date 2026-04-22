"""
Synthetic Reasoning Data Generator for PSRT

Generates tasks where multi-step reasoning is provably required:
1. Arithmetic chains: "12 * 3 + 7 - 2 = ?" (variable length)
2. Grid mazes: find path from S to E in a text grid
3. Logic deductions: "A > B, B > C, C > D. Is A > D?"
4. Counting/tracking: "3 people enter, 2 leave, 4 enter. How many?"
5. Pattern completion: "2, 4, 8, 16, ?"

Each task has scalable difficulty (easy → hard), so the model can learn
to use K=1 for easy and K=2+ for hard.
"""

import random
import string


# ============================================================
# 1. Arithmetic Chains
# ============================================================

def gen_arithmetic(difficulty=1):
    """Generate arithmetic chain. Difficulty 1-5 controls length and op complexity."""
    n_ops = difficulty + 1  # 2 ops for easy, 6 for hard
    ops = ['+', '-', '*']
    if difficulty >= 3:
        ops.append('//')

    # Start with a number
    val = random.randint(2, 20)
    expr = str(val)

    for _ in range(n_ops):
        op = random.choice(ops)
        if op == '*':
            n = random.randint(2, 5)
        elif op == '//':
            # Make sure it divides evenly
            divisors = [d for d in range(2, min(val + 1, 10)) if val % d == 0]
            if not divisors:
                op = '+'
                n = random.randint(1, 15)
            else:
                n = random.choice(divisors)
        elif op == '-':
            n = random.randint(1, max(min(val, 20), 1))
        else:
            n = random.randint(1, 20)

        expr += f" {op} {n}"
        if op == '+':
            val += n
        elif op == '-':
            val -= n
        elif op == '*':
            val *= n
        elif op == '//':
            val //= n
        val = max(val, 1)  # prevent zero/negative values

    return f"Calculate: {expr} = ", str(val), difficulty


# ============================================================
# 2. Grid Mazes
# ============================================================

def gen_maze(difficulty=1):
    """Generate a small text maze. Difficulty controls grid size."""
    sizes = {1: 3, 2: 5, 3: 7, 4: 9, 5: 11}
    size = sizes.get(difficulty, 5)

    # Generate maze using simple random DFS
    grid = [['#'] * size for _ in range(size)]

    # Carve paths
    def carve(x, y):
        grid[y][x] = '.'
        dirs = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and grid[ny][nx] == '#':
                grid[y + dy // 2][x + dx // 2] = '.'
                carve(nx, ny)

    # Start from (1,1)
    carve(1, 1)

    # Place start and end
    grid[1][1] = 'S'
    # Find furthest reachable cell for end
    end_x, end_y = 1, 1
    for y in range(size - 2, 0, -1):
        for x in range(size - 2, 0, -1):
            if grid[y][x] == '.':
                end_x, end_y = x, y
                break
        if end_x != 1 or end_y != 1:
            break
    grid[end_y][end_x] = 'E'

    # BFS to find shortest path
    from collections import deque
    q = deque([(1, 1, [(1, 1)])])
    visited = {(1, 1)}
    path = None
    while q:
        x, y, p = q.popleft()
        if (x, y) == (end_x, end_y):
            path = p
            break
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited and grid[ny][nx] != '#':
                visited.add((nx, ny))
                q.append((nx, ny, p + [(nx, ny)]))

    maze_str = '\n'.join(''.join(row) for row in grid)
    if path:
        directions = []
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            if dx == 1: directions.append('R')
            elif dx == -1: directions.append('L')
            elif dy == 1: directions.append('D')
            elif dy == -1: directions.append('U')
        answer = ''.join(directions)
    else:
        answer = "no path"

    prompt = f"Find the shortest path from S to E. Output directions (U/D/L/R):\n{maze_str}\nPath: "
    return prompt, answer, difficulty


# ============================================================
# 3. Logic Chains
# ============================================================

def gen_logic(difficulty=1):
    """Generate transitive logic chain. Difficulty = chain length."""
    n = difficulty + 2  # 3 items for easy, 7 for hard
    items = random.sample(string.ascii_uppercase[:15], n)

    # Create ordered chain
    chain = items[:]
    random.shuffle(chain)

    # Generate premises (shuffled order)
    premises = []
    for i in range(len(chain) - 1):
        premises.append(f"{chain[i]} > {chain[i + 1]}")
    random.shuffle(premises)

    # Ask about first vs last (requires full chain traversal)
    q_a, q_b = chain[0], chain[-1]
    answer = "Yes"

    # Sometimes ask about reverse
    if random.random() < 0.3:
        q_a, q_b = chain[-1], chain[0]
        answer = "No"

    # Sometimes ask about middle elements
    if random.random() < 0.2 and n >= 4:
        i, j = sorted(random.sample(range(n), 2))
        q_a, q_b = chain[i], chain[j]
        answer = "Yes"

    premise_str = '. '.join(premises)
    prompt = f"{premise_str}. Is {q_a} > {q_b}? Answer Yes or No: "
    return prompt, answer, difficulty


# ============================================================
# 4. Counting/Tracking
# ============================================================

def gen_counting(difficulty=1):
    """Track items through additions and removals."""
    n_events = difficulty + 2
    item = random.choice(["people", "apples", "books", "coins", "birds"])
    count = random.randint(3, 10)
    events = [f"Start with {count} {item}"]

    for _ in range(n_events):
        if random.random() < 0.5 and count > 1:
            n = random.randint(1, min(count - 1, 5))
            count -= n
            events.append(f"{n} {item} leave" if item == "people" else f"remove {n}")
        else:
            n = random.randint(1, 6)
            count += n
            events.append(f"{n} {item} arrive" if item == "people" else f"add {n}")

    prompt = f"{'. '.join(events)}. How many {item} now? Answer with just the number: "
    return prompt, str(count), difficulty


# ============================================================
# 5. Pattern Completion
# ============================================================

def gen_pattern(difficulty=1):
    """Numeric pattern completion."""
    patterns = [
        # Easy: arithmetic
        lambda: (random.randint(1, 5), lambda x, i: x + random.randint(2, 7)),
        # Medium: geometric
        lambda: (random.randint(1, 3), lambda x, i: x * random.randint(2, 3)),
        # Hard: alternating operations
        lambda: (random.randint(1, 5), lambda x, i: x + 3 if i % 2 == 0 else x * 2),
    ]

    if difficulty <= 2:
        start, fn = patterns[0]()
    elif difficulty <= 3:
        start, fn = patterns[1]()
    else:
        start, fn = patterns[2]()

    # Generate sequence
    seq = [start]
    # For arithmetic, use consistent step
    step = random.randint(2, 8) if difficulty <= 2 else None
    mult = random.choice([2, 3]) if 2 < difficulty <= 3 else None

    for i in range(difficulty + 3):
        if difficulty <= 2:
            seq.append(seq[-1] + step)
        elif difficulty <= 3:
            seq.append(seq[-1] * mult)
        else:
            if i % 2 == 0:
                seq.append(seq[-1] + 3)
            else:
                seq.append(seq[-1] * 2)

    shown = seq[:-1]
    answer = str(seq[-1])
    prompt = f"Complete the pattern: {', '.join(str(x) for x in shown)}, ? Answer: "
    return prompt, answer, difficulty


# ============================================================
# Master Generator
# ============================================================

GENERATORS = {
    'arithmetic': gen_arithmetic,
    'maze': gen_maze,
    'logic': gen_logic,
    'counting': gen_counting,
    'pattern': gen_pattern,
}


def generate_reasoning_example(difficulty=None):
    """Generate one reasoning example with tagged difficulty."""
    task = random.choice(list(GENERATORS.keys()))
    if difficulty is None:
        difficulty = random.randint(1, 5)
    gen_fn = GENERATORS[task]

    try:
        prompt, answer, diff = gen_fn(difficulty)
        return {
            'task': task,
            'difficulty': diff,
            'prompt': prompt,
            'answer': answer,
            'text': f"{prompt}{answer}",
        }
    except Exception:
        # Fallback to arithmetic
        prompt, answer, diff = gen_arithmetic(max(1, difficulty))
        return {
            'task': 'arithmetic',
            'difficulty': diff,
            'prompt': prompt,
            'answer': answer,
            'text': f"{prompt}{answer}",
        }


def stream_reasoning_data(tokenizer, seq_len, batch_size):
    """Yield tokenized batches of reasoning data."""
    import torch
    token_buffer = []

    while True:
        # Mix difficulties: 30% easy (1-2), 40% medium (3), 30% hard (4-5)
        r = random.random()
        if r < 0.30:
            diff = random.randint(1, 2)
        elif r < 0.70:
            diff = 3
        else:
            diff = random.randint(4, 5)

        ex = generate_reasoning_example(diff)
        tokens = tokenizer.encode(ex['text'], add_special_tokens=False,
                                  truncation=True, max_length=seq_len * 2)
        tokens.append(tokenizer.eos_token_id)
        token_buffer.extend(tokens)

        while len(token_buffer) >= (seq_len + 1) * batch_size:
            batch = []
            for _ in range(batch_size):
                chunk = token_buffer[:seq_len + 1]
                token_buffer = token_buffer[seq_len:]
                batch.append(chunk)
            t = torch.tensor(batch, dtype=torch.long)
            yield t[:, :-1], t[:, 1:]


if __name__ == '__main__':
    # Demo
    for diff in range(1, 6):
        print(f'\n=== Difficulty {diff} ===')
        for task in GENERATORS:
            ex = generate_reasoning_example(diff)
            if ex['task'] == task:
                print(f'  [{task}] {ex["prompt"][:80]}... → {ex["answer"]}')
            else:
                prompt, answer, _ = GENERATORS[task](diff)
                print(f'  [{task}] {prompt[:80]}... → {answer}')

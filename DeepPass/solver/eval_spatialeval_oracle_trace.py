"""
Oracle Trace Control — THE critical diagnostic experiment.

Give the frozen decoder a perfect BFS reachability trace in the prompt.
If accuracy jumps to 90%+, the decoder CAN reason from structured spatial data,
and the bottleneck is the solver's inability to provide structured info.
If still ~72%, the decoder truly has a reasoning ceiling.

This requires NO training — just BFS computation + prompt modification.

Tests:
  A. baseline: original prompt only
  B. reachable_set: append "Reachable cells from S: (r,c), ..." to prompt
  C. option_reachability: append "Option A (r,c): reachable/unreachable" for each option
  D. adjacency: convert maze to coordinate adjacency list
  E. full_trace: BFS frontier-by-frontier trace
"""
import os, sys, torch, json, random, re, time, argparse
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/spatialeval'


def load_maze_nav():
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze = [s for s in ds if s['id'].startswith('mazenav')]
    print(f'Loaded {len(maze)} Maze-Nav samples', flush=True)
    return maze


def parse_maze_grid(text):
    """Extract the ASCII maze grid from the text description."""
    lines = text.strip().split('\n')
    grid_lines = []
    for line in lines:
        stripped = line.strip()
        # Maze lines contain only #, spaces, ., S, E, X and similar chars
        if stripped and all(c in '#. SEX|' for c in stripped):
            grid_lines.append(stripped)
        elif stripped and len(stripped) > 3 and stripped[0] == '#' and stripped[-1] == '#':
            grid_lines.append(stripped)

    if not grid_lines:
        # Try another pattern: lines starting with #
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') and len(stripped) >= 5:
                grid_lines.append(stripped)

    return grid_lines


def find_cell(grid, char):
    """Find position of a character in the grid."""
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == char:
                return (r, c)
    return None


def bfs_reachable(grid, start):
    """BFS from start, return set of reachable (r,c) positions."""
    if not start:
        return set()
    rows, cols = len(grid), max(len(row) for row in grid)
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < len(grid[nr]) and (nr, nc) not in visited:
                ch = grid[nr][nc]
                if ch != '#':  # Not a wall
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    return visited


def bfs_frontiers(grid, start):
    """BFS returning frontier sets at each distance."""
    if not start:
        return []
    rows = len(grid)
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)
    frontiers = {}

    while queue:
        (r, c), dist = queue.popleft()
        if dist not in frontiers:
            frontiers[dist] = []
        frontiers[dist].append((r, c))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < len(grid[nr]) and (nr, nc) not in visited:
                if grid[nr][nc] != '#':
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist+1))

    return frontiers


def extract_options(text, grid):
    """Try to extract option coordinates from the text."""
    options = {}
    # Pattern: "A) (row, col)" or "A. (row, col)" or similar
    for match in re.finditer(r'([A-D])[).:\s]+\(?(\d+)\s*,\s*(\d+)\)?', text):
        letter, r, c = match.group(1), int(match.group(2)), int(match.group(3))
        options[letter] = (r, c)

    if not options:
        # Try "cell (row,col)" pattern
        for match in re.finditer(r'([A-D])[).\s:]+.*?cell\s*\(?(\d+)\s*,\s*(\d+)\)?', text, re.IGNORECASE):
            letter, r, c = match.group(1), int(match.group(2)), int(match.group(3))
            options[letter] = (r, c)

    if not options:
        # Try row X, column Y pattern
        for match in re.finditer(r'([A-D])[).\s:]+.*?row\s+(\d+).*?col(?:umn)?\s+(\d+)', text, re.IGNORECASE):
            letter, r, c = match.group(1), int(match.group(2)), int(match.group(3))
            options[letter] = (r, c)

    return options


def build_augmented_prompt(text, mode, grid, start, reachable, frontiers, options):
    """Build prompt with oracle spatial information appended."""
    base_prompt = text + "\nAnswer:"

    if mode == 'baseline':
        return base_prompt

    elif mode == 'reachable_set':
        cells = sorted(reachable)
        cell_str = ', '.join(f'({r},{c})' for r, c in cells[:50])  # Limit length
        hint = f"\n\n[Spatial hint: Cells reachable from S: {cell_str}]"
        return text + hint + "\nAnswer:"

    elif mode == 'option_reachability':
        if not options:
            return base_prompt  # Can't help without option coordinates
        hints = []
        for letter in sorted(options.keys()):
            coord = options[letter]
            status = "REACHABLE" if coord in reachable else "NOT reachable"
            hints.append(f"Option {letter} at ({coord[0]},{coord[1]}): {status}")
        hint = "\n\n[Spatial analysis: " + "; ".join(hints) + "]"
        return text + hint + "\nAnswer:"

    elif mode == 'adjacency':
        # Convert to adjacency list
        rows = len(grid)
        adj = []
        for r in range(rows):
            for c in range(len(grid[r])):
                if grid[r][c] != '#':
                    neighbors = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < len(grid[nr]) and grid[nr][nc] != '#':
                            neighbors.append(f'({nr},{nc})')
                    if neighbors:
                        adj.append(f'({r},{c})->[{",".join(neighbors)}]')
        adj_str = '; '.join(adj[:40])  # Limit
        hint = f"\n\n[Graph: {adj_str}]"
        return text + hint + "\nAnswer:"

    elif mode == 'full_trace':
        trace_parts = []
        for dist in sorted(frontiers.keys()):
            cells = frontiers[dist]
            cell_str = ','.join(f'({r},{c})' for r, c in cells)
            trace_parts.append(f'd={dist}: {cell_str}')
        trace_str = ' | '.join(trace_parts[:15])  # Limit
        if options:
            opt_status = []
            for letter in sorted(options.keys()):
                coord = options[letter]
                status = "YES" if coord in reachable else "NO"
                opt_status.append(f'{letter}={status}')
            trace_str += f' || Options: {", ".join(opt_status)}'
        hint = f"\n\n[BFS trace from S: {trace_str}]"
        return text + hint + "\nAnswer:"

    return base_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', type=str,
                        default='baseline,reachable_set,option_reachability,full_trace')
    args = parser.parse_args()
    modes = args.modes.split(',')

    print('Loading model...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False

    maze_data = load_maze_nav()
    random.seed(0)
    indices = list(range(len(maze_data)))
    random.shuffle(indices)
    eval_idx = indices[1000:]

    t0 = time.time()
    results = {}
    parse_stats = {'grid_found': 0, 'start_found': 0, 'options_found': 0}

    for mode in modes:
        print(f'\n=== Mode: {mode} ===', flush=True)
        correct = 0
        n_eval = len(eval_idx)
        parse_failures = 0

        for ei, idx in enumerate(eval_idx):
            sample = maze_data[idx]
            text = sample['text'][:1500]
            oracle = sample['oracle_option'].strip().upper()

            # Parse maze
            grid = parse_maze_grid(text)
            start = find_cell(grid, 'S') if grid else None
            reachable = bfs_reachable(grid, start) if start else set()
            frontiers = bfs_frontiers(grid, start) if start else {}
            options = extract_options(text, grid)

            if ei == 0:
                # Debug: show first sample's parsing
                print(f'  Grid lines: {len(grid)}', flush=True)
                print(f'  Start: {start}', flush=True)
                print(f'  Reachable cells: {len(reachable)}', flush=True)
                print(f'  Options found: {options}', flush=True)
                if grid:
                    for line in grid[:5]:
                        print(f'    {line}', flush=True)

            if not grid or not start:
                parse_failures += 1

            if grid: parse_stats['grid_found'] += 1
            if start: parse_stats['start_found'] += 1
            if options: parse_stats['options_found'] += 1

            # Build augmented prompt
            prompt = build_augmented_prompt(text, mode, grid, start, reachable, frontiers, options)

            # Run decoder
            enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                out = base_model.generate(enc['input_ids'], max_new_tokens=5, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
                answer = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()

            if oracle in answer.upper()[:10]:
                correct += 1

            if (ei + 1) % 100 == 0:
                print(f'  {ei+1}/{n_eval} | acc={correct/(ei+1):.3f} | parse_fail={parse_failures}', flush=True)

        acc = correct / n_eval
        results[mode] = {
            'accuracy': acc, 'correct': correct, 'total': n_eval,
            'parse_failures': parse_failures,
        }
        print(f'  FINAL: {mode} = {acc:.4f} ({correct}/{n_eval}), parse_fails={parse_failures}', flush=True)

    # Summary
    print(f'\n=== SUMMARY ===', flush=True)
    for mode, r in results.items():
        print(f'  {mode}: {r["accuracy"]:.4f} ({r["correct"]}/{r["total"]})', flush=True)

    print(f'\nParse stats (over all modes): grid={parse_stats["grid_found"]}, '
          f'start={parse_stats["start_found"]}, options={parse_stats["options_found"]}', flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': 'oracle_trace_control',
        'method': 'oracle_trace',
        'modes': modes,
        'results': results,
        'parse_stats': parse_stats,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'spatialeval_oracle_trace.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'\nSaved: spatialeval_oracle_trace.json ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()

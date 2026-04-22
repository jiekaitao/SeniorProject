"""
SpatialEval Mazenav surgery loaders — falsification tests for the
"Mazenav is secretly pattern-matching because the path is pre-marked with X"
hypothesis.

Three variants:
  - load_mazenav_no_x:      remove all X markers (replace with space)
  - load_mazenav_distract:  replace X with random decoys from {*, +, o}
  - load_mazenav_unsolved:  remove X AND rewrite the question so the model
                            must plan the path, not count direction changes

All three preserve the original oracle_option (correct answer): the puzzle gets
harder but the right answer doesn't change. The transformation only touches
lines inside the ASCII maze block — the surrounding legend/question/options
are left intact (the "X marks the specific route..." legend line still
mentions X, but since there are no X characters in the maze, the model is on
its own).

Train/eval split logic matches mega_runner.py: random.seed(0), shuffle all
mazenav items, take first 1000 as train, rest as eval. Returns formatted dict
{'train': [...], 'test': [...]} and n_choices=4.

Each formatted sample has keys: text, oracle_option, oracle_answer,
oracle_full_answer, id, label, n_choices. This keeps both the 'oracle_option'
convention (used by mega_runner.py / mega_runner_lora_spatial.py) and the
'label' convention (used by mega_runner_benchmarks.py) so either trainer can
consume the output.
"""
import os
import re
import sys
import random


CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


# --------------------------------------------------------------------------
# Core transform helpers
# --------------------------------------------------------------------------

_MAZE_BLOCK_RE = re.compile(
    r"(ASCII code:\n)(#+\n(?:[^\n]*\n)*?#+)(?=,?\s*where the symbols)",
    re.DOTALL,
)


def _is_maze_line(line):
    """A maze row is made only of '#', space, and the letters S/E/X (and
    possibly decoys). Return True if this line is part of the ASCII maze."""
    if not line:
        return False
    stripped = line.strip()
    if not stripped:
        return False
    # Maze rows must start and end with '#', and contain only the allowed
    # alphabet.
    if not (stripped.startswith('#') and stripped.endswith('#')):
        return False
    allowed = set('# SEX*+o')
    return all(c in allowed for c in stripped)


def _transform_maze_block(text, char_chooser):
    """Find the ASCII maze block inside `text` and apply `char_chooser` to
    every 'X' character inside it. `char_chooser()` returns the replacement
    for each X. The rest of `text` (question, options, legend) is untouched.

    Uses a regex to bracket the maze between `ASCII code:\n` and `, where
    the symbols`. Falls back to a line-based scan that only rewrites lines
    that pass `_is_maze_line`.
    """
    m = _MAZE_BLOCK_RE.search(text)
    if m is not None:
        prefix = m.group(1)
        maze_block = m.group(2)
        new_block_chars = []
        for c in maze_block:
            new_block_chars.append(char_chooser() if c == 'X' else c)
        new_block = ''.join(new_block_chars)
        return text[: m.start()] + prefix + new_block + text[m.end():]

    # Fallback: line-based scan
    out_lines = []
    in_maze = False
    for line in text.split('\n'):
        if _is_maze_line(line):
            in_maze = True
            out_lines.append(''.join(
                (char_chooser() if c == 'X' else c) for c in line
            ))
        else:
            if in_maze:
                in_maze = False
            out_lines.append(line)
    return '\n'.join(out_lines)


# --------------------------------------------------------------------------
# Public loaders
# --------------------------------------------------------------------------

def _load_mazenav_raw():
    os.environ.setdefault('HF_HOME', '/blue/cis4914/jietao/hf_cache')
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    items = [s for s in ds if s['id'].startswith('mazenav')]
    return items


def _split_indices(n_items, seed=0):
    random.seed(seed)
    idx = list(range(n_items))
    random.shuffle(idx)
    split = min(1000, n_items * 2 // 3)
    return idx[:split], idx[split:]


def _format_sample(raw, transformed_text):
    """Build a dict that's compatible with BOTH training loops."""
    oracle = raw['oracle_option'].strip().upper()
    label = CHOICE_MAP.get(oracle[:1], 0)
    return {
        'id': raw['id'],
        'text': transformed_text,
        'oracle_option': raw['oracle_option'],
        'oracle_answer': raw['oracle_answer'],
        'oracle_full_answer': raw['oracle_full_answer'],
        'label': label,
        'n_choices': 4,
    }


def _build_split(items, transform_fn, seed):
    train_idx, eval_idx = _split_indices(len(items), seed=seed)
    train = [_format_sample(items[i], transform_fn(items[i]['text'])) for i in train_idx]
    test = [_format_sample(items[i], transform_fn(items[i]['text'])) for i in eval_idx]
    return {'train': train, 'test': test}


def _print_sample(tag, sample):
    sys.stderr.write(f'\n[spatialeval_surgery] {tag} sample (id={sample["id"]}, oracle={sample["oracle_option"]})\n')
    sys.stderr.write('-' * 60 + '\n')
    sys.stderr.write(sample['text'] + '\n')
    sys.stderr.write('-' * 60 + '\n')
    sys.stderr.flush()


def load_mazenav_no_x(seed=0):
    """Mazenav with every 'X' inside the maze block replaced by a space.

    The legend still says "X marks the specific route..." but no X exists in
    the picture — the model must plan from scratch.
    """
    items = _load_mazenav_raw()

    def transform(text):
        return _transform_maze_block(text, lambda: ' ')

    data = _build_split(items, transform, seed)
    if data['train']:
        _print_sample('load_mazenav_no_x', data['train'][0])
    print(f'  mazenav_no_x: {len(data["train"])} train, {len(data["test"])} eval', flush=True)
    return data, 4


def load_mazenav_distract(seed=0):
    """Mazenav with every 'X' inside the maze block replaced by a random
    decoy from {*, +, o}. The model sees 'path-like' structure but it's
    not the true path (each X independently gets a random pick from three
    characters, which breaks any 'follow-the-X' shortcut).

    Random choices are seeded by `seed` so the transform is deterministic.
    """
    items = _load_mazenav_raw()

    rng = random.Random(seed + 1)
    decoys = ['*', '+', 'o']

    def transform(text):
        return _transform_maze_block(text, lambda: rng.choice(decoys))

    data = _build_split(items, transform, seed)
    if data['train']:
        _print_sample('load_mazenav_distract', data['train'][0])
    print(f'  mazenav_distract: {len(data["train"])} train, {len(data["test"])} eval', flush=True)
    return data, 4


# Precompiled regex for rewriting the question body on the unsolved variant.
# Matches either "right turns" or "total turns" framings; both are about
# counting direction changes along a path that, without X, the model must
# compute itself.
_QUESTION_RE = re.compile(
    r"How many (?:right|total) turns are there in the provided path "
    r"\(marked by X\) from S to E\?"
)


def load_mazenav_unsolved(seed=0):
    """Hardest variant: no X markers AND the question is rewritten to ask
    for the MINIMUM number of right turns to navigate from S to E. The
    model must actually plan the path, not count direction changes in a
    pre-marked solution.

    Only the subset of mazenav items that ask about turn counts along the
    marked path are rewritten; spatial-position questions ("Is E to the
    left of S") pass through unchanged since their answer does not depend
    on the X markers. All items still get X removed from the maze block.

    NOTE: the answer choices (and oracle) are preserved from the original
    item. Under this rewrite, the oracle is only correct when the pre-marked
    path was already the minimum-right-turn path. We keep oracle_option
    unchanged per spec — the surgery tests how the controller vs LoRA
    compare on a harder framing of the same task.
    """
    items = _load_mazenav_raw()

    def transform(text):
        no_x = _transform_maze_block(text, lambda: ' ')
        rewritten = _QUESTION_RE.sub(
            "What is the minimum number of right turns required to "
            "navigate from S to E?",
            no_x,
        )
        return rewritten

    data = _build_split(items, transform, seed)
    if data['train']:
        _print_sample('load_mazenav_unsolved', data['train'][0])
    print(f'  mazenav_unsolved: {len(data["train"])} train, {len(data["test"])} eval', flush=True)
    return data, 4


SURGERY_LOADERS = {
    'no_x': load_mazenav_no_x,
    'distract': load_mazenav_distract,
    'unsolved': load_mazenav_unsolved,
}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list(SURGERY_LOADERS), default='no_x')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    d, n = SURGERY_LOADERS[args.variant](seed=args.seed)
    print(f'\n=== {args.variant}: {len(d["train"])} train / {len(d["test"])} test / {n} choices ===')
    print(d['train'][0]['text'])

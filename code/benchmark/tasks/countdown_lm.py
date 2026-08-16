"""Countdown task for real language models.

Same task as the EGGROLL paper and our TinyPolicy benchmark: given N numbers,
combine them with +, -, * to reach a target. But formatted as natural language
prompts for a pretrained LM, not as token sequences for a from-scratch policy.

The 4-number version (EGGROLL paper default) has many valid solutions per problem,
so pass@k is meaningful and diversity collapse is detectable.

Reward is sparse, binary, and verifiable: 1 if the expression is valid and
evaluates to the target, 0 otherwise. No partial credit.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch


@dataclass
class CountdownProblem:
    numbers: list[int]
    target: int
    solutions: list[str] | None = None   # for SFT data


def generate_problems(n: int, seed: int = 0, n_numbers: int = 4,
                      max_num: int = 25, max_target: int = 100) -> list[CountdownProblem]:
    """Generate countdown problems with known solutions.

    Each problem has n_numbers integers and a target. We generate the target
    by randomly applying operations to the numbers, guaranteeing at least one
    solution exists.
    """
    rng = random.Random(seed)
    problems = []
    ops = ['+', '-', '*']

    for _ in range(n):
        nums = [rng.randint(1, max_num) for _ in range(n_numbers)]
        # Generate a target by applying random operations
        # Pick a random subset and order of the numbers
        perm = list(range(n_numbers))
        rng.shuffle(perm)
        # Build an expression from the shuffled numbers
        result = nums[perm[0]]
        expr_parts = [str(nums[perm[0]])]
        for i in range(1, n_numbers):
            op = rng.choice(ops)
            val = nums[perm[i]]
            if op == '+':
                result = result + val
            elif op == '-':
                result = result - val
            else:
                result = result * val
            expr_parts.append(op)
            expr_parts.append(str(val))

        # Clamp target to reasonable range
        if abs(result) > max_target * 10:
            result = rng.randint(1, max_target)

        target = result
        solution = ' '.join(expr_parts)
        problems.append(CountdownProblem(
            numbers=nums, target=target,
            solutions=[solution],
        ))

    return problems


def format_prompt(problem: CountdownProblem) -> str:
    """Format a problem as a prompt for a language model."""
    nums = ', '.join(str(n) for n in problem.numbers)
    return (f"Use the numbers {nums} with operations +, -, * to make {problem.target}. "
            f"Write ONLY the expression, e.g. '3 + 5 * 2 - 1'. "
            f"Answer: ")


def format_sft_example(problem: CountdownProblem) -> str:
    """Format a problem + solution for SFT training."""
    prompt = format_prompt(problem)
    solution = problem.solutions[0] if problem.solutions else ""
    return prompt + solution


def verify_answer(problem: CountdownProblem, completion: str) -> bool:
    """Check if a completion is a valid solution to the problem.

    Parses the expression, evaluates left-to-right (no precedence), and checks:
    1. Uses valid operations (+, -, *)
    2. All numbers used are from the problem (with replacement counting)
    3. Result equals the target
    """
    # Extract the expression — take the first line, strip whitespace
    expr = completion.strip().split('\n')[0].strip()
    # Remove any trailing punctuation or explanation
    expr = re.sub(r'[.!?].*$', '', expr).strip()

    try:
        # Tokenise: numbers and operators
        tokens = re.findall(r'-?\d+|[+\-*]', expr)
        if not tokens:
            return False

        # Extract numbers and operators
        nums_used = []
        operators = []
        expecting_num = True
        for tok in tokens:
            if expecting_num:
                try:
                    nums_used.append(int(tok))
                    expecting_num = False
                except ValueError:
                    return False
            else:
                if tok in ('+', '-', '*'):
                    operators.append(tok)
                    expecting_num = True
                else:
                    return False

        if not nums_used or expecting_num:
            return False

        # Check that the used numbers are a subset of the available numbers
        available = sorted(problem.numbers)
        used = sorted(nums_used)
        # Allow using each number at most once
        avail_copy = list(available)
        for n in used:
            if n in avail_copy:
                avail_copy.remove(n)
            else:
                return False

        # Evaluate left-to-right (no precedence)
        result = nums_used[0]
        for i, op in enumerate(operators):
            if i + 1 >= len(nums_used):
                return False
            val = nums_used[i + 1]
            if op == '+':
                result += val
            elif op == '-':
                result -= val
            elif op == '*':
                result *= val

        return result == problem.target

    except (ValueError, IndexError):
        return False


class CountdownLMTask:
    """Countdown task manager for language model benchmarking."""

    def __init__(self, n_train: int = 32, n_eval: int = 256, seed: int = 0,
                 n_numbers: int = 4):
        self.n_numbers = n_numbers
        self.train_problems = generate_problems(n_train * 10, seed=seed,
                                                n_numbers=n_numbers)
        self.eval_problems = generate_problems(n_eval, seed=99999,
                                               n_numbers=n_numbers)

    def train_batch(self, gen_seed: int, batch_size: int) -> list[CountdownProblem]:
        """Deterministic problem batch for one generation (CRN)."""
        rng = random.Random(gen_seed)
        return rng.sample(self.train_problems, min(batch_size, len(self.train_problems)))

    def eval_batch(self) -> list[CountdownProblem]:
        return self.eval_problems

    def format_prompts(self, problems: list[CountdownProblem]) -> list[str]:
        return [format_prompt(p) for p in problems]

    def score(self, problems: list[CountdownProblem],
              completions: list[str]) -> torch.Tensor:
        """Score completions. Returns (B,) float32 binary rewards."""
        rewards = [1.0 if verify_answer(p, c) else 0.0
                   for p, c in zip(problems, completions)]
        return torch.tensor(rewards, dtype=torch.float32)

    def sft_data(self, n: int, seed: int = 0) -> list[dict[str, str]]:
        """Generate SFT training data: prompt + correct completion."""
        problems = generate_problems(n, seed=seed + 42, n_numbers=self.n_numbers)
        data = []
        for p in problems:
            prompt = format_prompt(p)
            answer = p.solutions[0] if p.solutions else ""
            data.append({"prompt": prompt, "completion": answer})
        return data

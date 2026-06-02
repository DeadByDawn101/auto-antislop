"""
antislop_mlx.py — Anti-Slop Sampler for MLX

Port of the anti-slop concept to MLX. Works at the TOKEN LEVEL during
inference to detect and suppress repetitive patterns BEFORE they happen.

Three mechanisms:
1. N-gram repetition penalty: tracks generated n-grams, penalizes tokens
   that would create a repeated n-gram
2. Phrase-level detection: detects repeated phrases and suppresses continuation
3. Entropy monitoring: detects low-entropy (repetitive) generation and
   increases temperature dynamically

Usage:
    from antislop_mlx import AntislopSampler, generate_antislop

    # Simple generation with anti-slop
    response = generate_antislop(model, tokenizer, prompt, max_tokens=4096)

    # Or use the sampler directly
    sampler = AntislopSampler(
        ngram_size=6,           # detect 6-gram repetitions
        penalty=5.0,            # logit penalty for repeated n-grams
        window_size=256,        # look back 256 tokens for repetitions
        entropy_threshold=0.5,  # boost temp when entropy drops below this
        max_temp_boost=0.5,     # max temperature increase
    )

Author: RavenX LLC / DeadByDawn101
License: MIT
"""

from collections import defaultdict
from typing import Optional, List, Tuple
import math

import mlx.core as mx


class AntislopSampler:
    """Anti-slop sampler that prevents repetitive generation at the token level."""

    def __init__(
        self,
        ngram_size: int = 6,
        penalty: float = 5.0,
        window_size: int = 256,
        entropy_threshold: float = 0.5,
        max_temp_boost: float = 0.5,
        phrase_penalty: float = 10.0,
        min_phrase_len: int = 4,
        banned_phrases: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        self.ngram_size = ngram_size
        self.penalty = penalty
        self.window_size = window_size
        self.entropy_threshold = entropy_threshold
        self.max_temp_boost = max_temp_boost
        self.phrase_penalty = phrase_penalty
        self.min_phrase_len = min_phrase_len
        self.banned_phrases = banned_phrases or []
        self.verbose = verbose

        # State
        self.generated_tokens: List[int] = []
        self.ngram_counts: dict = defaultdict(int)
        self.repetition_warnings: int = 0
        self.temp_boost: float = 0.0

    def reset(self):
        """Reset state for new generation."""
        self.generated_tokens = []
        self.ngram_counts = defaultdict(int)
        self.repetition_warnings = 0
        self.temp_boost = 0.0

    def _get_recent_ngrams(self, n: int) -> set:
        """Get all n-grams in the recent window."""
        tokens = self.generated_tokens[-self.window_size:]
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.add(ngram)
        return ngrams

    def _count_ngram_repetitions(self, candidate_token: int) -> int:
        """Count how many times the n-gram ending with candidate_token appears."""
        if len(self.generated_tokens) < self.ngram_size - 1:
            return 0

        # Build the n-gram that would be created
        prefix = tuple(self.generated_tokens[-(self.ngram_size - 1):])
        candidate_ngram = prefix + (candidate_token,)

        # Count in recent window
        tokens = self.generated_tokens[-self.window_size:]
        count = 0
        for i in range(len(tokens) - self.ngram_size + 1):
            if tuple(tokens[i:i + self.ngram_size]) == candidate_ngram:
                count += 1

        return count

    def _detect_phrase_loop(self) -> bool:
        """Detect if we're in a phrase-level repetition loop."""
        if len(self.generated_tokens) < self.min_phrase_len * 3:
            return False

        recent = self.generated_tokens[-self.window_size:]

        # Check for repeating patterns of various lengths
        for phrase_len in range(self.min_phrase_len, min(64, len(recent) // 2)):
            last_phrase = tuple(recent[-phrase_len:])
            prev_phrase = tuple(recent[-2 * phrase_len:-phrase_len])
            if last_phrase == prev_phrase:
                if self.verbose:
                    print(f"[antislop] Phrase loop detected: {phrase_len}-token pattern repeating")
                return True

        return False

    def _compute_entropy(self, logits: mx.array) -> float:
        """Compute entropy of the logit distribution."""
        probs = mx.softmax(logits, axis=-1)
        # Avoid log(0)
        log_probs = mx.log(mx.clip(probs, 1e-10, None))
        entropy = -mx.sum(probs * log_probs).item()
        return entropy

    def apply(self, logits: mx.array, temperature: float = 1.0) -> Tuple[mx.array, float]:
        """
        Apply anti-slop penalties to logits.

        Args:
            logits: raw logits from model (vocab_size,)
            temperature: current temperature

        Returns:
            (modified_logits, adjusted_temperature)
        """
        vocab_size = logits.shape[-1]

        # 1. N-gram repetition penalty
        if len(self.generated_tokens) >= self.ngram_size - 1:
            # Build penalty mask for all tokens
            prefix = tuple(self.generated_tokens[-(self.ngram_size - 1):])
            tokens_window = self.generated_tokens[-self.window_size:]

            # Find all n-grams starting with our prefix
            penalty_tokens = []
            for i in range(len(tokens_window) - self.ngram_size + 1):
                ngram = tuple(tokens_window[i:i + self.ngram_size])
                if ngram[:self.ngram_size - 1] == prefix:
                    # The last token of this n-gram would create a repeat
                    penalty_tokens.append(ngram[-1])

            # Apply penalty
            if penalty_tokens:
                for tok in set(penalty_tokens):
                    if tok < vocab_size:
                        logits = logits.at[tok].add(-self.penalty)

        # 2. Phrase loop detection
        if self._detect_phrase_loop():
            self.repetition_warnings += 1
            # Increase temperature to break out of loop
            self.temp_boost = min(self.temp_boost + 0.1, self.max_temp_boost)

            # Heavy penalty on recent tokens
            recent_tokens = set(self.generated_tokens[-32:])
            for tok in recent_tokens:
                if tok < vocab_size:
                    logits = logits.at[tok].add(-self.phrase_penalty)

            if self.verbose:
                print(f"[antislop] Loop detected! Temp boost: +{self.temp_boost:.1f}, "
                      f"warnings: {self.repetition_warnings}")
        else:
            # Decay temp boost when not looping
            self.temp_boost = max(0, self.temp_boost - 0.02)

        # 3. Entropy monitoring
        entropy = self._compute_entropy(logits)
        if entropy < self.entropy_threshold:
            # Low entropy = model is very confident = potential repetition
            self.temp_boost = min(self.temp_boost + 0.05, self.max_temp_boost)
            if self.verbose:
                print(f"[antislop] Low entropy: {entropy:.2f}, boosting temp +{self.temp_boost:.1f}")

        adjusted_temp = temperature + self.temp_boost

        return logits, adjusted_temp

    def record_token(self, token_id: int):
        """Record a generated token for tracking."""
        self.generated_tokens.append(token_id)

    def get_stats(self) -> dict:
        """Return generation statistics."""
        return {
            "tokens_generated": len(self.generated_tokens),
            "repetition_warnings": self.repetition_warnings,
            "current_temp_boost": self.temp_boost,
            "unique_tokens": len(set(self.generated_tokens)),
            "token_diversity": len(set(self.generated_tokens)) / max(1, len(self.generated_tokens)),
        }


def generate_antislop(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.95,
    ngram_size: int = 6,
    penalty: float = 5.0,
    verbose: bool = False,
) -> str:
    """
    Generate text with anti-slop protection.

    Uses mlx-lm model + tokenizer with AntislopSampler wrapping the generation.
    """
    sampler = AntislopSampler(
        ngram_size=ngram_size,
        penalty=penalty,
        verbose=verbose,
    )

    input_ids = mx.array(tokenizer.encode(prompt))[None]
    tokens = input_ids[0].tolist()

    for step in range(max_tokens):
        logits = model(input_ids)
        logits = logits[:, -1, :]  # last token logits

        # Apply anti-slop
        modified_logits, adjusted_temp = sampler.apply(logits[0], temperature)

        # Temperature scaling
        if adjusted_temp > 0:
            modified_logits = modified_logits / adjusted_temp

        # Top-p sampling
        probs = mx.softmax(modified_logits, axis=-1)
        sorted_indices = mx.argsort(-probs)
        sorted_probs = mx.take(probs, sorted_indices)
        cumsum = mx.cumsum(sorted_probs)
        mask = cumsum - sorted_probs <= top_p
        filtered_probs = sorted_probs * mask.astype(mx.float32)
        filtered_probs = filtered_probs / (filtered_probs.sum() + 1e-10)

        # Sample
        next_token = sorted_indices[mx.random.categorical(mx.log(filtered_probs + 1e-10))].item()

        # Record and check for EOS
        sampler.record_token(next_token)
        tokens.append(next_token)
        input_ids = mx.array([[next_token]])

        if next_token == tokenizer.eos_token_id:
            break

        # Hard stop: if too many repetition warnings, force stop
        if sampler.repetition_warnings > 10:
            if verbose:
                print(f"[antislop] Force stop: {sampler.repetition_warnings} repetition warnings")
            break

        mx.eval(input_ids)

    # Decode
    output_tokens = tokens[len(tokenizer.encode(prompt)):]
    result = tokenizer.decode(output_tokens)

    if verbose:
        stats = sampler.get_stats()
        print(f"\n[antislop] Stats: {stats}")

    return result

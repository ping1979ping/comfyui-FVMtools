"""K2 Lab — Bindung von Prompt-Zeichenbereichen an Konditionierungs-Tokenindizes.

Der Krea-2-Textencoder (Qwen3-VL-4B) liefert Konditionierung, aus der ComfyUI das
System-/User-Präfix entfernt. Die Prompt-Token stehen dadurch ab Index 0. Um zu
wissen, welche Tokenindizes zu welcher Regionsklausel gehören, wird jeweils der
Prompt-*Präfix* tokenisiert und dessen Prompt-Tokenanzahl gezählt.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .prompt import CompiledPlan

IM_START = 151644
IM_END = 151645
NEWLINE = 198
USER = 872
PAD = 151643


def krea_prompt_token_count(tokenized: dict) -> int:
    """Zählt die prompt-eigenen Token in einem ComfyUI-Tokenize-Ergebnis.

    Verträgt sowohl die vollständige Chat-Vorlage als auch Loader, die Teile der
    Vorlage bereits entfernt haben.
    """
    if not tokenized:
        raise ValueError("Krea-Tokenisierung lieferte keine Tokengruppen")

    def ids(batches) -> list:
        if not isinstance(batches, list) or len(batches) != 1:
            return []
        return [pair[0] for pair in batches[0]]

    def is_token(value, token_id: int) -> bool:
        return isinstance(value, Integral) and int(value) == token_id

    candidates = list(tokenized.values())
    batches = next(
        (
            candidate
            for candidate in candidates
            if sum(is_token(token, IM_START) for token in ids(candidate)) >= 2
        ),
        candidates[0],
    )
    if not isinstance(batches, list) or len(batches) != 1:
        raise ValueError(
            "Vereinheitlichtes K2-Prompting braucht genau einen Token-Batch — der "
            "Prompt ist vermutlich zu lang für ein Fenster"
        )
    pairs = batches[0]
    token_ids = [pair[0] for pair in pairs]

    if not any(
        is_token(token, IM_START) or is_token(token, IM_END) for token in token_ids
    ):
        return len(pairs)

    second_start = None
    seen = 0
    for index, token in enumerate(token_ids):
        if is_token(token, IM_START):
            seen += 1
            if seen == 2:
                second_start = index
                break
    if second_start is None:
        raise ValueError("Qwen-Wrapper ohne zweites <|im_start|> — unerwartete Vorlage")

    start = second_start + 1
    if (
        len(pairs) > start + 1
        and is_token(token_ids[start], USER)
        and is_token(token_ids[start + 1], NEWLINE)
    ):
        start += 2
    for index in range(start, len(pairs)):
        if is_token(token_ids[index], IM_END):
            return index - start
        if is_token(token_ids[index], IM_START):
            end = index
            if end > start and is_token(token_ids[end - 1], NEWLINE):
                end -= 1
            return end - start

    end = len(pairs)
    while end > start and is_token(token_ids[end - 1], PAD):
        end -= 1
    return end - start


@dataclass
class TokenSpan:
    region_id: str
    name: str
    start: int
    end: int
    role: str
    field: np.ndarray
    mask: np.ndarray


@dataclass
class EmphasisSpan:
    scope_id: str
    phrase: str
    strength: float
    start: int
    end: int
    field: np.ndarray


@dataclass
class IdentitySpan:
    region_id: str
    start: int
    end: int


@dataclass
class BoundPlan:
    """Kompilierter Plan mit aufgelösten Tokenindizes."""

    plan: CompiledPlan
    text_token_count: int
    image_token_count: int
    spans: tuple[TokenSpan, ...]
    emphases: tuple[EmphasisSpan, ...] = ()
    identity_spans: tuple[IdentitySpan, ...] = ()
    trigger_spans: tuple[IdentitySpan, ...] = ()

    @property
    def sequence_length(self) -> int:
        return self.text_token_count + self.image_token_count

    def span_by_region(self, region_id: str) -> TokenSpan | None:
        return next((s for s in self.spans if s.region_id == region_id), None)

    def summary(self) -> dict:
        return {
            "text_token_count": self.text_token_count,
            "image_token_count": self.image_token_count,
            "regions": [
                {
                    "id": s.region_id,
                    "name": s.name,
                    "text_token_span": [s.start, s.end],
                    "role": s.role,
                    "image_tokens": int((s.mask > 0.0).sum()),
                }
                for s in self.spans
            ],
            "emphases": [
                {
                    "scope_id": e.scope_id,
                    "phrase": e.phrase,
                    "strength": e.strength,
                    "text_token_span": [e.start, e.end],
                }
                for e in self.emphases
            ],
            "identity_spans": [
                {"region_id": i.region_id, "text_token_span": [i.start, i.end]}
                for i in self.identity_spans
            ],
            "trigger_spans": [
                {"region_id": i.region_id, "text_token_span": [i.start, i.end]}
                for i in self.trigger_spans
            ],
        }


def bind_plan(
    plan: CompiledPlan,
    tokenize,
    *,
    conditioning_text_token_count: int | None = None,
) -> BoundPlan:
    """Löst alle Zeichenbereiche des Plans in Tokenindizes auf.

    `tokenize` ist üblicherweise ``clip.tokenize``.
    """
    cache: dict[int, int] = {}

    def prefix_tokens(char_offset: int) -> int:
        if char_offset in cache:
            return cache[char_offset]
        count = krea_prompt_token_count(tokenize(plan.prompt[:char_offset]))
        cache[char_offset] = count
        return count

    total = (
        prefix_tokens(len(plan.prompt))
        if conditioning_text_token_count is None
        else int(conditioning_text_token_count)
    )
    if total <= 0:
        raise ValueError("Krea-Konditionierung enthält keine Texttoken")

    spans = tuple(
        TokenSpan(
            region_id=region.region_id,
            name=region.name,
            start=prefix_tokens(region.char_span[0]),
            end=prefix_tokens(region.char_span[1]),
            role=region.role,
            field=region.field,
            mask=region.mask,
        )
        for region in plan.regions
    )
    for span in spans:
        if span.end <= span.start:
            raise ValueError(
                f"Region {span.name!r} besitzt keine eigenen Texttoken — Prompt zu kurz?"
            )
    if spans and max(s.end for s in spans) > total:
        raise ValueError(
            "Regionale Tokenspanne liegt außerhalb der Konditionierungssequenz "
            f"({max(s.end for s in spans)} > {total})"
        )

    emphases = tuple(
        EmphasisSpan(
            scope_id=e.scope_id,
            phrase=e.phrase,
            strength=e.strength,
            start=prefix_tokens_rstrip(plan.prompt, e.char_span[0], prefix_tokens, tokenize),
            end=prefix_tokens(e.char_span[1]),
            field=e.field,
        )
        for e in plan.emphases
    )
    for emphasis in emphases:
        if emphasis.end <= emphasis.start:
            raise ValueError(
                f"Emphasis-Phrase {emphasis.phrase!r} deckt kein vollständiges Token ab"
            )
    if emphases and max(e.end for e in emphases) > total:
        raise ValueError("Emphasis-Tokenspanne liegt außerhalb der Konditionierung")

    identity_spans = tuple(
        IdentitySpan(
            region_id=region.region_id,
            start=prefix_tokens(region.identity_char_span[0]),
            end=prefix_tokens(region.identity_char_span[1]),
        )
        for region in plan.regions
        if region.identity_char_span is not None
    )
    trigger_spans = tuple(
        IdentitySpan(
            region_id=region.region_id,
            start=prefix_tokens(a),
            end=prefix_tokens(b),
        )
        for region in plan.regions
        for a, b in region.trigger_char_spans
    )
    for identity in (*identity_spans, *trigger_spans):
        if identity.end <= identity.start:
            raise ValueError(
                f"Identity-Span der Region {identity.region_id!r} deckt kein Token ab"
            )
        if identity.end > total:
            raise ValueError("Identity-Span liegt außerhalb der Konditionierung")

    return BoundPlan(
        plan=plan,
        text_token_count=total,
        image_token_count=plan.token_count,
        spans=spans,
        emphases=emphases,
        identity_spans=identity_spans,
        trigger_spans=trigger_spans,
    )


def prefix_tokens_rstrip(prompt: str, offset: int, prefix_tokens, tokenize) -> int:
    """Startindex einer Phrase; führendes Leerzeichen gehört zum Folgetoken.

    Qwen-BPE hängt ein Leerzeichen an das nachfolgende Wort, daher liefert der
    rstrip-Präfix den korrekten Tokenstart.
    """
    stripped = prompt[:offset].rstrip()
    if len(stripped) == offset:
        return prefix_tokens(offset)
    return krea_prompt_token_count(tokenize(stripped))


__all__ = [
    "BoundPlan",
    "EmphasisSpan",
    "IdentitySpan",
    "TokenSpan",
    "bind_plan",
    "krea_prompt_token_count",
]

"""Reality check — ask a vision LLM whether a generated picture holds up.

An image generator fails in a handful of recognisable ways: it renders the
subject twice, melts two bodies into one, grows a third leg, or twists a torso
through an angle no spine allows. This module asks a local vision model about
those failure modes and turns the answers into one verdict.

The design rests on one measured finding: **a small vision model cannot judge,
but it can count and it can name what it sees.** Asked "is this pose
anatomically impossible?" an 8B model answers "no" for every picture, including
a torso rotated 180 degrees. Asked "do you see breasts or shoulder blades?" and
"do you see the groin or the buttock cleft?" it answers both correctly — and the
contradiction between those two answers *is* the impossible pose. So every probe
here asks a perception question with a closed set of answers, and the inference
happens in Python where it can be inspected and tested.

Two-stage verdict, for the same reason. A landmark answer alone produced a false
alarm on a woman sitting with open legs, where the model called the visible
crotch a "buttock cleft". A differently worded confirmation question — a forced
choice between two concrete descriptions — got that picture right, so a twist
suspicion is only upheld when the second, independent wording agrees. See
``tests/live/reality/calibrate.py`` for the acceptance run behind these choices.

Contract, mirroring :mod:`lmstudio_client`:

- **Import touches no network** and needs neither torch nor ComfyUI, so the
  calibration harness and the unit tests can import it directly.
- **Every probe degrades gracefully.** An unreachable server, an HTTP error or
  an unparsable answer yields a probe result with ``ok=False`` that abstains
  from the verdict instead of failing the run.

Entry point is :func:`assess`.
"""

from __future__ import annotations

import json

from .lmstudio_client import (
    DEFAULT_BASE_URL, DEFAULT_TIMEOUT, chat_vision, parse_json_response,
)

# ──── Constants ────

#: Reading a photograph is a perception task, not a creative one. Measured on
#: the acceptance set, every probe returned byte-identical answers across three
#: seeds at this temperature — see DEFAULT_PASSES.
DEFAULT_TEMPERATURE = 0.1

#: Anatomy giveaways live in the details — the join between hip and thigh, the
#: cleft that tells a rear view from a front one. Below roughly 900px they blur
#: away and the model calls everything plausible.
DEFAULT_MAX_IMAGE_SIZE = 1024

#: One pass per probe by default. Repeating a probe at temperature 0.1 returned
#: the identical answer three times out of three on all 17 acceptance images, so
#: extra passes bought accuracy nothing and tripled the runtime. Robustness here
#: comes from asking a *differently worded* question (the confirmation stage),
#: not from asking the same one again. Raise this together with ``temperature``
#: if you want genuine resampling.
DEFAULT_PASSES = 1

#: A picture fails when the weighted violation reaches this. With the default
#: probes every violation is either 0.0 or ~1.0, so the exact cut matters little
#: — it sits at 0.5 to leave room for weighting probes down.
DEFAULT_THRESHOLD = 0.5

SYSTEM_PROMPT = """\
You are a visual measurement instrument. You report what is in the picture, \
literally and precisely.

You never judge quality, beauty, style, clothing or subject matter, and you \
never reassure. You count what you can see and you name what you can see, \
exactly as it appears — even when the result sounds strange or contradicts \
itself. A contradiction in your answers is useful information, not a mistake to \
smooth over.

Judge only what is actually visible. A limb hidden behind a body, cropped by the \
frame or covered by bedding is simply not visible; it is not evidence of \
anything.

Answer with ONE JSON object and nothing else: no markdown fence, no preamble, no \
explanation outside the object. Use exactly the keys you are asked for.
"""


def _clamp01(value, default=0.0):
    """Coerce anything number-ish into 0..1, falling back to ``default``.

    Models express the same confidence as 0.85, "0.85", "85%" or 85, so all four
    have to land on the same number.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        raw = value.strip().rstrip("%").strip().replace(",", ".")
        try:
            number = float(raw)
        except ValueError:
            return default
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
    if number != number:  # NaN
        return default
    if number > 1.0:      # 85 and "85%" both mean "85 percent"
        number /= 100.0
    return max(0.0, min(1.0, number))


def _as_bool(value, default=False):
    """Coerce a model's idea of a boolean — including "yes"/"no" — into bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0.5
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("true", "yes", "y", "1"):
            return True
        if token in ("false", "no", "n", "0", "none"):
            return False
    return default


def _as_count(value):
    """Coerce a count. ``None`` means "no usable number in the answer"."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return max(0, int(round(float(value))))
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            try:
                return int(digits)
            except ValueError:
                return None
    return None


def _token(value):
    """Lower-cased bare token of an enum-ish answer, '' when unusable."""
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if not isinstance(value, str):
        return ""
    return value.strip().strip('"\'').lower()


def _text(value, limit=160):
    """Flatten a reason field — models answer with a string or a list of them."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = "; ".join(str(v) for v in value if v)
    return " ".join(str(value).split())[:limit]


# ──── Scorers ────
#
# A scorer turns one parsed answer into {violation_name: (severity, note)}.
# Severity is 0..1. Names are stable strings so the calibration harness, the
# node report and the unit tests can all refer to the same finding.

#: How many of each part one body may show. Anything above this is a defect;
#: anything below is occlusion and means nothing.
PART_LIMITS = {"heads": 1, "arms": 2, "hands": 2, "legs": 2, "feet_or_shoes": 2}


def _score_parts(data, config):
    """Body parts counted beyond what the expected number of bodies allows."""
    expected = max(1, int(config.get("expected_people", 1)))
    found = {}
    for key, per_body in PART_LIMITS.items():
        limit = per_body * expected
        count = _as_count(data.get(key))
        if count is None or count <= limit:
            continue
        # A head too many is a second person; a limb too many is a growth. Both
        # are conclusive, so both score 1.0 rather than scaling with the excess.
        found[f"extra_{key}"] = (1.0, f"{count} {key.replace('_', ' ')} "
                                      f"(at most {limit} expected)")
    return found


def _score_people(data, config):
    """A second body, either counted or fused into the first."""
    expected = max(1, int(config.get("expected_people", 1)))
    found = {}
    count = _as_count(data.get("people"))
    if count is not None and count != expected:
        found["person_count"] = (1.0 if count > expected else 0.85,
                                 f"{count} people, expected {expected}")
    if _as_bool(data.get("bodies_merged")):
        found["bodies_merged"] = (1.0, _text(data.get("reason")) or
                                  "bodies merge into each other")
    return found


def _score_landmarks(data, config):
    """Front-of-body and back-of-body landmarks seen at the same time.

    The spine allows roughly 90 degrees of rotation between shoulders and hips.
    Breasts and buttock cleft in one frame need about 180, so the pair cannot
    both be present — whichever one the generator got wrong, the picture is
    broken. Same for a navel above a buttock cleft.
    """
    chest = _token(data.get("chest_shows"))
    pelvis = _token(data.get("pelvis_shows"))
    navel = _as_bool(data.get("navel_visible"))
    reason = _text(data.get("reason"))

    found = {}
    if chest == "breasts" and pelvis == "buttock_cleft":
        found["torso_twist"] = (1.0, reason or
                                "chest seen from the front, pelvis from behind")
    if navel and pelvis == "buttock_cleft":
        found["navel_and_buttocks"] = (1.0, "navel and buttock cleft both visible")
    if chest == "shoulder_blades" and pelvis == "groin":
        found["torso_twist_reverse"] = (1.0, reason or
                                        "back seen from behind, pelvis from the front")
    return found


def _score_hands(data, config):
    """Finger counts. Off by default: an 8B model cannot resolve fingers."""
    found = {}
    confidence = _clamp01(data.get("confidence"), 1.0)
    if _as_bool(data.get("wrong_finger_count")) or _as_bool(data.get("malformed")):
        found["hand_malformed"] = (0.7 * max(0.5, confidence),
                                   _text(data.get("reason")) or "malformed hand")
    return found


def _score_physics(data, config):
    """Contact with the world: floating bodies, limbs sunk into furniture."""
    found = {}
    if _as_bool(data.get("floating")):
        found["floating"] = (0.9, _text(data.get("reason")) or "no visible support")
    if _as_bool(data.get("intersects_objects")):
        found["intersects"] = (0.9, _text(data.get("reason")) or
                               "body passes through an object")
    return found


def _score_text(data, config):
    """Gibberish lettering — a reliable synthesis tell when text is present."""
    if _as_bool(data.get("garbled_text")):
        return {"garbled_text": (0.6, _text(data.get("reason")) or
                                 "unreadable lettering")}
    return {}


#: name → probe. ``weight`` scales the violations in the weighted-max aggregate;
#: ``confirms`` names the violations this probe is allowed to overturn.
PROBES = {
    "parts": {
        "weight": 1.0,
        "score": _score_parts,
        "prompt": (
            "Count the body parts visible in this photograph. Count PARTS, not "
            "people: if you can see three legs, answer 3, even if that seems "
            "impossible. Count a part if any recognisable portion of it is "
            "visible. Do not guess at parts you cannot see.\n"
            'Answer JSON: {"heads": <int>, "arms": <int>, "hands": <int>, '
            '"legs": <int>, "feet_or_shoes": <int>}'
        ),
    },
    "people": {
        "weight": 1.0,
        "score": _score_people,
        "prompt": (
            "How many distinct human beings appear in this photograph?\n"
            "Count every person, including any who are partly out of frame, "
            "turned away, or lying behind another. Two heads means two people.\n"
            "Then decide whether any two bodies are fused: one torso continuing "
            "into two different lower bodies, a single pair of legs shared "
            "between two heads, or a limb flowing from one person into another.\n"
            'Answer JSON: {"people": <int>, "bodies_merged": <true|false>, '
            '"reason": "<short>"}'
        ),
    },
    "landmarks": {
        "weight": 1.0,
        "score": _score_landmarks,
        "prompt": (
            "Name the anatomical landmarks you can actually see in this "
            "photograph. Judge each region on its own and report it literally, "
            "even if the answers seem to contradict each other.\n"
            "\n"
            'chest_shows — answer "breasts" if you see the front of the chest '
            "(breasts, cleavage, the cups of a bra); "
            '"shoulder_blades" if you see the upper back (shoulder blades, '
            "spine, the back strap of a bra); "
            '"neither" if the chest is hidden or seen exactly edge-on.\n'
            "\n"
            'pelvis_shows — answer "groin" if you see the FRONT of the pelvis: '
            "the lower belly, the groin creases, the pubic area, the front panel "
            "of the underwear; "
            '"buttock_cleft" if you see the REAR: two buttock cheeks with the '
            "cleft between them, or the rear panel of the underwear over them; "
            '"neither" if the pelvis is hidden or seen exactly edge-on.\n'
            "Note: a person sitting or lying face-up with open legs shows the "
            "GROIN, not the buttock cleft, even when the underside of the thighs "
            "is visible.\n"
            "\n"
            "navel_visible — true only if the navel itself is visible.\n"
            'Answer JSON: {"chest_shows": "<value>", "pelvis_shows": "<value>", '
            '"navel_visible": <bool>, "reason": "<short>"}'
        ),
    },
    "hands": {
        "weight": 0.8,
        "score": _score_hands,
        "prompt": (
            "Look at every clearly visible hand in this photograph and count the "
            "fingers.\n"
            "A defect means a hand with more or fewer than five fingers when all "
            "are visible, fused or melted digits, or a thumb on the wrong side. "
            "Curled, overlapping or partially hidden fingers are NOT defects — "
            "count only what you can clearly resolve, and say so with a low "
            "confidence if you cannot.\n"
            'Answer JSON: {"wrong_finger_count": <true|false>, "malformed": '
            '<true|false>, "confidence": <0.0-1.0>, "reason": "<short>"}'
        ),
    },
    "physics": {
        "weight": 0.8,
        "score": _score_physics,
        "prompt": (
            "Decide whether everything in this photograph is properly supported "
            "by the world around it.\n"
            "Defects: a person hovering with nothing underneath, a body resting "
            "on a surface that is not there, or a limb passing through solid "
            "furniture. Normal contact — sitting, leaning, lying, a mattress "
            "compressing under weight — is not a defect.\n"
            'Answer JSON: {"floating": <true|false>, "intersects_objects": '
            '<true|false>, "reason": "<short>"}'
        ),
    },
    "text": {
        "weight": 0.6,
        "score": _score_text,
        "prompt": (
            "Is there any lettering in this photograph — a sign, label, print, "
            "screen, book, or garment text?\n"
            "If there is, decide whether it is real writing or generator "
            "gibberish: invented letterforms, melted glyphs, words that are not "
            "words in any language. Text that is merely small, blurred or at an "
            "angle is not gibberish. If there is no lettering at all, answer "
            "false.\n"
            'Answer JSON: {"has_text": <true|false>, "garbled_text": '
            '<true|false>, "reason": "<short>"}'
        ),
    },
    # ── Confirmation stage ──
    # Runs only when the violations it can overturn were actually raised, and
    # can only ever clear them, never raise new ones.
    "pelvis_confirm": {
        "weight": 1.0,
        "score": None,
        "confirms": ("torso_twist", "navel_and_buttocks"),
        "prompt": (
            "Look at the pelvis and upper thighs of the person in this "
            "photograph. Exactly one of these descriptions fits what you see "
            "there:\n"
            "A — THE FRONT: the crotch, the pubic mound or the front panel of "
            "the underwear, with the inner thighs opening toward the viewer.\n"
            "B — THE REAR: two rounded buttock cheeks with the vertical cleft "
            "between them, or the rear panel of the underwear over them.\n"
            "C — neither can be made out.\n"
            "Choose the one that matches the pixels, not the one that seems most "
            "likely.\n"
            'Answer JSON: {"choice": "A"|"B"|"C", "reason": "<short>"}'
        ),
    },
}

#: Probes that carry the verdict by default, in run order.
#:
#: ``hands``, ``physics`` and ``text`` are off: measured against the acceptance
#: set they never fired on a broken picture and occasionally fired on a clean
#: one, because an 8B model cannot resolve fingers and reads a compressed
#: mattress as a body sinking through it. Turn them on deliberately.
DEFAULT_PROBES = ("parts", "people", "landmarks", "pelvis_confirm")

#: Answer to :data:`PROBES`\\ ``["pelvis_confirm"]`` that upholds a twist.
_CONFIRM_REAR = "b"


# ──── Aggregation ────

def aggregate_passes(per_pass, mode="median"):
    """Reduce one violation's repeated passes to a single severity.

    ``median`` by default: with three passes a single hallucinated answer cannot
    move a median, while it drags a mean a third of the way and owns a max
    outright. Passes that did not raise the violation count as 0.0, so a finding
    seen once in three passes is correctly discounted.
    """
    values = sorted(v for v in per_pass if v is not None)
    if not values:
        return 0.0
    if mode == "max":
        return values[-1]
    if mode == "mean":
        return sum(values) / len(values)
    if mode == "majority":
        flagged = [v for v in values if v > 0.0]
        return sum(flagged) / len(flagged) if len(flagged) * 2 > len(values) else 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def combine_violations(violations, weights=None):
    """Weighted worst-of across findings.

    Not a sum: these defects are alternatives, not ingredients. One conclusive
    fault — a second head — must condemn a picture on its own, and four mild
    doubts must not add up to a conviction.
    """
    worst = 0.0
    for finding in (violations or {}).values():
        weight = float((weights or {}).get(finding["probe"],
                                           PROBES.get(finding["probe"], {}).get("weight", 1.0)))
        worst = max(worst, finding["severity"] * weight)
    return min(1.0, worst)


# ──── Probe execution ────

def _ask(image_rgb, prompt, base_url, model_id, temperature, seed, timeout,
         max_tokens, max_image_size, system_prompt):
    """One vision call → parsed dict, or (None, error)."""
    response = chat_vision(
        base_url=base_url, model_id=model_id,
        system_prompt=system_prompt or SYSTEM_PROMPT, user_prompt=prompt,
        images=[image_rgb], temperature=temperature, max_tokens=max_tokens,
        seed=seed, timeout=timeout, max_image_size=max_image_size,
    )
    if not response.get("ok"):
        return None, response.get("error") or "vision request failed"
    parsed = parse_json_response(response.get("content", ""))
    if parsed is None:
        snippet = (response.get("content") or "")[:120]
        return None, f"unparsable answer: {snippet!r}"
    return parsed, None


def run_probe(image_rgb, probe_name, base_url=DEFAULT_BASE_URL, model_id="",
              passes=DEFAULT_PASSES, config=None, temperature=DEFAULT_TEMPERATURE,
              seed=None, timeout=DEFAULT_TIMEOUT, max_tokens=300,
              max_image_size=DEFAULT_MAX_IMAGE_SIZE, aggregation="median",
              system_prompt=None):
    """Run one probe ``passes`` times and reduce it to named violations.

    Returns ``{"ok", "violations": {name: {severity, note}}, "answers", "error"}``.
    ``ok`` is False when every pass failed — an unreachable model abstains rather
    than condemning or clearing the picture.
    """
    probe = PROBES.get(probe_name)
    if probe is None:
        return {"ok": False, "violations": {}, "answers": [],
                "error": f"unknown probe {probe_name!r}"}

    config = config or {}
    answers, errors = [], []
    per_pass = []

    for index in range(max(1, int(passes))):
        # Vary the seed per pass; a deterministic server would otherwise return
        # the identical answer and the repetition would buy nothing.
        pass_seed = None if seed is None else int(seed) + index
        parsed, error = _ask(image_rgb, probe["prompt"], base_url, model_id,
                             temperature, pass_seed, timeout, max_tokens,
                             max_image_size, system_prompt)
        if parsed is None:
            errors.append(error)
            answers.append({"ok": False, "error": error})
            continue
        answers.append({"ok": True, "data": parsed})
        per_pass.append(probe["score"](parsed, config) if probe["score"] else {})

    if not per_pass:
        return {"ok": False, "violations": {}, "answers": answers,
                "error": errors[0] if errors else "no usable answer"}

    merged = {}
    for name in {n for found in per_pass for n in found}:
        severities = [found.get(name, (0.0, ""))[0] for found in per_pass]
        notes = [found[name][1] for found in per_pass if name in found and found[name][1]]
        severity = aggregate_passes(severities, aggregation)
        if severity > 0.0:
            merged[name] = {"severity": severity, "probe": probe_name,
                            "note": notes[0] if notes else ""}

    return {"ok": True, "violations": merged, "answers": answers, "error": None}


def _confirmation_verdict(answers):
    """Did the confirmation stage uphold the suspicion? ``None`` = no answer."""
    choices = [_token((a.get("data") or {}).get("choice"))
               for a in answers if a.get("ok")]
    choices = [c[0] for c in choices if c]      # tolerate "B — the rear ..."
    if not choices:
        return None
    rear = sum(1 for c in choices if c == _CONFIRM_REAR)
    return rear * 2 >= len(choices)


def assess(image_rgb, base_url=DEFAULT_BASE_URL, model_id="",
           probes=DEFAULT_PROBES, passes=DEFAULT_PASSES,
           threshold=DEFAULT_THRESHOLD, expected_people=1,
           temperature=DEFAULT_TEMPERATURE, seed=None, timeout=DEFAULT_TIMEOUT,
           max_image_size=DEFAULT_MAX_IMAGE_SIZE, aggregation="median",
           weights=None, system_prompt=None):
    """Run the enabled probes over one image and return a single verdict.

    Returns ``{"passed", "score", "issues", "violations", "probes", "report",
    "ok", "error"}``. ``passed`` is True when the weighted violation stays below
    ``threshold``.

    If no probe reaches the model, ``ok`` is False and ``passed`` is True: an
    inspector that cannot see must not silently reject a whole batch.
    """
    config = {"expected_people": int(expected_people)}
    selected = [name for name in (probes or DEFAULT_PROBES) if name in PROBES]
    kwargs = dict(base_url=base_url, model_id=model_id, passes=passes,
                  config=config, temperature=temperature, seed=seed,
                  timeout=timeout, max_image_size=max_image_size,
                  aggregation=aggregation, system_prompt=system_prompt)

    results, violations = {}, {}
    for name in selected:
        if PROBES[name].get("confirms"):
            continue                      # confirmation stage runs afterwards
        results[name] = run_probe(image_rgb, name, **kwargs)
        violations.update(results[name]["violations"])

    # Confirmation stage: a differently worded second opinion that can only
    # clear a suspicion, never raise one. Skipped when nothing it covers fired,
    # which keeps the common (clean) case at one call per probe.
    overturned = {}
    for name in selected:
        covers = PROBES[name].get("confirms")
        if not covers:
            continue
        pending = [v for v in covers if v in violations]
        if not pending:
            continue
        results[name] = run_probe(image_rgb, name, **kwargs)
        upheld = _confirmation_verdict(results[name]["answers"])
        results[name]["upheld"] = upheld
        if upheld is False:
            for finding in pending:
                overturned[finding] = violations.pop(finding)

    reachable = [r for r in results.values() if r.get("ok")]
    if not reachable:
        error = next((r.get("error") for r in results.values() if r.get("error")),
                     "no probe reached the model")
        return {"passed": True, "score": 0.0, "issues": [], "violations": {},
                "probes": results, "ok": False, "error": error,
                "report": f"Reality check unavailable: {error}"}

    score = combine_violations(violations, weights)
    passed = score < float(threshold)
    issues = [f"{name}: {finding['note']}" if finding["note"] else name
              for name, finding in sorted(violations.items(),
                                          key=lambda kv: -kv[1]["severity"])]

    lines = [f"{'PASS' if passed else 'FAIL'}  score={score:.2f} "
             f"(threshold {float(threshold):.2f})"]
    for name, result in results.items():
        if not result.get("ok"):
            lines.append(f"  {name:<14} unavailable: {result.get('error')}")
        elif result.get("upheld") is False:
            lines.append(f"  {name:<14} overturned {', '.join(overturned)}")
        elif result.get("upheld") is True:
            lines.append(f"  {name:<14} upheld")
        else:
            found = result["violations"]
            lines.append(f"  {name:<14} " +
                         (", ".join(f"{n} {v['severity']:.2f}"
                                    for n, v in found.items()) if found else "clean"))
    for name, finding in violations.items():
        if finding["note"]:
            lines.append(f"    • {name}: {finding['note']}")

    return {"passed": passed, "score": score, "issues": issues,
            "violations": violations, "probes": results, "ok": True,
            "error": None, "report": "\n".join(lines)}


def verdict_to_json(result):
    """Compact JSON of a verdict — for sidecar files and downstream nodes."""
    return json.dumps({
        "passed": bool(result.get("passed")),
        "score": round(float(result.get("score", 0.0)), 4),
        "issues": list(result.get("issues") or []),
        "violations": {name: round(float(finding["severity"]), 4)
                       for name, finding in (result.get("violations") or {}).items()},
    }, ensure_ascii=False)

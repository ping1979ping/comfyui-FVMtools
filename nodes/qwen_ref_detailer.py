"""Qwen Image 2.1 reference head detailer.

Re-renders one person's head per node with Qwen Image 2.1 Edit: the head crop
(with context) is image_1, the person's reference photos are image_2..N, the
conditioning is built per crop the way TextEncodeQwenImage21 does it, and the
result is stitched back feathered. Chain one node per reference; image, pipe
and PERSON_DATA pass straight through.
"""

import time

import cv2
import numpy as np
import torch

import comfy.samplers

from .utils.inpaint_pipeline import compute_crop_region, inpaint_slot
from .utils.mask_utils import expand_mask, feather_mask, fill_mask_holes_2d
from .utils.qwen_ref import (
    DEFAULT_PROMPT_TEMPLATE,
    EXPRESSION_TEXTS,
    MASK_TYPE_PARTS,
    OPEN_MOUTH_KINDS,
    MAX_REFS,
    _to_bgr_uint8,
    apply_lab_shift,
    cheek_mask,
    build_prompt,
    candidate_score,
    candidate_seed,
    classify_expression,
    format_candidates,
    get_person_mask,
    make_grid,
    mask_bbox,
    mouth_metrics,
    mouth_state,
    pick_candidate,
    pose_delta,
    pick_face_for_mask,
    prepare_refs,
    qwen_dims,
    resize_image,
    skin_tone_shift,
)

QWEN_PIPE = "FVM_QWEN_PIPE"
_PIPE_KEYS = ("model", "clip", "vae")

_face_analyzer = None


# ── Pipe ──────────────────────────────────────────────────────────────────


def _from_bundle(bundle, key):
    """Pick model/clip/vae out of an OBVPM bundle; field names are free-form."""
    if not isinstance(bundle, dict):
        return None
    for name, value in bundle.items():
        if str(name).strip().lower() == key:
            return value
    return None


def resolve_pipe(pipe=None, bundle=None, model=None, clip=None, vae=None):
    """Merge sources into one pipe dict. Precedence: single inputs > bundle > pipe."""
    out = {}
    for key, single in zip(_PIPE_KEYS, (model, clip, vae)):
        value = single
        if value is None:
            value = _from_bundle(bundle, key)
        if value is None and isinstance(pipe, dict):
            value = pipe.get(key)
        out[key] = value
    missing = [k for k in _PIPE_KEYS if out[k] is None]
    if missing:
        raise ValueError(
            f"Qwen 2.1 Pipe: {', '.join(missing)} missing - connect them directly, "
            "via a bundle or via an upstream pipe."
        )
    return out


class FVM_QwenPipe:
    """Bundles MODEL/CLIP/VAE into one wire for the Qwen ref detailer chain."""

    CATEGORY = "FVM Tools/Face"
    FUNCTION = "execute"
    RETURN_TYPES = (QWEN_PIPE, "MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("pipe", "model", "clip", "vae")
    DESCRIPTION = (
        "Bundles Qwen Image 2.1 MODEL, CLIP (qwen_image text encoder) and VAE into one pipe.\n"
        "Sources, strongest first: the single inputs, an OBVPM bundle (fields named model/clip/vae), "
        "an upstream pipe."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "pipe": (
                    QWEN_PIPE,
                    {
                        "tooltip": "An existing Qwen 2.1 pipe, e.g. from another Qwen 2.1 Pipe node or a "
                        "detailer's pipe output.\n"
                        "Lowest priority: its model / clip / vae are only used for fields that neither the "
                        "single inputs nor the bundle provide. Handy for swapping out just one part, e.g. "
                        "pipe in + a LoRA-patched model -> new pipe.",
                    },
                ),
                "bundle": (
                    "OBVPM_BUNDLE",
                    {
                        "tooltip": "OBVPM bundle (Bundle node) with fields named exactly model, clip and vae - "
                        "e.g. the 'BUN Qwen' bundle in the V6 workflow.\n"
                        "Beats the pipe input, loses to the single model / clip / vae inputs. Other fields "
                        "in the bundle are ignored.",
                    },
                ),
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Qwen Image 2.1 diffusion model (UNETLoader, e.g. "
                        "qwen_image_2.1_int8_convrot). Highest priority - overrides the bundle and the pipe. "
                        "A LoRA-patched model can go here too.",
                    },
                ),
                "clip": (
                    "CLIP",
                    {
                        "tooltip": "Qwen3-VL text encoder, loaded with CLIPLoader type 'qwen_image' (e.g. "
                        "qwen3vl_8b_int8_convrot). Highest priority. Not the qwen3.5 prompt-enhancer "
                        "models - those only work with Generate Text.",
                    },
                ),
                "vae": (
                    "VAE",
                    {
                        "tooltip": "Qwen Image 2.1 VAE (qwen_image_2.1_vae_bf16). Highest priority. The older "
                        "Qwen Image VAE does not fit the 2.1 model. Note: this VAE decodes RGBA; the "
                        "detailer drops the alpha channel itself.",
                    },
                ),
            },
        }

    def execute(self, pipe=None, bundle=None, model=None, clip=None, vae=None):
        out = resolve_pipe(pipe, bundle, model, clip, vae)
        return (out, out["model"], out["clip"], out["vae"])


# ── Conditioning ──────────────────────────────────────────────────────────


def _encode_qwen21(clip, prompt, negative_prompt, vae, images):
    """Run core TextEncodeQwenImage21 with resolution=0 (sizes are ours).

    Imported lazily: comfy_extras is not available in the unit tests.
    """
    from comfy_extras.nodes_qwen import TextEncodeQwenImage21

    out = TextEncodeQwenImage21.execute(
        clip=clip,
        prompt=prompt,
        negative_prompt=negative_prompt,
        vae=vae,
        resolution=0,
        images=images,
    )
    return out.args


def _get_face_analyzer():
    """Reuse the selector's InsightFace instance; build one only if needed."""
    global _face_analyzer
    try:
        from .person_selector_sam3 import PersonSelectorSAM3

        if PersonSelectorSAM3._face_analyzer is not None:
            return PersonSelectorSAM3._face_analyzer
    except Exception:
        pass
    if _face_analyzer is None:
        from .utils.face_analyzer import FaceAnalyzer

        _face_analyzer = FaceAnalyzer(640)
    return _face_analyzer


def _placeholder():
    return torch.zeros(1, 64, 64, 3, dtype=torch.float32)


def _stack(images):
    """Stack previews of different sizes by resizing to the first."""
    if not images:
        return _placeholder()
    h, w = images[0].shape[1], images[0].shape[2]
    return torch.cat(
        [resize_image(im[..., :3].float().cpu(), w, h) for im in images], dim=0
    )


def _ref_mouth_state(refs, analyzer):
    """Majority mouth state of the reference photos: "open", "closed" or None."""
    if analyzer is None:
        return None, []
    kinds = []
    for ref in refs:
        try:
            bgr = _to_bgr_uint8(ref)
            faces = analyzer.detect_faces(bgr)
        except Exception:
            continue
        faces = [f for f in faces if getattr(f, "landmark_3d_68", None) is not None]
        if not faces:
            continue
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        kinds.append(classify_expression(mouth_metrics(f.landmark_3d_68, bgr)))
    if not kinds:
        return None, kinds
    n_open = sum(1 for k in kinds if k in OPEN_MOUTH_KINDS)
    if n_open * 2 > len(kinds):
        return "open", kinds
    if n_open * 2 < len(kinds):
        return "closed", kinds
    return None, kinds


def _expression_text(img_hwc, mask_2d, expression_hint, analyzer, ref_state=None):
    """(text, note): worded expression of the masked person's face, or ("", why).

    auto: only when the references mostly show the other mouth state (open vs
    closed) - that is when the model copies the portraits' mouth. Describing
    an expression the references already share was measured to cost identity
    (Jerri, IMG_0014: own_sim 0.87 -> 0.66-0.69) without being needed.
    always: whenever a face is found.
    """
    if expression_hint not in ("auto", "always"):
        return "", ""
    if analyzer is None:
        return "", "expression hint: no face analyzer"
    try:
        bgr = _to_bgr_uint8(img_hwc[None, ..., :3])
        face = pick_face_for_mask(analyzer.detect_faces(bgr), mask_2d)
    except Exception as e:  # the hint is optional, never fatal
        return "", f"expression hint failed ({e})"
    if face is None or getattr(face, "landmark_3d_68", None) is None:
        return "", "expression hint: no face with landmarks inside the mask"
    m = mouth_metrics(face.landmark_3d_68, bgr)
    kind = classify_expression(m)
    desc = f"expression {kind} (mar {m['mar']:.3f}, smile {m['smile']:.3f}, teeth {m['teeth']:.3f})"
    state = "open" if kind in OPEN_MOUTH_KINDS else "closed"
    if expression_hint == "auto" and ref_state == state:
        return "", f"{desc}; references mostly {ref_state} too - no hint"
    return EXPRESSION_TEXTS[
        kind
    ], f"{desc} - hint added (references: {ref_state or 'unknown'})"


def _ref_embedding(refs, analyzer):
    """Normalised mean InsightFace embedding of the references' largest faces."""
    embs = []
    for ref in refs:
        try:
            faces = analyzer.detect_faces(_to_bgr_uint8(ref))
        except Exception:
            continue
        faces = [f for f in faces if getattr(f, "normed_embedding", None) is not None]
        if not faces:
            continue
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        e = np.asarray(f.normed_embedding, dtype=np.float64)
        embs.append(e / (np.linalg.norm(e) + 1e-12))
    if not embs:
        return None
    m = np.mean(embs, axis=0)
    return m / (np.linalg.norm(m) + 1e-12)


def _face_in_region(img_hwc, mask_2d, box, analyzer):
    """(face, bgr) of the person under `mask_2d`, detected inside `box`, or (None, None).

    Detecting on the head crop instead of the whole picture keeps the face
    large enough for a stable embedding and landmarks.
    """
    H, W = img_hwc.shape[0], img_hwc.shape[1]
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(W, int(box[2])), min(H, int(box[3]))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, None
    try:
        bgr = _to_bgr_uint8(img_hwc[None, y0:y1, x0:x1, :3])
        faces = analyzer.detect_faces(bgr)
    except Exception:
        return None, None
    return pick_face_for_mask(faces, mask_2d[y0:y1, x0:x1]), bgr


def _judge_face(face, bgr, ref_emb):
    """(identity cosine, expression kind, pose (pitch, yaw, roll), mouth metrics)
    of a detected face; each None when unavailable."""
    if face is None:
        return None, None, None, None
    sim = None
    emb = getattr(face, "normed_embedding", None)
    if emb is not None and ref_emb is not None:
        e = np.asarray(emb, dtype=np.float64)
        sim = float(e @ ref_emb / (np.linalg.norm(e) + 1e-12))
    kind = metrics = None
    lm = getattr(face, "landmark_3d_68", None)
    if lm is not None:
        metrics = mouth_metrics(lm, bgr)
        kind = classify_expression(metrics)
    pose = getattr(face, "pose", None)
    if pose is not None:
        pose = [float(v) for v in pose]
    return sim, kind, pose, metrics


def _cheeks_in_region(img_hwc, mask_2d, box, analyzer):
    """Full-size cheek mask of the person's face found inside `box`, or None."""
    if analyzer is None:
        return None
    face, bgr = _face_in_region(img_hwc, mask_2d, box, analyzer)
    lm = getattr(face, "landmark_3d_68", None) if face is not None else None
    if lm is None:
        return None
    H, W = img_hwc.shape[0], img_hwc.shape[1]
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    full = np.zeros((H, W), np.uint8)
    part = cheek_mask(lm, bgr.shape)
    full[y0 : y0 + part.shape[0], x0 : x0 + part.shape[1]] = part
    return full


def _match_skin_tone(
    orig_hwc,
    new_hwc,
    person_data,
    ri,
    b,
    processed,
    blend_px,
    direction,
    strength,
    analyzer=None,
    box=None,
):
    """Shift the re-rendered head in Lab so its facial skin matches the original's.

    Qwen takes skin colour and white balance from the reference portraits even
    when told not to (IMG_0001, warm evening light: new faces up to +12 L and
    -25 b* against their own necks). The offset is measured on the cheeks of
    each face (InsightFace landmarks of the original and of the new face);
    without landmarks on the original's facial-skin mask (eroded; face mask as
    fallback) in both images - which misreads when the new hair falls where the
    old skin was (IMG_0010: faces brightened by +8 L). It is added under the
    feathered head mask, keyed to pixels of the new skin colour
    (skin_key) and to pixels the render actually changed, so sky, clothes and
    untouched skin keep their colour.
    Returns (image, note).
    """
    H, W = orig_hwc.shape[0], orig_hwc.shape[1]
    o = orig_hwc[..., :3].detach().float().cpu().numpy()
    n = new_hwc[..., :3].detach().float().cpu().numpy()
    weight = feather_mask(processed, blend_px, direction).numpy()
    if box is not None:
        co = _cheeks_in_region(orig_hwc, processed, box, analyzer)
        cn = _cheeks_in_region(new_hwc, processed, box, analyzer)
        if co is not None and cn is not None:
            shift, key = skin_tone_shift(o, n, co, cn)
            if shift is not None:
                out = apply_lab_shift(n, shift, weight, strength, key_lab=key, ref_rgb=o)
                note = (
                    f"skin tone match (cheeks, strength {strength:.2f}): "
                    f"dL {shift[0]:+.1f} da {shift[1]:+.1f} db {shift[2]:+.1f}"
                )
                return torch.from_numpy(out).to(new_hwc.device, new_hwc.dtype), note
    skin, part = None, None
    for part in ("facial_skin", "face"):
        skin, _ = get_person_mask(person_data, ri, b, part, H, W)
        if skin is not None:
            break
    if skin is None:
        return new_hwc, "skin tone match: no facial_skin/face mask - skipped"
    sk = (skin.numpy() > 0.5).astype(np.uint8)
    bb = mask_bbox(skin)
    k = max(3, int(0.04 * max(bb[2] - bb[0], bb[3] - bb[1]))) | 1
    sk = cv2.erode(sk, np.ones((k, k), np.uint8)) & (processed.numpy() > 0.5)
    shift, key = skin_tone_shift(o, n, sk)
    if shift is None:
        return new_hwc, f"skin tone match: too few {part} pixels - skipped"
    out = apply_lab_shift(n, shift, weight, strength, key_lab=key, ref_rgb=o)
    note = (
        f"skin tone match ({part}, strength {strength:.2f}): "
        f"dL {shift[0]:+.1f} da {shift[1]:+.1f} db {shift[2]:+.1f}"
    )
    return torch.from_numpy(out).to(new_hwc.device, new_hwc.dtype), note


# ── Core ──────────────────────────────────────────────────────────────────


def run_ref_head_detail(
    image,
    model,
    clip,
    vae,
    person_data,
    ref_index,
    reference_images,
    prompt_template,
    extra_prompt,
    negative_prompt,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    denoise,
    latent_mode,
    ref_mode,
    max_refs,
    ref_crop,
    ref_crop_factor,
    canvas_resolution,
    ref_resolution,
    mask_type,
    mask_expand_percent,
    mask_blend_pixels,
    context_expand_factor,
    output_padding,
    delta_clamp,
    feather_direction,
    expression_hint="auto",
    analyzer_fn=None,
    take_hairstyle="yes",
    candidates=1,
    skin_tone_match=0.0,
):
    """Returns (images, refined_crops, canvas_crops, ref_sheet, info_lines)."""
    t0 = time.perf_counter()
    images = image[..., :3]
    info = [f"ref {ref_index}"]
    ri = int(ref_index) - 1
    num_refs = int(person_data.get("num_references", 0))
    if ri >= num_refs:
        info.append(
            f"skip: PERSON_DATA has {num_refs} references, ref_index {ref_index} unused"
        )
        return images, _placeholder(), _placeholder(), _placeholder(), info

    analyzer = None
    candidates = max(1, int(candidates))
    if (
        ref_crop == "face"
        or expression_hint in ("auto", "always")
        or candidates > 1
        or skin_tone_match > 0
    ):
        try:
            analyzer = (analyzer_fn or _get_face_analyzer)()
        except Exception as e:
            info.append(f"face analyzer unavailable ({e}), references used uncropped")
    refs, notes = prepare_refs(
        reference_images,
        max_refs,
        ref_mode,
        ref_crop,
        ref_crop_factor,
        ref_resolution,
        analyzer,
    )
    info.extend(notes)
    if not refs:
        info.append("skip: no usable reference images")
        return images, _placeholder(), _placeholder(), _placeholder(), info
    # In grid mode `refs` holds the one tiled image; the prompt wording still
    # needs to know whether it shows one photo or several.
    n_refs = (
        min(reference_images.shape[0], max(1, min(int(max_refs), MAX_REFS)))
        if ref_mode == "grid"
        else len(refs)
    )
    info.append(
        f"refs: {len(refs)} image(s) {', '.join(f'{r.shape[2]}x{r.shape[1]}' for r in refs)} "
        f"(mode {ref_mode}, crop {ref_crop})"
    )
    last_prompt = None
    ref_state = None
    if expression_hint == "auto":
        # Judge the uncropped photos: on the tight face crops the detector
        # missed most faces (IMG_0014 refs: 0/4 and 2/4 found), and in grid
        # mode `refs` is one tiled sheet.
        n_used = len(refs) if ref_mode != "grid" else n_refs
        ref_imgs = [reference_images[i : i + 1, ..., :3] for i in range(n_used)]
        ref_state, ref_kinds = _ref_mouth_state(ref_imgs, analyzer)
        info.append(
            f"reference expressions: {', '.join(ref_kinds) or 'none found'} -> {ref_state}"
        )

    ref_emb = None
    if candidates > 1:
        if analyzer is not None:
            n_used = len(refs) if ref_mode != "grid" else n_refs
            ref_emb = _ref_embedding(
                [reference_images[i : i + 1, ..., :3] for i in range(n_used)], analyzer
            )
        if ref_emb is None:
            info.append(
                f"candidates {candidates}: no reference face embedding (face analyzer "
                "missing or no face in the references) - rendering 1"
            )
            candidates = 1

    B, H, W = images.shape[0], images.shape[1], images.shape[2]
    out_images, refined, canvases = [], [], []
    for b in range(B):
        img_b = images[b]
        mask, reason = get_person_mask(person_data, ri, b, mask_type, H, W)
        if reason and mask is not None:
            info.append(f"image {b + 1}: {reason}")
        if mask is None:
            info.append(f"image {b + 1}: skip - {reason}")
            out_images.append(img_b)
            continue
        processed = fill_mask_holes_2d((mask > 0.5).float())
        bbox = mask_bbox(processed)
        if bbox is None:
            info.append(f"image {b + 1}: skip - mask empty after hole fill")
            out_images.append(img_b)
            continue
        size = max(bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
        px = int(round(mask_expand_percent * size))
        if px > 0:
            processed = expand_mask(processed, px)
        expression, note = _expression_text(
            img_b, processed, expression_hint, analyzer, ref_state
        )
        if note:
            info.append(f"image {b + 1}: {note}")
        prompt = build_prompt(
            prompt_template, n_refs, ref_mode, extra_prompt, expression, take_hairstyle
        )
        if prompt != last_prompt:
            info.append(f"prompt: {prompt}")
            last_prompt = prompt
        crop0 = compute_crop_region(processed, context_expand_factor, output_padding)
        tw, th = qwen_dims(crop0["w"], crop0["h"], canvas_resolution)

        cache = {}

        def cond_fn(
            m, _pos, _neg, crop_image, _tw=tw, _th=th, prompt=prompt, cache=cache
        ):
            # Hook meant for ControlNet; here it builds the whole Qwen 2.1
            # conditioning from the actual sampled crop (image_1) + references.
            # Candidates re-sample the very same crop, so text/vision encoding
            # and the reference latents are computed once and reused.
            if "cond" in cache:
                return (m,) + cache["cond"]
            canvas = crop_image[..., :3].detach().float().cpu()
            canvases.append(canvas)
            imgs = {"image_1": canvas}
            for i, ref in enumerate(refs):
                imgs[f"image_{i + 2}"] = ref
            pos, neg, latent = _encode_qwen21(clip, prompt, negative_prompt, vae, imgs)
            shape = tuple(latent["samples"].shape[-2:])
            if shape != (_th // 16, _tw // 16):
                raise RuntimeError(
                    f"Qwen latent {shape} does not match the crop {_tw}x{_th}"
                )
            cache["cond"] = (pos, neg)
            return m, pos, neg

        noise_mask = torch.ones_like(processed) if latent_mode == "full" else None
        box = (
            crop0["x"],
            crop0["y"],
            crop0["x"] + crop0["w"],
            crop0["y"] + crop0["h"],
        )
        orig_kind = orig_pose = orig_m = None
        if candidates > 1:
            _, orig_kind, orig_pose, orig_m = _judge_face(
                *_face_in_region(img_b, processed, box, analyzer), None
            )
        tb = time.perf_counter()
        rows, results = [], []
        for k in range(candidates):
            cseed = candidate_seed(seed, b, k, candidates)
            stitched, decoded = inpaint_slot(
                image=img_b,
                mask_2d=processed,
                model=model,
                positive_cond=None,
                negative_cond=None,
                vae=vae,
                seed=cseed,
                steps=steps,
                denoise=denoise,
                sampler_name=sampler_name,
                scheduler=scheduler,
                target_width=tw,
                target_height=th,
                mask_expand_pixels=0,
                mask_blend_pixels=mask_blend_pixels,
                mask_fill_holes=False,
                context_expand_factor=context_expand_factor,
                output_padding=output_padding,
                dd_enabled=False,
                repeat=1,
                controlnet_apply_fn=cond_fn,
                cfg=cfg,
                delta_clamp=delta_clamp,
                feather_direction=feather_direction,
                noise_mask_2d=noise_mask,
            )
            results.append((stitched, decoded))
            if candidates > 1:
                sim, kind, pose, cm = _judge_face(
                    *_face_in_region(stitched, processed, box, analyzer), ref_emb
                )
                o, c = mouth_state(orig_kind), mouth_state(kind)
                dp = pose_delta(orig_pose, pose)
                ds = mr = None
                if cm is not None and orig_m is not None:
                    ds = cm["smile"] - orig_m["smile"]
                    mr = cm["mar"] / max(orig_m["mar"], 1e-6)
                rows.append(
                    {
                        "seed": cseed,
                        "sim": sim,
                        "kind": kind,
                        "mismatch": o is not None and c is not None and o != c,
                        "d_pose": dp,
                        "d_smile": ds,
                        "mar_ratio": mr,
                        "score": candidate_score(
                            sim, orig_kind, kind, d_pose=dp, d_smile=ds, mar_ratio=mr
                        ),
                    }
                )
        chosen = pick_candidate([r["score"] for r in rows]) if rows else 0
        stitched, decoded = results[chosen]
        if skin_tone_match > 0 and decoded is not None:
            # Same ramp width inpaint_slot used for the stitch (sampling px -> image px).
            blend_px = min(
                256, max(1, int(round(mask_blend_pixels * crop0["w"] / max(1, tw))))
            )
            stitched, note = _match_skin_tone(
                img_b,
                stitched,
                person_data,
                ri,
                b,
                processed,
                blend_px,
                feather_direction,
                float(skin_tone_match),
                analyzer,
                box,
            )
            info.append(f"image {b + 1}: {note}")
        out_images.append(stitched)
        if decoded is not None:
            refined.append(decoded[..., :3].float().cpu())
        if rows:
            info.append(
                f"image {b + 1}: original mouth {orig_kind or 'unknown'}; "
                + format_candidates(rows, chosen)
            )
        dt = time.perf_counter() - tb
        info.append(
            f"image {b + 1}: mask bbox {bbox}, expand {px}px, crop {crop0['w']}x{crop0['h']} "
            f"-> sample {tw}x{th}, {candidates} candidate(s), {dt:.1f}s "
            f"({dt / candidates:.1f}s each)"
        )

    result = torch.stack(
        [o[..., :3].to(images.device, images.dtype) for o in out_images], dim=0
    )
    sheet = make_grid(refs, 256) if len(refs) > 1 else refs[0]
    info.append(f"total {time.perf_counter() - t0:.1f}s")
    return result, _stack(refined), _stack(canvases), sheet, info


class FVM_QwenRefHeadDetailer:
    """One person's head re-rendered by Qwen Image 2.1 Edit from reference photos."""

    CATEGORY = "FVM Tools/Face"
    FUNCTION = "execute"
    RETURN_TYPES = (
        "IMAGE",
        QWEN_PIPE,
        "PERSON_DATA",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "STRING",
    )
    RETURN_NAMES = (
        "image",
        "pipe",
        "person_data",
        "refined_crops",
        "canvas_crops",
        "ref_sheet",
        "info",
    )
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Re-renders the head of ONE reference person with Qwen Image 2.1 Edit.\n\n"
        "The head crop (with context) becomes <image1>, the person's reference photos "
        "<image2>..<imageN>; the conditioning is built per crop like TextEncodeQwenImage21 "
        "(reference latents included), sampled at the crop's size and stitched back feathered.\n\n"
        "Chain one node per reference: image, pipe and person_data pass through.\n"
        "Prompt placeholders: {canvas} {refs} {ref_count} {hair} {expression} {extra}."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The picture whose head gets re-rendered. First node: the same image that "
                        "went into the Person Selector as current_image. Later nodes: the image output of "
                        "the previous detailer.\n"
                        "Must keep the Person Selector's size - PERSON_DATA masks are in its pixels "
                        "(a different size is resized with a warning). Batches work; alpha is dropped.",
                    },
                ),
                "pipe": (
                    QWEN_PIPE,
                    {
                        "tooltip": "model / clip / vae of Qwen Image 2.1 - from the Qwen 2.1 Pipe node or the "
                        "pipe output of the previous detailer (passed through unchanged).",
                    },
                ),
                "person_data": (
                    "PERSON_DATA",
                    {
                        "tooltip": "From Person Selector SAM3 (Native), or the person_data output of the previous "
                        "detailer (passed through unchanged). Supplies the head / face / hair / neck masks "
                        "of every matched reference person.",
                    },
                ),
                "ref_index": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "tooltip": "Which person this node re-renders: N = the person the selector matched to its "
                        "reference_N input. Must fit reference_images (reference_3 set -> 3).\n"
                        "If the selector found nobody for this reference, the image passes through unchanged "
                        "and info says why.",
                    },
                ),
                "reference_images": (
                    "IMAGE",
                    {
                        "tooltip": "Photos of this ONE person as a batch (e.g. Image Batch Multiple). Every photo "
                        "is used, up to max_refs - unlike TextEncodeQwenImage21, which only takes the first "
                        "image of a batch.\n"
                        "Best: 2-4 sharp photos, different angles, face clearly visible. Mixed sizes are fine "
                        "after batching; each photo is cropped to the face (ref_crop).",
                    },
                ),
                "prompt_template": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_PROMPT_TEMPLATE,
                        "tooltip": "Edit instruction for Qwen. Placeholders are filled per run:\n"
                        "  {canvas}     -> <image1>, the head crop being edited\n"
                        "  {refs}       -> <image2>, <image3> ... the reference photos\n"
                        "  {ref_count}  -> 'these 4 photos' / 'this photo grid' / 'this photo'\n"
                        "  {hair}       -> hairstyle wording, see take_hairstyle\n"
                        "  {expression} -> mouth description, see expression_hint\n"
                        "  {extra}      -> extra_prompt\n"
                        "The default is tuned: identity from the references, expression / pose / light / "
                        "skin tone from the picture. Only change it on purpose.",
                    },
                ),
                "extra_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Your own addition, inserted at {extra} at the end of the prompt, e.g. "
                        "'wearing sunglasses', 'wet hair', 'light stubble'. Refer to the picture as <image1>. "
                        "Leave empty for a plain identity swap.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "What to avoid, e.g. 'blurry, passport photo, neutral expression'. Only has an "
                        "effect with cfg above 1.0 - at cfg 1.0 it is ignored.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Noise seed. With candidates > 1 the node renders seed, seed+1, ... and keeps "
                        "the best. In a batch each image gets its own seeds. Results vary a lot between "
                        "seeds - that is what candidates is for.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Sampling steps per candidate. 30 is the tested value (~10 s per candidate on "
                        "an RTX 5090). The official Qwen pipeline uses 40-50; below ~20 the face gets less "
                        "detailed.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Guidance. 1.0 (Qwen Image 2.1 default) = negative prompt off, fastest.\n"
                        "1.2-1.5 turns the negative prompt on and can sharpen the identity a little, but "
                        "costs ~1.7x the time; in testing it gave no reliable gain.",
                    },
                ),
                "sampler_name": (
                    comfy.samplers.SAMPLER_NAMES,
                    {
                        "default": "euler",
                        "tooltip": "Sampler. euler is what Qwen Image 2.1 is tuned and tested with.",
                    },
                ),
                "scheduler": (
                    comfy.samplers.SCHEDULER_NAMES,
                    {
                        "default": "simple",
                        "tooltip": "Noise schedule. simple is the Qwen Image 2.1 default.",
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "How much of the head is re-generated. Keep 1.0.\n"
                        "Tested: at 0.92 and below Qwen just returns the original face (no identity "
                        "change at all); 0.95-0.97 flips unpredictably between original and new face. "
                        "Expression is preserved via the prompt instead.",
                    },
                ),
                "latent_mode": (
                    ["masked", "full"],
                    {
                        "default": "masked",
                        "tooltip": "masked: only the (expanded) head mask is re-noised; everything else in the crop, "
                        "including neighbouring people, is held fixed in latent space. Safe for groups.\n"
                        "full: the whole crop is re-generated, then only the head is pasted back. Lets the "
                        "head grow past the mask (longer hair), but risks shifts and seams at the edge.",
                    },
                ),
                "ref_mode": (
                    ["separate", "grid", "first_only"],
                    {
                        "default": "separate",
                        "tooltip": "How the reference photos reach Qwen.\n"
                        "separate: each photo is its own image (<image2>, <image3>, ...) - best identity.\n"
                        "grid: all photos tiled into one image - same token cost, the model may read it as "
                        "a collage.\n"
                        "first_only: only the first photo - what TextEncodeQwenImage21 does with a batch "
                        "(for comparison).",
                    },
                ),
                "max_refs": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": MAX_REFS,
                        "tooltip": "Upper limit of reference photos used (the rest of the batch is ignored). "
                        "Qwen takes at most 10 images, the head crop counts as one.\n"
                        "More photos = more stable identity, but more tokens: time and VRAM grow with every "
                        "photo (x ref_resolution²). 4 is the tested value.",
                    },
                ),
                "ref_crop": (
                    ["face", "none"],
                    {
                        "default": "face",
                        "tooltip": "face: each reference photo is cut to a square around its largest face "
                        "(InsightFace), so the pixel budget goes to the face instead of body and background.\n"
                        "none: use the photos as they are (only for photos that are already head shots). "
                        "Photos without a detectable face are always used uncropped.",
                    },
                ),
                "ref_crop_factor": (
                    "FLOAT",
                    {
                        "default": 2.2,
                        "min": 1.2,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Size of the reference face crop: square side = this x the face box, shifted "
                        "up a little for the hair.\n"
                        "~1.5: face only (hair gets cut); 2.2: head with hair and a bit of neck (tested "
                        "default); 3+: head and shoulders, less face detail. Only with ref_crop = face.",
                    },
                ),
                "canvas_resolution": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 2048,
                        "step": 32,
                        "tooltip": "Pixel budget of the head crop that is rendered (side of a square of the same "
                        "area; the crop keeps its aspect ratio, sizes snap to multiples of 32).\n"
                        "1024 = ~1 MP, tested default. Higher = more face detail, but slower and more VRAM; "
                        "pointless if the head is small in the source.",
                    },
                ),
                "ref_resolution": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 2048,
                        "step": 32,
                        "tooltip": "Pixel budget per reference photo (grid mode: per grid cell).\n"
                        "512 is the tested default; 768 gave no reliable gain. Each step up costs tokens for "
                        "every photo - with 4 photos at 1024 the model's prefix cache may no longer fit.",
                    },
                ),
                "mask_type": (
                    list(MASK_TYPE_PARTS),
                    {
                        "default": "head",
                        "tooltip": "Which region of the person is re-rendered (masks from PERSON_DATA).\n"
                        "head: face + hair (default).\n"
                        "head+neck: also the neck - helps when the new face's skin tone meets the neck "
                        "with a visible edge.\n"
                        "face: face only, the original hair stays.\n"
                        "face+hair: face and hair masks combined, for when the head mask is poor.",
                    },
                ),
                "mask_expand_percent": (
                    "FLOAT",
                    {
                        # 0.20: room for the new head shape; on IMG_0010 Kevin's
                        # own_sim went 0.50 -> 0.73 (0.10 -> 0.20), IMG_0014 unchanged.
                        "default": 0.20,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Grows the mask before rendering, as a fraction of the head size (0.20 = 20% of "
                        "the larger side of the head box).\n"
                        "Gives the new head room for a different face shape / hairstyle. Too small: identity "
                        "stays stuck to the old outline. Too large: background around the head gets "
                        "re-rendered too.",
                    },
                ),
                "mask_blend_pixels": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 128,
                        "tooltip": "Width of the soft transition where the new head is blended into the picture, "
                        "in pixels of the rendered crop (converted to image pixels automatically).\n"
                        "Larger = softer, less visible seam; 0 = hard edge. See feather_direction for which "
                        "side of the mask edge the ramp sits on.",
                    },
                ),
                "context_expand_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "How much surrounding picture the head crop includes, relative to the head "
                        "mask (2.0 = crop twice as big as the head).\n"
                        "More context gives Qwen the scene's light and pose, but in groups also pulls "
                        "neighbouring faces into the crop (they stay protected in masked mode). 1.0 = tight "
                        "crop around the head.",
                    },
                ),
                "output_padding": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 256,
                        "tooltip": "Extra margin in image pixels added around the crop on every side, on top of "
                        "context_expand_factor.",
                    },
                ),
                "delta_clamp": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.05,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Maximum change per pixel at the mask edge when the new head is blended back "
                        "(0.35 = at most 35% of the black-to-white range). The cap relaxes to unlimited "
                        "~24 px inside the mask, so the face itself changes fully while the seam stays calm.\n"
                        "Raise towards 1.0 if a visible ring or halo of the OLD head remains at the edge; "
                        "lower if the seam shows a colour step.",
                    },
                ),
                "feather_direction": (
                    ["both", "inward", "outward"],
                    {
                        "default": "both",
                        "tooltip": "Where the soft blend (mask_blend_pixels) sits relative to the mask edge.\n"
                        "both: half inside, half outside - default.\n"
                        "inward: entirely inside the mask - nothing outside the head is touched, but the rim "
                        "of the new head is only partly applied.\n"
                        "outward: the head is applied at full strength, the transition reaches into the "
                        "surroundings.",
                    },
                ),
                "take_hairstyle": (
                    ["yes", "no"],
                    {
                        "default": "yes",
                        "tooltip": "Fills {hair} in the prompt.\n"
                        "yes: hairstyle and hair color come from the reference photos (tested default - "
                        "helps the identity).\n"
                        "no: the person keeps the hair from the picture; only the face changes. Combine with "
                        "mask_type = face for a stricter result.",
                    },
                ),
                "skin_tone_match": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Qwen tends to copy the skin tone and white balance of the reference photos, "
                        "which looks pale or cold in warm scene light. After rendering, the new face is "
                        "shifted (Lab colour) to the original face's cheek colour and brightness.\n"
                        "Only changed skin pixels are adjusted; sky, clothes and hair stay as rendered.\n"
                        "1.0 = full match (tested default), 0.5 = halfway, 0 = off.",
                    },
                ),
                "candidates": (
                    "INT",
                    {
                        # 3: 36 person x seed cases (3 images x 3 seeds x 4 people)
                        # never fell below own_sim 0.69; with 1 candidate 3 of 36
                        # stayed near the original (down to 0.25). ~10 s each.
                        "default": 3,
                        "min": 1,
                        "max": 8,
                        "tooltip": "Number of variants rendered per person (seed, seed+1, ...). The node keeps the "
                        "one whose face is most similar to the reference photos (InsightFace), with a "
                        "penalty if the mouth opened/closed or the head turned compared to the original. "
                        "Scores are listed in info.\n"
                        "1 = fast (~15 s per person) but some seeds miss the identity; 3 = tested default "
                        "(~33 s); each extra variant ~10 s. The prompt is encoded only once.",
                    },
                ),
                "expression_hint": (
                    ["auto", "always", "off"],
                    {
                        "default": "auto",
                        "tooltip": "Qwen likes to copy the mouth of the reference photos (neutral portraits -> "
                        "the laughing person comes out with a closed mouth). This measures the mouth in the "
                        "picture and writes it into the prompt at {expression} (laugh / toothy smile / "
                        "closed smile / neutral).\n"
                        "auto: only when the reference photos mostly show the other mouth state - "
                        "recommended, because the hint costs a little identity.\n"
                        "always: every time. off: never.",
                    },
                ),
            },
            "optional": {
                "model_override": (
                    "MODEL",
                    {
                        "tooltip": "Use a different model for THIS person only, e.g. with a character LoRA "
                        "applied. The pipe output still passes the original model on to the next node.",
                    },
                ),
                "clip_override": (
                    "CLIP",
                    {
                        "tooltip": "Use a different text encoder for this person only (e.g. LoRA-patched CLIP). "
                        "The pipe output stays unchanged.",
                    },
                ),
                "vae_override": (
                    "VAE",
                    {
                        "tooltip": "Use a different VAE for this person only. Rarely needed; the pipe output "
                        "stays unchanged.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def execute(
        self,
        image,
        pipe,
        person_data,
        ref_index,
        reference_images,
        prompt_template,
        extra_prompt,
        negative_prompt,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        latent_mode,
        ref_mode,
        max_refs,
        ref_crop,
        ref_crop_factor,
        canvas_resolution,
        ref_resolution,
        mask_type,
        mask_expand_percent,
        mask_blend_pixels,
        context_expand_factor,
        output_padding,
        delta_clamp,
        feather_direction,
        expression_hint="auto",
        take_hairstyle="yes",
        candidates=3,
        skin_tone_match=1.0,
        model_override=None,
        clip_override=None,
        vae_override=None,
        unique_id=None,
    ):
        use = resolve_pipe(pipe, None, model_override, clip_override, vae_override)
        result, refined, canvases, sheet, info = run_ref_head_detail(
            image,
            use["model"],
            use["clip"],
            use["vae"],
            person_data,
            ref_index,
            reference_images,
            prompt_template,
            extra_prompt,
            negative_prompt,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            latent_mode,
            ref_mode,
            max_refs,
            ref_crop,
            ref_crop_factor,
            canvas_resolution,
            ref_resolution,
            mask_type,
            mask_expand_percent,
            mask_blend_pixels,
            context_expand_factor,
            output_padding,
            delta_clamp,
            feather_direction,
            expression_hint,
            take_hairstyle=take_hairstyle,
            candidates=candidates,
            skin_tone_match=skin_tone_match,
        )
        text = "\n".join(info)
        print("[FVM QwenRef] " + text.replace("\n", "\n[FVM QwenRef] "))
        return {
            "ui": {"text": [text]},
            "result": (result, pipe, person_data, refined, canvases, sheet, text),
        }


NODE_CLASS_MAPPINGS = {
    "FVM_QwenPipe": FVM_QwenPipe,
    "FVM_QwenRefHeadDetailer": FVM_QwenRefHeadDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FVM_QwenPipe": "Qwen 2.1 Pipe",
    "FVM_QwenRefHeadDetailer": "Qwen 2.1 Ref Head Detailer",
}

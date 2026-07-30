"""Download the OCR models used by nodes/utils/ocr_backend.py.

Nothing in FVMtools ever downloads at runtime - this script is the only place
models are fetched, and only when a human runs it.

  ONNX backend (default)   PP-OCRv4 detection + recognition + angle classifier
                           + the PaddleOCR character dictionary.
  EasyOCR backend          the CRAFT detector and the english_g2 recognizer
                           weights, unzipped into <target>/easyocr/.

Run with --dry-run first: it prints every URL it would fetch and queries the
HuggingFace tree API so the real remote filenames can be confirmed before any
bytes move. sha256 values are never hardcoded - they are computed from the
files that actually land on disk and printed in the summary table.

Examples:
    python scripts/fetch_ocr_models.py --dry-run
    python scripts/fetch_ocr_models.py --backend onnx
    python scripts/fetch_ocr_models.py --backend both --target D:\\AI\\AI_Models\\onnx\\ocr
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

FALLBACK_TARGET = r"D:\AI\AI_Models\onnx\ocr"

# ── Remote sources ───────────────────────────────────────────────────────────

HF_REPO = "SWHL/RapidOCR"
HF_SUBDIR = "PP-OCRv4"
HF_RESOLVE = f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_SUBDIR}/"
# The angle classifier was never re-released for v4 - it still lives in PP-OCRv1/.
HF_RESOLVE_V1 = f"https://huggingface.co/{HF_REPO}/resolve/main/PP-OCRv1/"
HF_TREE_APIS = [
    (f"{HF_REPO} : {HF_SUBDIR}/  (det + rec)",
     f"https://huggingface.co/api/models/{HF_REPO}/tree/main/{HF_SUBDIR}"),
    (f"{HF_REPO} : PP-OCRv1/  (angle classifier)",
     f"https://huggingface.co/api/models/{HF_REPO}/tree/main/PP-OCRv1"),
    (f"{HF_REPO} : repo root",
     f"https://huggingface.co/api/models/{HF_REPO}/tree/main"),
]

# key -> (local filename, url, name_confirmed)
# name_confirmed=False means the remote filename is a guess - --dry-run lists
# the real trees so a human can correct it before any real download.
ONNX_SOURCES = {
    "det": ("ch_PP-OCRv4_det_infer.onnx", HF_RESOLVE + "ch_PP-OCRv4_det_infer.onnx", True),
    "rec": ("ch_PP-OCRv4_rec_infer.onnx", HF_RESOLVE + "ch_PP-OCRv4_rec_infer.onnx", True),
    "cls": ("ch_ppocr_mobile_v2.0_cls_infer.onnx",
            HF_RESOLVE_V1 + "ch_ppocr_mobile_v2.0_cls_infer.onnx", True),
    # Character dictionary is not part of the RapidOCR repo - canonical source
    # is the PaddleOCR repository itself.
    "keys": ("ppocr_keys_v1.txt",
             "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/"
             "ppocr/utils/ppocr_keys_v1.txt", True),
}

EASYOCR_SOURCES = {
    "detector": ("craft_mlt_25k.pth", "craft_mlt_25k.zip",
                 "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/"
                 "craft_mlt_25k.zip"),
    "recognizer": ("english_g2.pth", "english_g2.zip",
                   "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/"
                   "english_g2.zip"),
}

EASYOCR_SUBDIR = "easyocr"
UA = {"User-Agent": "FVMtools-fetch-ocr-models/1.0"}


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def http_head(url: str) -> tuple[int, int]:
    """Return (status, content_length). (0, -1) on failure."""
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, int(resp.headers.get("Content-Length") or -1)
    except urllib.error.HTTPError as exc:
        return exc.code, -1
    except Exception:
        return 0, -1


def http_get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str) -> str:
    """Stream url to dest with a progress line. Returns the sha256."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as out:
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                pct = f"{done * 100.0 / total:5.1f}%" if total else "  ??.?%"
                sys.stdout.write(f"\r    {os.path.basename(dest)}  {pct}  "
                                 f"{done / 1048576.0:8.2f} MB")
                sys.stdout.flush()
    sys.stdout.write("\n")
    os.replace(tmp, dest)
    return h.hexdigest()


# ── Target resolution ────────────────────────────────────────────────────────

def default_target() -> str:
    try:
        from nodes.utils.ocr_backend import resolve_ocr_dir
        found = resolve_ocr_dir()
        if found:
            return found
    except Exception as exc:
        print(f"  (could not import ocr_backend.resolve_ocr_dir: {exc})")
    return FALLBACK_TARGET


# ── Dry run ──────────────────────────────────────────────────────────────────

def print_tree(url: str, label: str) -> None:
    print(f"\n  {label}")
    print(f"    {url}")
    data = http_get_json(url)
    if isinstance(data, dict) and "__error__" in data:
        print(f"    !! query failed: {data['__error__']}")
        return
    if not isinstance(data, list):
        print(f"    !! unexpected response: {str(data)[:200]}")
        return
    for entry in data:
        kind = entry.get("type", "?")
        path = entry.get("path", "?")
        size = entry.get("size", 0) or 0
        marker = "/" if kind == "directory" else " "
        print(f"    {kind:9} {size:>12,}  {os.path.basename(path)}{marker}")


def dry_run(do_onnx: bool, do_easyocr: bool, target: str) -> int:
    print("=" * 78)
    print("DRY RUN - nothing will be downloaded")
    print("=" * 78)
    print(f"target directory : {target}")
    print(f"exists           : {os.path.isdir(target)}")

    if do_onnx:
        print("\n[onnx] planned downloads")
        for key, (fname, url, confirmed) in ONNX_SOURCES.items():
            status, length = http_head(url)
            flag = "name confirmed" if confirmed else "NAME UNVERIFIED"
            size = f"{length:,} bytes" if length > 0 else "size unknown"
            reachable = "OK" if status == 200 else "!! UNREACHABLE"
            print(f"  {key:5} -> {os.path.join(target, fname)}")
            print(f"        {url}")
            print(f"        HTTP {status}  {size}  [{flag}]  {reachable}")
        print("\n[onnx] actual remote filenames (HuggingFace tree API):")
        for label, url in HF_TREE_APIS:
            print_tree(url, label)
        print("\n  ^ confirm the 'cls' and 'keys' filenames against these listings "
              "before running a real download.")

    if do_easyocr:
        print("\n[easyocr] planned downloads")
        for key, (fname, zname, url) in EASYOCR_SOURCES.items():
            status, length = http_head(url)
            size = f"{length:,} bytes" if length > 0 else "size unknown"
            print(f"  {key:11} -> {os.path.join(target, EASYOCR_SUBDIR, fname)}  (via {zname})")
            print(f"        {url}")
            print(f"        HTTP {status}  {size}")

    print("\nNo sha256 values are printed here on purpose - they are computed from")
    print("the downloaded bytes and reported in the summary table of a real run.")
    return 0


# ── Real download ────────────────────────────────────────────────────────────

def fetch_onnx(target: str, force: bool) -> list:
    print("\n[onnx] downloading PP-OCRv4 models")
    os.makedirs(target, exist_ok=True)
    rows = []
    for key, (fname, url, verified) in ONNX_SOURCES.items():
        dest = os.path.join(target, fname)
        status, remote_size = http_head(url)
        if os.path.isfile(dest) and not force:
            local_size = os.path.getsize(dest)
            if remote_size <= 0 or local_size == remote_size:
                print(f"  skip {fname} (already present, {local_size:,} bytes)")
                rows.append((dest, local_size, sha256_of(dest), "skipped"))
                continue
            print(f"  re-fetch {fname} (size {local_size:,} != remote {remote_size:,})")
        if status not in (200, 302, 0):
            print(f"  !! {fname}: HTTP {status} for {url} - skipping")
            rows.append((dest, -1, "-", f"HTTP {status}"))
            continue
        try:
            digest = download(url, dest)
        except Exception as exc:
            print(f"  !! {fname}: download failed: {exc}")
            rows.append((dest, -1, "-", f"failed: {exc}"))
            continue
        rows.append((dest, os.path.getsize(dest), digest, "downloaded"))
    return rows


def fetch_easyocr(target: str, force: bool) -> list:
    print("\n[easyocr] downloading CRAFT + english_g2 weights")
    model_dir = os.path.join(target, EASYOCR_SUBDIR)
    os.makedirs(model_dir, exist_ok=True)
    rows = []
    tmpdir = tempfile.mkdtemp(prefix="fvm_ocr_")
    try:
        for key, (fname, zname, url) in EASYOCR_SOURCES.items():
            dest = os.path.join(model_dir, fname)
            if os.path.isfile(dest) and not force:
                size = os.path.getsize(dest)
                print(f"  skip {fname} (already present, {size:,} bytes)")
                rows.append((dest, size, sha256_of(dest), "skipped"))
                continue
            zpath = os.path.join(tmpdir, zname)
            try:
                zdigest = download(url, zpath)
                print(f"    archive sha256 {zdigest}")
                with zipfile.ZipFile(zpath) as zf:
                    names = zf.namelist()
                    match = next((n for n in names if n.endswith(fname)), None)
                    if match is None:
                        print(f"  !! {zname} does not contain {fname} (has: {names})")
                        rows.append((dest, -1, "-", "not in archive"))
                        continue
                    with zf.open(match) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
            except Exception as exc:
                print(f"  !! {fname}: failed: {exc}")
                rows.append((dest, -1, "-", f"failed: {exc}"))
                continue
            rows.append((dest, os.path.getsize(dest), sha256_of(dest), "downloaded"))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return rows


def print_summary(rows: list) -> int:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'bytes':>13}  {'sha256':<64}  status / path")
    failures = 0
    for path, size, digest, state in rows:
        if size < 0:
            failures += 1
        size_s = f"{size:,}" if size >= 0 else "-"
        print(f"  {size_s:>13}  {digest:<64}  {state}")
        print(f"  {'':>13}  {'':<64}  {path}")
    ok = sum(1 for r in rows if r[1] >= 0)
    print(f"\n  {ok}/{len(rows)} files present.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["onnx", "easyocr", "both"], default="onnx")
    parser.add_argument("--target", default=None,
                        help="destination directory (default: the resolved OCR dir "
                             f"or {FALLBACK_TARGET})")
    parser.add_argument("--dry-run", action="store_true",
                        help="print URLs and the remote tree listing, download nothing")
    parser.add_argument("--force", action="store_true",
                        help="re-download files that already exist")
    args = parser.parse_args()

    target = args.target or default_target()
    do_onnx = args.backend in ("onnx", "both")
    do_easyocr = args.backend in ("easyocr", "both")

    if args.dry_run:
        return dry_run(do_onnx, do_easyocr, target)

    print(f"target directory : {target}")
    rows = []
    if do_onnx:
        rows += fetch_onnx(target, args.force)
    if do_easyocr:
        rows += fetch_easyocr(target, args.force)
    rc = print_summary(rows)
    print("\nSet [models] ocr_path in outfit_config.ini if this directory is not "
          "auto-detected.")
    return rc


if __name__ == "__main__":
    sys.exit(main())

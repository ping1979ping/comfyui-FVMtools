"""One-shot migration: flat indoor_/outdoor_ slugs -> hierarchical paths.

For location_lists/:
  indoor_<cat>_<rest>/    -> indoor/<cat>/<rest>/
  outdoor_<cat>_<rest>/   -> outdoor/<cat>/<rest>/
  indoor_indoor_activities_<rest>/  -> indoor/activities/<rest>/
  outdoor_outdoor_activities_<rest>/ -> outdoor/activities/<rest>/

For outfit_lists/:
  <style>_<gender>_<sub>/ -> <gender>/<style>/<sub>/
  <style>_<gender>/       -> <gender>/<style>/general/

Uses `git mv` so history is preserved. Idempotent: skips slugs already in
hierarchical form. Run with --dry-run first to preview.
"""

import argparse
import os
import subprocess
import sys


REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LOCATION_ROOT = os.path.join(REPO_ROOT, "location_lists")
OUTFIT_ROOT = os.path.join(REPO_ROOT, "outfit_lists")

# Multi-token category names that must be matched as a unit.
LOCATION_MULTI_TOKEN_CATEGORIES = (
    "summer_vacation",
    "winter_vacation",
    "family_vacation",
    "family_activities",
)


def location_target_path(slug: str) -> str | None:
    """Map a flat location slug to its hierarchical relative path.

    Returns POSIX-form path or None if the slug doesn't follow the indoor_/
    outdoor_ convention.
    """
    if "/" in slug:
        return None  # already hierarchical
    for prefix in ("indoor_", "outdoor_"):
        if not slug.startswith(prefix):
            continue
        scope = prefix.rstrip("_")
        tail = slug[len(prefix):]
        # Collapse double-prefix: indoor_indoor_activities_X -> activities_X
        if tail.startswith(scope + "_"):
            tail = tail[len(scope) + 1:]
        # Try multi-token categories first.
        for cat in LOCATION_MULTI_TOKEN_CATEGORIES:
            if tail.startswith(cat + "_"):
                rest = tail[len(cat) + 1:]
                return f"{scope}/{cat}/{rest}"
        # Fallback: first underscore segment is the category.
        if "_" in tail:
            cat, rest = tail.split("_", 1)
            return f"{scope}/{cat}/{rest}"
        return f"{scope}/{tail}"
    return None


def outfit_target_path(slug: str) -> str | None:
    """Map a flat outfit slug to its hierarchical relative path.

    Returns POSIX-form path or None if the slug doesn't contain female/male.
    """
    if "/" in slug:
        return None
    tokens = slug.split("_")
    if "female" in tokens:
        gender = "female"
    elif "male" in tokens:
        gender = "male"
    else:
        return None
    gender_idx = tokens.index(gender)
    pre = tokens[:gender_idx]
    post = tokens[gender_idx + 1:]
    if not pre:
        return None
    style = "_".join(pre)
    sub = "_".join(post) if post else "general"
    return f"{gender}/{style}/{sub}"


def collect_renames(root: str, mapper) -> list[tuple[str, str]]:
    """Walk top-level dirs in root and collect (slug, target_path) tuples to rename."""
    if not os.path.isdir(root):
        return []
    pairs = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            continue
        if entry.startswith(".") or entry.startswith("_"):
            continue
        target = mapper(entry)
        if target is None:
            continue  # already hierarchical or not applicable
        pairs.append((entry, target))
    return pairs


def git_mv(src_abs: str, dst_abs: str) -> None:
    """Run git mv, creating parent dirs of dst as needed."""
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    subprocess.run(
        ["git", "mv", src_abs, dst_abs],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without executing")
    parser.add_argument("--locations-only", action="store_true")
    parser.add_argument("--outfits-only", action="store_true")
    args = parser.parse_args()

    do_locations = not args.outfits_only
    do_outfits = not args.locations_only

    plans: list[tuple[str, str, str]] = []  # (root, src_slug, target_path)
    if do_locations:
        for src, target in collect_renames(LOCATION_ROOT, location_target_path):
            plans.append((LOCATION_ROOT, src, target))
    if do_outfits:
        for src, target in collect_renames(OUTFIT_ROOT, outfit_target_path):
            plans.append((OUTFIT_ROOT, src, target))

    if not plans:
        print("Nothing to migrate. All slugs already hierarchical.")
        return 0

    print(f"Planned migrations: {len(plans)}")
    for root, src, target in plans[:10]:
        kind = "LOC" if root == LOCATION_ROOT else "OUT"
        print(f"  [{kind}] {src} -> {target}")
    if len(plans) > 10:
        print(f"  ... and {len(plans) - 10} more")

    if args.dry_run:
        print("\n--dry-run; not executing.")
        return 0

    failed = []
    for i, (root, src, target) in enumerate(plans, 1):
        src_abs = os.path.join(root, src)
        dst_abs = os.path.join(root, *target.split("/"))
        try:
            git_mv(src_abs, dst_abs)
            if i % 50 == 0:
                print(f"  ({i}/{len(plans)}) ... migrated through {target}")
        except subprocess.CalledProcessError as e:
            failed.append((src, target, str(e)))

    print(f"\nMigrated {len(plans) - len(failed)}/{len(plans)}.")
    if failed:
        print("\nFailures:")
        for src, target, err in failed:
            print(f"  {src} -> {target}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

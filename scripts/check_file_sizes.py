"""Reject file blobs >= 1 MiB introduced anywhere in a commit range."""

import argparse
import html
import json
import os
import subprocess

LIMIT = 1024 * 1024


def git(*args):
    return subprocess.check_output(["git", *args])


FIX_GUIDANCE = """How to fix:
1. Reduce each file to less than 1 MiB (1,048,576 bytes), or move it to external
   storage and commit only a link.
2. Rewrite the commits containing the oversized blobs. For the latest commit,
   stage the replacement/removal and use `git commit --amend`. For older commits,
   run `git rebase -i --rebase-merges <earliest-bad-commit>^`, replacing
   `<earliest-bad-commit>` with the earliest offending commit SHA listed above.
   If it is the root commit, use `git rebase -i --rebase-merges --root` instead.
   Change `pick` to `edit` for affected commits. At each stop, fix the files,
   stage the changes with `git add -- <path>`, run `git commit --amend --no-edit`,
   then run `git rebase --continue`. Use `git rebase --abort` to cancel.
   Adding a later deletion or smaller replacement is NOT enough: every PR commit
   is checked.
3. Re-run the check. If the branch was already pushed, coordinate with collaborators
   and update it using `git push --force-with-lease`.
"""


def report(failures, commit_count):
    result = f"Checked {commit_count} commits; {len(failures)} oversized file occurrence(s)."
    print(result)
    summary = ["## File Size Gate", "", result, ""]
    if failures:
        reason = "Files must be smaller than 1 MiB (1,048,576 bytes); exactly 1 MiB is rejected."
        print(reason)
        print(FIX_GUIDANCE)
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(
                "::error title=File Size Gate::Files >= 1 MiB found in PR history. "
                "See the job summary for commits, paths, sizes and repair instructions. "
                "A later deletion does not fix earlier commits."
            )
        summary.extend([reason, "", "| Commit | File (JSON-escaped) | Bytes | MiB |", "| --- | --- | ---: | ---: |"])
        for commit, name, size in failures:
            # Paths are untrusted; do not let Markdown or HTML alter the report.
            path = html.escape(json.dumps(name)).replace("|", "&#124;").replace("`", "&#96;")
            summary.append(f"| <code>{commit}</code> | <code>{path}</code> | {size:,} | {size / LIMIT:.3f} |")
        summary.extend(["", FIX_GUIDANCE.rstrip(), ""])
    else:
        summary.append("Passed: no new file objects >= 1 MiB in the checked history.")
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a", encoding="utf-8") as stream:
            stream.write("\n".join(summary) + "\n")


def check(base, head):
    # Resolve to object IDs before using caller-supplied revisions in ranges.
    base = git("rev-parse", "--verify", f"{base}^{{commit}}").decode().strip()
    head = git("rev-parse", "--verify", f"{head}^{{commit}}").decode().strip()
    commits = git("rev-list", "--reverse", f"{base}..{head}").decode().splitlines()
    # A merge can introduce an upstream file relative to its feature-side parent.
    # Exempt objects reachable anywhere in base history, not just its current tree.
    # --no-object-names keeps arbitrary filenames out of this object-ID set.
    base_objects = set(git("rev-list", "--objects", "--no-object-names", base).splitlines())
    failures = []
    for commit in commits:
        # Compare merge commits with every parent, and handle root commits.
        changed = set(
            git(
                "diff-tree",
                "--root",
                "-m",
                "-r",
                "--no-commit-id",
                "--no-renames",
                "--name-only",
                "-z",
                "--diff-filter=ACMT",
                commit,
            ).split(b"\0")
        )
        for entry in git("ls-tree", "-r", "-l", "-z", commit).split(b"\0"):
            if not entry:
                continue
            metadata, path = entry.split(b"\t", 1)
            if path not in changed:
                continue
            name = path.decode("utf-8", errors="surrogateescape")
            mode, kind, blob, size = metadata.split()
            if kind == b"blob" and int(size) >= LIMIT and blob not in base_objects:
                print(f"FAIL {commit} {json.dumps(name)}: {int(size)} bytes (limit: < {LIMIT})")
                failures.append((commit, name, int(size)))
    report(failures, len(commits))
    return bool(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("head")
    args = parser.parse_args()
    raise SystemExit(check(args.base, args.head))

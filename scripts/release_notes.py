"""Print release notes for a tag, grouped by commit type.

Usage: python scripts/release_notes.py v0.2.0 [--repo OWNER/NAME]

The notes list the user-visible commits (feat, fix, perf) since the previous tag,
one section per type in a fixed order, then a link to the full changelog for
everything else. The same commits always give the same notes.
"""

import argparse
import re
import subprocess

SECTIONS = [
    ("feat", "Features"),
    ("fix", "Fixes"),
    ("perf", "Performance"),
]
SUBJECT = re.compile(r"^(?P<type>\w+)(\([^)]*\))?!?: (?P<text>.+)$")


def git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def previous_tag(tag):
    try:
        return git("describe", "--tags", "--abbrev=0", f"{tag}^")
    except subprocess.CalledProcessError:
        return None


def repo_slug():
    url = git("remote", "get-url", "origin")
    return re.sub(r"^.*github\.com[:/]|\.git$", "", url)


def commits(tag, previous):
    span = f"{previous}..{tag}" if previous else tag
    return git("log", "--no-merges", "--reverse", "--format=%s", span).splitlines()


def notes(tag, repo, ref=None):
    """Notes for tag; ref is what to read commits from before the tag exists."""
    ref = ref or tag
    previous = previous_tag(ref)
    groups = {key: [] for key, _ in SECTIONS}
    for subject in commits(ref, previous):
        match = SUBJECT.match(subject)
        if match and match["type"] in groups:
            text = match["text"]
            groups[match["type"]].append(f"- {text[0].upper()}{text[1:]}")

    out = []
    for key, title in SECTIONS:
        if groups[key]:
            out += [f"## {title}", "", *groups[key], ""]
    base = f"https://github.com/{repo}"
    link = f"compare/{previous}...{tag}" if previous else f"commits/{tag}"
    out.append(f"**Full Changelog**: {base}/{link}")
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag")
    parser.add_argument("--repo", help="OWNER/NAME, taken from origin by default")
    args = parser.parse_args()
    print(notes(args.tag, args.repo or repo_slug()))

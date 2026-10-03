# Contributing

Thanks for helping with 3D Gaussian Splatting Object Extraction. This page covers where to send changes and what a change should look like.

## Where to send changes

Pull requests go to develop, not to main. The main branch holds released states only, and the maintainer merges develop into it when a version is released.

Start a short-lived branch from develop, open the pull request against develop, and wait for the build check to pass.

## Before you start

For a bug or a new feature, open an issue first using the bug report or feature request template. A small fix can go straight to a pull request.

## Building and testing

The README explains how to set up the tool. The tests run with pytest:

```bash
pip install -e ".[dev,gui,images]"
pytest
```

Set QT_QPA_PLATFORM=offscreen when there is no display. Say in the pull request what you ran and, for a change to the viewer or to extraction, what scene you checked it on.

## Commit messages and pull request titles

Use a type prefix, then a lowercase imperative subject of at most 50 characters with no trailing period, for example "fix: show the next view after removing one". The types are feat, fix, docs, style, refactor, test and chore. Add a one-line body that says why the change is needed.

External pull requests are squash merged, so the pull request title becomes the commit subject. Write it in the same format.

## Pull request description

Follow the pull request template: a short summary of what changed, what you actually checked, and Fixes #N when an issue exists. Leave out anything you did not check unless a reviewer needs it to judge the change.

#!/bin/sh
# Gets llama.cpp at the base commit of the patch and applies the patch.
# Usage: scripts/setup.sh [directory]
# The default directory is llama.cpp/ in this repository. The Python scripts find it there.
set -eu

BASE=338fce6511c744893cc0d3d958b275f40552c7a4
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PATCH="$ROOT/apple5-lfm3.patch"
DIR=${1:-"$ROOT/llama.cpp"}

if [ -e "$DIR" ] && [ ! -d "$DIR/.git" ]; then
    echo "$DIR exists and is not a git checkout" >&2
    exit 1
fi
if [ ! -d "$DIR/.git" ]; then
    git init -q "$DIR"
    git -C "$DIR" remote add origin https://github.com/ggml-org/llama.cpp
fi
if ! git -C "$DIR" cat-file -e "$BASE^{commit}" 2>/dev/null; then
    git -C "$DIR" fetch -q --depth 1 origin "$BASE"
fi

if [ "$(git -C "$DIR" rev-parse HEAD 2>/dev/null)" = "$BASE" ] && git -C "$DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "The patch is already applied in $DIR"
    exit 0
fi
if [ -n "$(git -C "$DIR" status --porcelain 2>/dev/null)" ]; then
    echo "$DIR has local changes. Use a clean checkout." >&2
    exit 1
fi

git -C "$DIR" checkout -q --detach "$BASE"
git -C "$DIR" apply --check "$PATCH"
git -C "$DIR" apply "$PATCH"
echo "Applied the patch in $DIR"

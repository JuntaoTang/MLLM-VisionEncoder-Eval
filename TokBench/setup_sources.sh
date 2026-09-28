#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEST_ROOT="$SCRIPT_DIR/tokenzier_vae_scripts/image_scripts"
MANIFEST="$SCRIPT_DIR/source_versions.tsv"

mkdir -p "$DEST_ROOT"
while IFS=$'\t' read -r directory repository commit _used_by; do
    [[ -n "$directory" && "$directory" != \#* ]] || continue
    destination="$DEST_ROOT/$directory"
    if [[ ! -d "$destination/.git" ]]; then
        [[ ! -e "$destination" ]] || {
            echo "Refusing to overwrite non-git path: $destination" >&2
            exit 1
        }
        git clone "$repository" "$destination"
    fi
    current_url="$(git -C "$destination" remote get-url origin)"
    [[ "$current_url" == "$repository" ]] || {
        echo "Unexpected origin for $destination: $current_url" >&2
        exit 1
    }
    if ! git -C "$destination" cat-file -e "$commit^{commit}" 2>/dev/null; then
        git -C "$destination" fetch origin "$commit"
    fi
    git -C "$destination" checkout --detach "$commit"
    actual="$(git -C "$destination" rev-parse HEAD)"
    [[ "$actual" == "$commit" ]] || { echo "Commit mismatch for $directory" >&2; exit 1; }
    echo ">> $directory @ $actual"
done < "$MANIFEST"

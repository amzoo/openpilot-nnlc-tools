#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
    cat <<EOF
Usage: scripts/offload_data.sh DESTINATION [OPTIONS]

Move a data directory to an external disk and leave a symlink in its place
so all existing tool invocations continue to work transparently.

Arguments:
  DESTINATION          Target path on the external disk (e.g. /Volumes/MyDrive/nnlc-data)

Options:
  -s, --source DIR     Directory to move (default: ./data)
  -h, --help           Show help

Example:
  scripts/offload_data.sh /Volumes/MyDrive/nnlc-data
  scripts/offload_data.sh /Volumes/MyDrive/nnlc-output --source ./output
EOF
}

# Defaults
SOURCE="./data"
DEST=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -s|--source)
            SOURCE="${2:?--source requires a value}"
            shift 2
            ;;
        -*)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -z "$DEST" ]]; then
                DEST="$1"
            else
                echo "Error: unexpected argument: $1" >&2
                usage >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$DEST" ]]; then
    echo "Error: DESTINATION is required" >&2
    usage >&2
    exit 1
fi

# Pre-flight checks (before resolving, so symlink detection works)
if [[ ! -e "$SOURCE" && ! -L "$SOURCE" ]]; then
    echo "Error: source does not exist: $SOURCE" >&2
    exit 1
fi

if [[ -L "$SOURCE" ]]; then
    echo "Error: source is already a symlink: $SOURCE" >&2
    echo "       Already offloaded? To re-offload to a new destination:" >&2
    echo "         rm \"$SOURCE\"  # remove old symlink" >&2
    echo "         scripts/offload_data.sh \"$DEST\" --source <original-dir>" >&2
    exit 1
fi

# Resolve absolute path of SOURCE (after symlink check)
SOURCE="$(realpath "$SOURCE" 2>/dev/null || echo "$SOURCE")"

if [[ -e "$DEST" ]]; then
    echo "Error: destination already exists: $DEST" >&2
    echo "       Remove it manually if you want to proceed." >&2
    exit 1
fi

DEST_PARENT="$(dirname "$DEST")"
if [[ ! -d "$DEST_PARENT" ]]; then
    echo "Error: destination parent directory does not exist: $DEST_PARENT" >&2
    echo "       Is the external disk mounted?" >&2
    exit 1
fi

# Move and symlink
echo "Moving $SOURCE → $DEST ..."
mv "$SOURCE" "$DEST"

echo "Creating symlink $SOURCE → $DEST ..."
ln -s "$DEST" "$SOURCE"

# Verify
if [[ ! -L "$SOURCE" ]]; then
    echo "Error: symlink was not created at $SOURCE" >&2
    exit 1
fi
if [[ ! -d "$SOURCE" ]]; then
    echo "Error: symlink $SOURCE is not accessible (is the destination readable?)" >&2
    exit 1
fi

echo ""
echo "Done."
echo "  Original path : $SOURCE"
echo "  Data now at   : $DEST"
echo "  Symlink       : $SOURCE -> $DEST"
echo ""
echo "All tool invocations using $SOURCE will continue to work transparently."

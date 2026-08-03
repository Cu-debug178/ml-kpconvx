#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
用法:
  sync_share.sh <实验目录> [share根目录]

说明:
  将一个实验目录镜像到 share/<实验目录名>，并排除模型权重文件。
  未指定 share 根目录时，默认使用实验目录上级的 share/。
EOF
}

if (( $# < 1 || $# > 2 )); then
    usage >&2
    exit 2
fi

if ! command -v rsync >/dev/null 2>&1; then
    printf '错误: 未找到 rsync，请先安装 rsync。\n' >&2
    exit 1
fi

SOURCE_ARG="$1"
if [[ "$SOURCE_ARG" != /* ]]; then
    SOURCE_ARG="$(pwd)/$SOURCE_ARG"
fi
if [[ ! -d "$SOURCE_ARG" ]]; then
    printf '错误: 实验目录不存在: %s\n' "$SOURCE_ARG" >&2
    exit 1
fi

SOURCE_DIR="$(cd -- "$SOURCE_ARG" && pwd -P)"
EXPERIMENT_NAME="$(basename "$SOURCE_DIR")"

if (( $# == 2 )); then
    SHARE_ROOT="$2"
    if [[ "$SHARE_ROOT" != /* ]]; then
        SHARE_ROOT="$(pwd)/$SHARE_ROOT"
    fi
else
    SHARE_ROOT="$(dirname "$SOURCE_DIR")/share"
fi
SHARE_ROOT="$(mkdir -p "$SHARE_ROOT" && cd -- "$SHARE_ROOT" && pwd -P)"
SHARE_DIR="$SHARE_ROOT/$EXPERIMENT_NAME"

# Prevent accidentally mirroring a directory into itself or its descendants.
case "$SHARE_DIR/" in
    "$SOURCE_DIR/"*)
        printf '错误: share 目标不能位于实验目录内: %s\n' "$SHARE_DIR" >&2
        exit 1
        ;;
esac

mkdir -p "$SHARE_DIR"

# Keep directory structure while excluding common model checkpoint formats.
rsync -a --delete --delete-excluded \
    --exclude='*.tar' \
    --exclude='*.pth' \
    --exclude='*.pt' \
    --exclude='*.ckpt' \
    --exclude='*.bin' \
    --exclude='*.safetensors' \
    "$SOURCE_DIR/" "$SHARE_DIR/"

printf 'share 已同步: %s\n' "$SHARE_DIR"
printf '已排除的权重文件数量: '
find "$SHARE_DIR" -type f \( \
    -iname '*.tar' -o -iname '*.pth' -o -iname '*.pt' -o \
    -iname '*.ckpt' -o -iname '*.bin' -o -iname '*.safetensors' \
    \) -print | wc -l

#!/usr/bin/env bash
set -euo pipefail

cmd=(uv run modal run main.py)

case "${1:-}" in
  "")
    exec "${cmd[@]}"
    ;;
  smoke)
    shift
    exec "${cmd[@]}" \
      --minutes 0.1 \
      --eval-blocks 2 \
      --grad-accum 1 \
      --no-compile-model \
      --no-compile-warmup \
      --attn-implementation sdpa \
      "$@"
    ;;
  full-smoke)
    shift
    exec "${cmd[@]}" \
      --tuning-mode full \
      --minutes 0.1 \
      --eval-blocks 2 \
      --grad-accum 1 \
      --no-compile-model \
      --no-compile-warmup \
      --attn-implementation sdpa \
      "$@"
    ;;
  full-compile-smoke)
    shift
    exec "${cmd[@]}" \
      --tuning-mode full \
      --minutes 0.1 \
      --eval-blocks 2 \
      --grad-accum 1 \
      --attn-implementation sdpa \
      "$@"
    ;;
  track1)
    shift
    exec "${cmd[@]}" --track 1 "$@"
    ;;
  track2)
    shift
    exec "${cmd[@]}" --track 2 "$@"
    ;;
  track3)
    shift
    exec "${cmd[@]}" --track 3 "$@"
    ;;
  full-track1)
    shift
    exec "${cmd[@]}" --tuning-mode full --track 1 "$@"
    ;;
  full-track2)
    shift
    exec "${cmd[@]}" --tuning-mode full --track 2 "$@"
    ;;
  full-track3)
    shift
    exec "${cmd[@]}" --tuning-mode full --track 3 "$@"
    ;;
  *)
    exec "${cmd[@]}" "$@"
    ;;
esac

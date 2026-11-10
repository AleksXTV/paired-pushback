#!/usr/bin/env bash
# Full sweep (run during the sprint): every local model in $MODELS, then analysis and the PDF report.
#   bash run_all.sh                    # all models below that exist in $MODELS (default /workspace/models)
#   ONLY="qwen2.5-7b llama3.1-8b" bash run_all.sh
set -u
cd "$(dirname "$0")"
MODELS=${MODELS:-/workspace/models}
LIST=${ONLY:-"qwen2.5-0.5b llama3.2-1b qwen2.5-1.5b gemma2-2b qwen2.5-3b llama3.2-3b phi3.5-mini qwen2.5-7b llama3.1-8b"}
for m in $LIST; do
  [ -f "$MODELS/$m.gguf" ] || { echo "skip $m (no $MODELS/$m.gguf)"; continue; }
  (cd src && python run_eval.py --gguf "$MODELS/$m.gguf")
done
python src/analyze.py
python src/build_report.py
echo RUNALL_DONE

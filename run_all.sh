#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

export AUDIT_DATA_DIR="${AUDIT_DATA_DIR:-../data_audit}"
export AUDIT_RESULTS_DIR="${AUDIT_RESULTS_DIR:-$AUDIT_DATA_DIR}"
: "${MTB_RESULTS:?}" "${MTB_SOURCE:?}" "${REPORTS_DIR:?}" "${NURSING_DIR:?}"

METRICS_DIR=results/metrics
FIGURES_DIR=results/figures
LOG_DIR=logs
mkdir -p "$AUDIT_DATA_DIR" "$METRICS_DIR" "$LOG_DIR"

uv run -m preprocessing.format_mtb_errors \
    --input-dir "$MTB_RESULTS" \
    --source-dir "$MTB_SOURCE" \
    --output "$AUDIT_DATA_DIR/mtb_errors.jsonl" \
    ${MTB_PATH_REWRITE:+--summary-path-rewrite "$MTB_PATH_REWRITE"} \
    | tee "$LOG_DIR/format_mtb_errors.log"

uv run -m preprocessing.format_reports_epikrise \
    --reports-csv "$REPORTS_DIR/summaries/summaries_op_35_samples.csv" --reports-category op \
    --reports-csv "$REPORTS_DIR/summaries/summaries_radio_35_samples.csv" --reports-category radio \
    --reports-csv "$REPORTS_DIR/summaries/summaries_patho_35_samples.csv" --reports-category patho \
    --reports-results-dir "$REPORTS_DIR/summaries/survey/" \
    --reports-output "$AUDIT_DATA_DIR/reports.jsonl" \
    --epikrise-csv "$REPORTS_DIR/epikrise/epikrise_100_samples_with_generated_epikrise_hilsth_randomized.csv" \
    --epikrise-results-dir "$REPORTS_DIR/epikrise/survey/" \
    --epikrise-output "$AUDIT_DATA_DIR/epikrise.jsonl" \
    | tee "$LOG_DIR/format_reports_epikrise.log"

uv run -m preprocessing.format_nursing \
    --use-case-1-csv "$NURSING_DIR/summary_results.csv" \
    --use-case-4-csv "$NURSING_DIR/op_report_results.csv" \
    --results-dir "$NURSING_DIR/results/" \
    --output "$AUDIT_DATA_DIR/nursing.jsonl" \
    | tee "$LOG_DIR/format_nursing.log"

uv run -m preprocessing.compute_mtb_human_recall \
    --input-dir "$MTB_RESULTS" \
    --output "$METRICS_DIR/human_recall_mtb.csv" \
    | tee "$LOG_DIR/mtb_human_agreement.log"

uv run -m judging.main | tee "$LOG_DIR/judging.log"

uv run -m metrics.main \
    --input "$AUDIT_RESULTS_DIR/judge_outputs.jsonl" \
    --outdir "$METRICS_DIR" \
    --compare-models Q-235B Q-235B-Think \
    --compare-to-human \
    --compare-configs \
    | tee "$LOG_DIR/metrics.log"

uv run -m plotting.main \
    --metrics-dir "$METRICS_DIR" \
    --outdir "$FIGURES_DIR" \
    | tee "$LOG_DIR/plotting.log"

uv run -m preprocessing.text_lengths \
    --data-dir "$AUDIT_DATA_DIR" \
    --outdir "$METRICS_DIR" \
    | tee "$LOG_DIR/text_lengths.log"

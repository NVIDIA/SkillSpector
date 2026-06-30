-- Hypothetical verdict-threshold sweep: precision/recall/F1 you WOULD get if the
-- "malicious" decision were `risk_score >= threshold`, swept 0..100 by 5.
-- NOTE: the harness's real verdict is recommendation == DO_NOT_INSTALL, not a score
-- cut -- so this does not reproduce the recorded outcomes. It answers a different
-- question: where is the best score cutoff, and what precision/recall trade-off does
-- it buy? Pick the threshold that maximizes F1 (or favors recall for a security tool).
-- Failed scans (no score) are excluded.
-- Defaults to the most recent run; edit the `run` CTE (or drop the WHERE) to change scope.
WITH run AS (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1),
scored AS (
    SELECT truth_malicious, risk_score
    FROM evaluation
    WHERE run_id IN (SELECT run_id FROM run)
      AND status <> 'Error'
      AND risk_score IS NOT NULL
),
counts AS (
    SELECT
        t.threshold,
        count(*) FILTER (WHERE s.truth_malicious     AND s.risk_score >= t.threshold) AS tp,
        count(*) FILTER (WHERE NOT s.truth_malicious AND s.risk_score >= t.threshold) AS fp,
        count(*) FILTER (WHERE s.truth_malicious     AND s.risk_score <  t.threshold) AS fn,
        count(*) FILTER (WHERE NOT s.truth_malicious AND s.risk_score <  t.threshold) AS tn
    FROM range(0, 101, 5) AS t(threshold)
    CROSS JOIN scored s
    GROUP BY t.threshold
)
SELECT
    threshold, tp, fp, fn, tn,
    round(tp::DOUBLE / nullif(tp + fp, 0), 3)        AS precision,
    round(tp::DOUBLE / nullif(tp + fn, 0), 3)        AS recall,
    round(2.0 * tp / nullif(2 * tp + fp + fn, 0), 3) AS f1
FROM counts
ORDER BY threshold;

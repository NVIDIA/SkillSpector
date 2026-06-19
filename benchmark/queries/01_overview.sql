-- Per-run overview: config + confusion matrix + headline metrics.
-- One row per run, newest first -- use this to compare runs at a glance.
WITH m AS (
    SELECT
        run_id,
        count(*) FILTER (WHERE outcome = 'TP')    AS tp,
        count(*) FILTER (WHERE outcome = 'FP')    AS fp,
        count(*) FILTER (WHERE outcome = 'TN')    AS tn,
        count(*) FILTER (WHERE outcome = 'FN')    AS fn,
        count(*) FILTER (WHERE outcome = 'ERROR') AS errors
    FROM evaluation
    GROUP BY run_id
)
SELECT
    r.run_id,
    r.model,
    r.use_llm,
    r.total_units,
    r.started_at,
    date_diff('second', r.started_at, r.finished_at) AS duration_s,
    m.tp, m.fp, m.tn, m.fn, m.errors,
    round(m.tp::DOUBLE / nullif(m.tp + m.fp, 0), 3)                 AS precision,
    round(m.tp::DOUBLE / nullif(m.tp + m.fn, 0), 3)                 AS recall,
    round(2.0 * m.tp / nullif(2 * m.tp + m.fp + m.fn, 0), 3)        AS f1,
    round((m.tp + m.tn)::DOUBLE / nullif(m.tp + m.tn + m.fp + m.fn, 0), 3) AS accuracy
FROM runs r
LEFT JOIN m USING (run_id)
ORDER BY r.started_at DESC;

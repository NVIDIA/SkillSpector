-- Precision / recall / F1 / accuracy broken out by unit category (skill/code/prompt).
-- Defaults to the most recent run; edit the `run` CTE (or drop the WHERE) to change scope.
WITH run AS (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1),
m AS (
    SELECT
        category,
        count(*) FILTER (WHERE outcome = 'TP')    AS tp,
        count(*) FILTER (WHERE outcome = 'FP')    AS fp,
        count(*) FILTER (WHERE outcome = 'TN')    AS tn,
        count(*) FILTER (WHERE outcome = 'FN')    AS fn,
        count(*) FILTER (WHERE outcome = 'ERROR') AS errors
    FROM evaluation
    WHERE run_id IN (SELECT run_id FROM run)
    GROUP BY category
)
SELECT
    category, tp, fp, tn, fn, errors,
    round(tp::DOUBLE / nullif(tp + fp, 0), 3)              AS precision,
    round(tp::DOUBLE / nullif(tp + fn, 0), 3)              AS recall,
    round(2.0 * tp / nullif(2 * tp + fp + fn, 0), 3)       AS f1,
    round((tp + tn)::DOUBLE / nullif(tp + tn + fp + fn, 0), 3) AS accuracy
FROM m
ORDER BY category;

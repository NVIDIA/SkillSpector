-- Precision / recall / F1 / accuracy broken out by attack vector (CI / PI / MIXED).
-- Benign units have no vector, so they fall in the NULL row -- read the malicious
-- (non-NULL) rows for recall and the NULL row mainly for benign TN/FP behavior.
-- Defaults to the most recent run; edit the `run` CTE (or drop the WHERE) to change scope.
WITH run AS (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1),
m AS (
    SELECT
        attack_vector,
        count(*) FILTER (WHERE outcome = 'TP')    AS tp,
        count(*) FILTER (WHERE outcome = 'FP')    AS fp,
        count(*) FILTER (WHERE outcome = 'TN')    AS tn,
        count(*) FILTER (WHERE outcome = 'FN')    AS fn,
        count(*) FILTER (WHERE outcome = 'ERROR') AS errors
    FROM evaluation
    WHERE run_id IN (SELECT run_id FROM run)
    GROUP BY attack_vector
)
SELECT
    attack_vector, tp, fp, tn, fn, errors,
    round(tp::DOUBLE / nullif(tp + fp, 0), 3)        AS precision,
    round(tp::DOUBLE / nullif(tp + fn, 0), 3)        AS recall,
    round(2.0 * tp / nullif(2 * tp + fp + fn, 0), 3) AS f1
FROM m
ORDER BY attack_vector NULLS LAST;

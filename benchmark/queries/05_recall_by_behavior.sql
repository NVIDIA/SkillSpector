-- Recall per malicious behavior (B1..B15), worst first -- i.e. which attack
-- behaviors SkillSpector most often misses. Behavior is only defined for malicious
-- units, so this is a recall (not precision) view. `errored` units are scans that
-- failed outright and are excluded from the recall denominator (shown separately).
-- Defaults to the most recent run; edit the `run` CTE (or drop the WHERE) to change scope.
WITH run AS (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1)
SELECT
    behavior,
    count(*) FILTER (WHERE outcome IN ('TP', 'FN')) AS scored_malicious,
    count(*) FILTER (WHERE outcome = 'TP')          AS caught,
    count(*) FILTER (WHERE outcome = 'FN')          AS missed,
    count(*) FILTER (WHERE outcome = 'ERROR')       AS errored,
    round(
        count(*) FILTER (WHERE outcome = 'TP')::DOUBLE
        / nullif(count(*) FILTER (WHERE outcome IN ('TP', 'FN')), 0), 3
    ) AS recall
FROM evaluation
WHERE run_id IN (SELECT run_id FROM run)
  AND behavior IS NOT NULL
GROUP BY behavior
ORDER BY recall ASC NULLS LAST, scored_malicious DESC;

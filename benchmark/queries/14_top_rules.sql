-- Which detection rules fire most, and whether they fire on malicious or benign
-- units. `on_benign` is the false-positive driver: a rule that fires heavily on
-- benign units is hurting precision. `on_malicious` with few `on_benign` is a
-- clean, discriminating rule. Counts are issue occurrences (a unit can have many).
-- Defaults to the most recent run; edit the `run` CTE (or drop the WHERE) to change scope.
WITH run AS (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1)
SELECT
    i.rule_id,
    i.category,
    i.severity,
    count(*)                                          AS occurrences,
    count(DISTINCT i.unit_path)                       AS units,
    count(*) FILTER (WHERE e.truth_malicious)         AS on_malicious,
    count(*) FILTER (WHERE NOT e.truth_malicious)     AS on_benign
FROM issues i
JOIN evaluation e USING (run_id, unit_path)
WHERE i.run_id IN (SELECT run_id FROM run)
GROUP BY i.rule_id, i.category, i.severity
ORDER BY occurrences DESC;

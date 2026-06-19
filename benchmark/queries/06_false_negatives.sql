-- FALSE NEGATIVES: malware that SkillSpector cleared (the security-critical misses).
-- Ordered by risk_score ascending so the units it was LEAST suspicious of -- the
-- worst, most confident misses -- surface first. A FN with a high risk_score was
-- "almost caught" and is more about the verdict threshold than a detection gap.
-- This reads the base tables directly (not the evaluation view) because the detail
-- columns display_name/source_path/num_issues aren't carried on the view. FN is just
-- "malicious truth + benign prediction"; c.is_malicious IS NULL means the scan
-- errored, and `= FALSE` excludes those.
-- Defaults to the most recent run; edit the `run` CTE (or drop the WHERE) to change scope.
WITH run AS (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1)
SELECT
    u.unit_path,
    u.display_name,
    u.category,
    u.corpus,
    u.attack_vector,
    u.behavior,
    c.risk_score,
    c.risk_severity,
    c.num_issues,
    u.source_path
FROM units u
JOIN classifications c USING (run_id, unit_path)
WHERE u.run_id IN (SELECT run_id FROM run)
  AND u.is_malicious            -- ground truth: malware
  AND c.is_malicious = FALSE    -- SkillSpector said benign (NULL = scan error, excluded)
ORDER BY c.risk_score ASC NULLS FIRST, u.category, u.behavior;

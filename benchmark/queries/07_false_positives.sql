-- FALSE POSITIVES: benign units flagged as malware (the trust / noise cost).
-- Shows the rules that fired so you can see WHAT tripped the verdict. Ordered by
-- risk_score descending: the most confidently-wrong flags first.
-- Reads the base tables directly (not the evaluation view) because the detail
-- columns display_name/source_path/num_issues aren't carried on the view. FP is just
-- "benign truth + malicious prediction".
-- Defaults to the most recent run; edit the `run` CTE (or drop the WHERE) to change scope.
WITH run AS (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1)
SELECT
    u.unit_path,
    u.display_name,
    u.category,
    u.corpus,
    c.risk_score,
    c.risk_severity,
    c.num_issues,
    (
        SELECT string_agg(DISTINCT i.rule_id, ', ')
        FROM issues i
        WHERE i.run_id = u.run_id AND i.unit_path = u.unit_path
    ) AS rules_fired,
    u.source_path
FROM units u
JOIN classifications c USING (run_id, unit_path)
WHERE u.run_id IN (SELECT run_id FROM run)
  AND NOT u.is_malicious        -- ground truth: benign
  AND c.is_malicious = TRUE     -- SkillSpector flagged it
ORDER BY c.risk_score DESC NULLS LAST, u.category;

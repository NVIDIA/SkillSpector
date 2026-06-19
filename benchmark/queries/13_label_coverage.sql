-- Ground-truth label provenance and coverage. label_source tells you HOW each
-- unit's truth/taxonomy was resolved:
--   inventory  -- authoritative MalSkillBench source inventory (most trusted)
--   classified -- joined from a curated _classified.json behavior corpus
--   field      -- explicit 0/1 label in the record
--   name       -- parsed from a __VECTOR_Bxx directory-name suffix
--   dir        -- only malware/benign known; fine taxonomy unresolved
--   corpus     -- best-guess from the source corpus (least trusted; see best_guess_label)
-- A large `dir`/`corpus` share means behavior- and vector-level breakdowns
-- (queries 03 and 05) cover only part of the malicious set.
-- Defaults to the most recent run; edit the `run` CTE (or drop the WHERE) to change scope.
WITH run AS (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1)
SELECT
    label_source,
    count(*)                                       AS units,
    count(*) FILTER (WHERE is_malicious)           AS malicious,
    count(*) FILTER (WHERE label IS NOT NULL)      AS with_fine_label,
    count(*) FILTER (WHERE best_guess_label IS NOT NULL) AS best_guess_only
FROM units
WHERE run_id IN (SELECT run_id FROM run)
GROUP BY label_source
ORDER BY units DESC;

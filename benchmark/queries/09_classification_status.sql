-- How each verdict was produced, with accuracy per mode. The important signal is
-- StaticAnalysisFallback: the LLM was requested but unavailable, so those units
-- were NOT classified by the model even though you ran with --llm. A large
-- fallback count means you are not measuring what you think you are.
--   LLM                   -- classified by the model
--   StaticAnalysis        -- static-only (expected under --no-llm)
--   StaticAnalysisFallback-- LLM requested but unavailable -> fell back to static
--   Error                 -- scan failed (see 08_errors.sql)
-- Defaults to the most recent run; edit the `run` CTE (or drop the WHERE) to change scope.
WITH run AS (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1)
SELECT
    status,
    count(*)                                  AS units,
    count(*) FILTER (WHERE outcome = 'TP')    AS tp,
    count(*) FILTER (WHERE outcome = 'FP')    AS fp,
    count(*) FILTER (WHERE outcome = 'TN')    AS tn,
    count(*) FILTER (WHERE outcome = 'FN')    AS fn,
    round(
        count(*) FILTER (WHERE outcome IN ('TP', 'TN'))::DOUBLE
        / nullif(count(*) FILTER (WHERE outcome IN ('TP', 'FP', 'TN', 'FN')), 0), 3
    ) AS accuracy
FROM evaluation
WHERE run_id IN (SELECT run_id FROM run)
GROUP BY status
ORDER BY units DESC;

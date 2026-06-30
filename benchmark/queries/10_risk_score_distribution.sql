-- Risk-score distribution (0-100, in buckets of 10) split by ground truth.
-- A good classifier separates the two columns: benign mass low, malicious mass
-- high. Overlap in the middle is where the verdict threshold has to make hard
-- calls. risk_score is a severity score, not a probability, but separation still
-- tells you how much signal it carries. Rows with no score (failed scans) excluded.
-- Defaults to the most recent run; edit the `run` CTE (or drop the WHERE) to change scope.
WITH run AS (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1)
SELECT
    (risk_score // 10) * 10                          AS bucket_low,
    count(*) FILTER (WHERE truth_malicious)          AS malicious,
    count(*) FILTER (WHERE NOT truth_malicious)      AS benign,
    count(*)                                         AS total
FROM evaluation
WHERE run_id IN (SELECT run_id FROM run)
  AND risk_score IS NOT NULL
GROUP BY bucket_low
ORDER BY bucket_low;

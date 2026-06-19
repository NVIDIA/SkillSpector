-- Scan errors: units whose scan failed (status = 'Error'), so they were never
-- truly classified. A high error count invalidates the headline metrics -- chase
-- these down before trusting precision/recall. The error string is
-- "<ExceptionType>: <message>"; the first query buckets by exception type, the
-- second lists samples so you can read the actual messages.
-- Defaults to the most recent run; edit the `run` CTE (or drop the WHERE) to change scope.
WITH run AS (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1)
SELECT
    split_part(error, ':', 1) AS error_type,
    count(*)                  AS n
FROM classifications
WHERE run_id IN (SELECT run_id FROM run)
  AND status = 'Error'
GROUP BY error_type
ORDER BY n DESC;

-- Sample failing units with their full error message.
WITH run AS (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1)
SELECT unit_path, error
FROM classifications
WHERE run_id IN (SELECT run_id FROM run)
  AND status = 'Error'
ORDER BY unit_path
LIMIT 30;

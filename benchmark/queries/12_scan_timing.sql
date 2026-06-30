-- Scan timing: wall-clock per scan (seconds). First query is the per-run summary
-- (avg / median / p95 / max / total CPU-seconds); second lists the slowest units.
-- Use it to gauge cost and spot pathological scans (e.g. ones hitting the timeout).
-- Defaults to the most recent run; edit the `run` CTE (or drop the WHERE) to change scope.
WITH run AS (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1)
SELECT
    count(*)                                   AS scans,
    round(avg(run_time), 2)                    AS avg_s,
    round(median(run_time), 2)                 AS median_s,
    round(quantile_cont(run_time, 0.95), 2)    AS p95_s,
    round(max(run_time), 2)                    AS max_s,
    round(sum(run_time), 1)                    AS total_cpu_s
FROM classifications
WHERE run_id IN (SELECT run_id FROM run);

-- Slowest individual scans.
WITH run AS (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1)
SELECT
    c.unit_path,
    u.category,
    c.status,
    round(c.run_time, 2) AS run_time_s,
    c.num_issues
FROM classifications c
JOIN units u USING (run_id, unit_path)
WHERE c.run_id IN (SELECT run_id FROM run)
ORDER BY c.run_time DESC NULLS LAST
LIMIT 20;

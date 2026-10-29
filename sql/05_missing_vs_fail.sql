-- Is a missing reading itself a warning sign? Fail rate when the sensor is missing vs. present.
-- Only sensors where both groups have at least 30 runs, so the rates are not driven by a handful of units.
WITH flags AS (
  SELECT m.sensor, m.value IS NULL AS is_missing, r.fail
  FROM measurements m JOIN runs r USING (run_id)
),
per_sensor AS (
  SELECT sensor,
         SUM(is_missing)                                         AS n_missing,
         SUM(NOT is_missing)                                     AS n_present,
         1.0 * SUM(fail * is_missing) / NULLIF(SUM(is_missing), 0)      AS fail_rate_missing,
         1.0 * SUM(fail * (NOT is_missing)) / NULLIF(SUM(NOT is_missing), 0) AS fail_rate_present
  FROM flags
  GROUP BY sensor
)
SELECT sensor, n_missing, n_present,
       ROUND(fail_rate_missing, 3) AS fail_rate_missing,
       ROUND(fail_rate_present, 3) AS fail_rate_present,
       ROUND(fail_rate_missing - fail_rate_present, 3) AS diff
FROM per_sensor
WHERE n_missing >= 30 AND n_present >= 30
ORDER BY ABS(fail_rate_missing - fail_rate_present) DESC
LIMIT 10;

-- Missing-value audit: share of runs with no reading, per sensor, grouped into buckets.
WITH per_sensor AS (
  SELECT sensor,
         COUNT(*)                         AS n_runs,
         COUNT(*) - COUNT(value)          AS n_missing,
         1.0 * (COUNT(*) - COUNT(value)) / COUNT(*) AS missing_share
  FROM measurements
  GROUP BY sensor
)
SELECT CASE WHEN missing_share = 0    THEN '0 none'
            WHEN missing_share < 0.05 THEN '1 under 5%'
            WHEN missing_share < 0.50 THEN '2 5% to 50%'
            ELSE                           '3 50% or more' END AS bucket,
       COUNT(*)                     AS sensors,
       ROUND(AVG(missing_share), 3) AS avg_missing_share
FROM per_sensor
GROUP BY bucket
ORDER BY bucket;

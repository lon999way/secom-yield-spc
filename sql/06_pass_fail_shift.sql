-- Screening: standardized mean difference (fail minus pass) per sensor, using the pooled standard deviation.
-- A quick SQL-only screen; src/screen.py repeats it with a rank test and false-discovery control.
WITH stats AS (
  SELECT m.sensor, r.fail,
         COUNT(m.value) AS n,
         AVG(m.value) AS mean,
         AVG(m.value * m.value) - AVG(m.value) * AVG(m.value) AS var_pop
  FROM measurements m JOIN runs r USING (run_id)
  WHERE m.value IS NOT NULL
  GROUP BY m.sensor, r.fail
),
pivot AS (
  SELECT sensor,
         MAX(CASE WHEN fail = 0 THEN n END)       AS n_pass,
         MAX(CASE WHEN fail = 1 THEN n END)       AS n_fail,
         MAX(CASE WHEN fail = 0 THEN mean END)    AS mean_pass,
         MAX(CASE WHEN fail = 1 THEN mean END)    AS mean_fail,
         MAX(CASE WHEN fail = 0 THEN var_pop END) AS var_pass,
         MAX(CASE WHEN fail = 1 THEN var_pop END) AS var_fail
  FROM stats
  GROUP BY sensor
)
SELECT sensor, n_pass, n_fail,
       ROUND((mean_fail - mean_pass) / SQRT((n_pass * var_pass + n_fail * var_fail) / (n_pass + n_fail)), 3) AS std_diff
FROM pivot
WHERE n_fail >= 30 AND var_pass + var_fail > 0
ORDER BY ABS(std_diff) DESC
LIMIT 15;

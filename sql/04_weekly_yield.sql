-- Weekly yield (weeks start on Monday) and change versus the previous week.
WITH weekly AS (
  SELECT date(ts, '-6 days', 'weekday 1') AS week_start, COUNT(*) AS runs, SUM(fail) AS fails
  FROM runs
  GROUP BY week_start
)
SELECT week_start, runs, fails,
       ROUND(1.0 - 1.0 * fails / runs, 4) AS yield,
       ROUND((1.0 - 1.0 * fails / runs) - LAG(1.0 - 1.0 * fails / runs) OVER (ORDER BY week_start), 4) AS change_vs_prev
FROM weekly
ORDER BY week_start;

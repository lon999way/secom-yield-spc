-- Daily yield with a 7-day rolling yield weighted by volume (sum of passes / sum of runs, not mean of daily %).
WITH daily AS (
  SELECT date(ts) AS day, COUNT(*) AS runs, SUM(fail) AS fails
  FROM runs
  GROUP BY day
)
SELECT day, runs, fails,
       ROUND(1.0 - 1.0 * fails / runs, 4) AS yield,
       ROUND(1.0 - 1.0 * SUM(fails) OVER w / SUM(runs) OVER w, 4) AS yield_rolling_7d
FROM daily
WINDOW w AS (ORDER BY julianday(day) RANGE BETWEEN 6 PRECEDING AND CURRENT ROW)
ORDER BY day;

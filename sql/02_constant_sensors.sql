-- Sensors that carry no information: one distinct non-null value (or no values at all).
SELECT COUNT(*) AS constant_sensors
FROM (
  SELECT sensor
  FROM measurements
  GROUP BY sensor
  HAVING COUNT(DISTINCT value) <= 1
);

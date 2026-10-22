-- Feature-engineering queries run against data/processed/finance.db.
-- Each query aggregates one signal group down to one row per customer_id.

-- name: monthly_spend_by_category
-- Average and latest-month spend per category, used for the subscription-creep
-- trend feature below.
SELECT
    customer_id,
    category,
    AVG(amount) AS avg_monthly_amount,
    MAX(CASE WHEN month = (SELECT MAX(month) FROM transactions) THEN amount END) AS latest_month_amount
FROM transactions
GROUP BY customer_id, category;

-- name: subscription_trend
-- Month-over-month growth rate of "subscriptions" spend using a window
-- function (LAG) over each customer's ordered months, then averaged across
-- every month-over-month change (11 changes for 12 months of history).
WITH sub AS (
    SELECT
        customer_id,
        month,
        amount,
        LAG(amount) OVER (PARTITION BY customer_id ORDER BY month) AS prev_amount
    FROM transactions
    WHERE category = 'subscriptions'
),
growth AS (
    SELECT
        customer_id,
        month,
        CASE WHEN prev_amount > 0 THEN (amount - prev_amount) / prev_amount ELSE 0 END AS mom_growth
    FROM sub
    WHERE prev_amount IS NOT NULL
)
SELECT
    customer_id,
    AVG(mom_growth) AS avg_subscription_mom_growth
FROM growth
GROUP BY customer_id;

-- name: holdings_summary
-- Portfolio-level unrealized gain/loss per customer, for tax-loss-harvest
-- candidate detection.
SELECT
    customer_id,
    SUM(market_value) AS total_market_value,
    SUM(cost_basis) AS total_cost_basis,
    SUM(market_value - cost_basis) AS total_unrealized_gain_loss,
    SUM(CASE WHEN market_value < cost_basis THEN cost_basis - market_value ELSE 0 END) AS harvestable_loss,
    COUNT(*) AS num_lots
FROM holdings
GROUP BY customer_id;

-- name: total_monthly_spend
-- Total discretionary spend per customer per month, used to derive
-- emergency-fund coverage in months.
SELECT
    customer_id,
    AVG(monthly_total) AS avg_monthly_spend
FROM (
    SELECT customer_id, month, SUM(amount) AS monthly_total
    FROM transactions
    GROUP BY customer_id, month
)
GROUP BY customer_id;

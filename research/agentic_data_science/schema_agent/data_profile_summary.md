# Data Profile Summary

## Column Profiles

| Column | Meaning | Role | Quality | Hypotheses |
|--------|---------|------|---------|------------|
| order_datetime |  |  |  | [] |
| year |  |  |  | [] |
| month |  |  |  | [] |
| week_of_year |  |  |  | [] |
| day_of_week |  |  |  | [] |
| order_hour |  |  |  | [] |
| is_weekend |  |  |  | [] |
| country | Represents the country where the transaction originated. | Feature | Data is well-distributed across several countries with a predominance in the United Kingdom. | 1. Transactions from the United Kingdom have a higher total value than other countries.<br>2. Countries with a lower transaction count like Sweden have a higher average transaction value.<br>3. Country-specific marketing strategies positively impact sales volume. |
| country_code | 3-letter code representing the country of each transaction. | Feature | Consistent with country, providing coded labels for countries. | 1. Country codes correlate strongly with country-specific purchasing patterns.<br>2. The use of certain country codes predicts higher shipping costs.<br>3. Country codes are better predictors for regional discounts than country names. |
| product_id | Unique identifier for each product sold. | Feature | Varied distribution across products indicates a potential for high product diversity. | 1. Products with higher sale counts like 'POST' have a higher discount rate applied.<br>2. Products with lower counts have a higher average profit margin.<br>3. Rarely sold products are linked with specific promotional campaigns. |
| customer_id |  |  |  | [] |
| unit_price_gbp |  |  |  | [] |
| quantity_sold |  |  |  | [] |
| sales_amount_gbp |  |  |  | [] |
| population_total |  |  |  | [] |
| gdp_current_usd |  |  |  | [] |
| gdp_growth_pct |  |  |  | [] |
| inflation_consumer_pct |  |  |  | [] |

## Numeric Column Statistics

### ecommerce_data

| Column | Metric | Value |
|--------|--------|-------|
| year | mean | 2,009.93 |
| year | std | 0.2564 |
| year | min | 2,009.00 |
| year | median | 2,010.00 |
| year | max | 2,010.00 |
| month | mean | 7.38 |
| month | std | 3.46 |
| month | min | 1.00 |
| month | median | 8.00 |
| month | max | 12.00 |
| week_of_year | mean | 29.92 |
| week_of_year | std | 15.00 |
| week_of_year | min | 1.00 |
| week_of_year | median | 33.00 |
| week_of_year | max | 52.00 |
| day_of_week | mean | 2.58 |
| day_of_week | std | 1.92 |
| day_of_week | min | 0.0000 |
| day_of_week | median | 2.00 |
| day_of_week | max | 6.00 |
| order_hour | mean | 12.68 |
| order_hour | std | 2.35 |
| order_hour | min | 7.00 |
| order_hour | median | 13.00 |
| order_hour | max | 20.00 |
| is_weekend | mean | 0.1540 |
| is_weekend | std | 0.3609 |
| is_weekend | min | 0.0000 |
| is_weekend | median | 0.0000 |
| is_weekend | max | 1.00 |
| customer_id | mean | 14,768.13 |
| customer_id | std | 1,799.16 |
| customer_id | min | 12,346.00 |
| customer_id | median | 14,646.00 |
| customer_id | max | 18,287.00 |
| unit_price_gbp | mean | 3.89 |
| unit_price_gbp | std | 59.75 |
| unit_price_gbp | min | 0.0010 |
| unit_price_gbp | median | 1.95 |
| unit_price_gbp | max | 10,953.50 |
| quantity_sold | mean | 18.66 |
| quantity_sold | std | 159.35 |
| quantity_sold | min | 1.00 |
| quantity_sold | median | 6.00 |
| quantity_sold | max | 19,152.00 |
| sales_amount_gbp | mean | 26.95 |
| sales_amount_gbp | std | 92.39 |
| sales_amount_gbp | min | 0.0010 |
| sales_amount_gbp | median | 14.98 |
| sales_amount_gbp | max | 10,953.50 |
| population_total | mean | 54,098,116.96 |
| population_total | std | 26,644,482.35 |
| population_total | min | 318,041.00 |
| population_total | median | 62,766,365.00 |
| population_total | max | 309,378,227.00 |
| gdp_current_usd | mean | 2,161,192,799,869.42 |
| gdp_current_usd | std | 1,115,049,256,125.82 |
| gdp_current_usd | min | 9,035,824,366.01 |
| gdp_current_usd | median | 2,485,482,596,184.71 |
| gdp_current_usd | max | 15,048,971,000,000.00 |
| gdp_growth_pct | mean | 0.4626 |
| gdp_growth_pct | std | 6.13 |
| gdp_growth_pct | min | -19.63 |
| gdp_growth_pct | median | 3.01 |
| gdp_growth_pct | max | 32.50 |
| inflation_consumer_pct | mean | 1.10 |
| inflation_consumer_pct | std | 1.66 |
| inflation_consumer_pct | min | -15.18 |
| inflation_consumer_pct | median | 1.59 |
| inflation_consumer_pct | max | 16.53 |


-- Create a view customer_financial_summary
CREATE OR REPLACE VIEW customer_financial_summary as
SELECT c.customerid, 
	c.estimatedsalary,
	cp.productid
FROM customers AS c
	-- Merge entity
	LEFT JOIN customerproducts AS cp 
	on c.customerid = cp.customerid;

    CREATE OR REPLACE VIEW customer_financial_summary AS
SELECT c.customerid, 
	-- Create a new conditional attribute
    CASE 
        WHEN AVG(c.estimatedsalary) > 150000 THEN 'Top Income'
        WHEN AVG(c.estimatedsalary) > 90000  THEN 'High Income'
        WHEN AVG(c.estimatedsalary) > 20000  THEN 'Average Income'
        ELSE 'Low Income'
    END AS customer_category,
    -- Add aggregation to the attributes
    count(distinct cp.productid) AS product_count
FROM customers AS c
	LEFT JOIN customerproducts AS cp 
	ON c.customerid = cp.customerid
-- Group the results
GROUP BY c.customerid;
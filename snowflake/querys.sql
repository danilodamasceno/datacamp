DESC TABLE ecommerceonlineretail;

-- Create a new products entity
create or replace table products(
    -- List the entity's attributes
    stockcode VARCHAR(255),
    description VARCHAR(255) 
);

CREATE OR REPLACE TABLE orders (
  	invoiceno VARCHAR(10) PRIMARY KEY,
  	invoicedate TIMESTAMP_NTZ(9),
  	unitprice NUMBER(10,2),
  	quantity NUMBER(38,0),
  	-- Add columns that will refer the foreign key 
	customerid NUMBER(38,0),
	stockcode VARCHAR(255)
);

-- Create customers table 
create or replace table customers (
  -- Define unique identifier
  customerid numeric(38,0) PRIMARY KEY,
  country VARCHAR(255)
);

CREATE OR REPLACE TABLE suppliers (

    name VARCHAR(255),

    location VARCHAR(255),

    supplier_id NUMBER(38,0)

);

-- Alter suppliers table
ALTER TABLE suppliers
-- Add new column
ADD COLUMN IF NOT EXISTS region VARCHAR(255);

-- Alter suppliers table
ALTER TABLE suppliers
-- Add the new column
ADD COLUMN IF NOT EXISTS contact VARCHAR(255);

-- Alter suppliers table
ALTER TABLE suppliers
-- Assign the unique identifier
ADD PRIMARY KEY (supplier_id);

-- Create entity
CREATE OR REPLACE TABLE batchdetails (
	-- Add numerical attribute
	batch_id NUMBER(10,0),
	-- Add characters attributes
    batch_number VARCHAR(255),
    production_notes VARCHAR(255)
);

SELECT manufacturer, 
	company_location, 
	COUNT(*) AS product_count
FROM productqualityrating
GROUP BY manufacturer, 
	company_location
-- Add a filter for occurrence count greater than 1
HAVING COUNT(*) >1;

-- Create a new entity
CREATE OR REPLACE TABLE ingredients (
	-- Add unique identifier 
    ingredient_id NUMBER (10,0) PRIMARY KEY,
  	-- Add other attributes 
    ingredient VARCHAR(255)
);

-- Create a new entity
CREATE OR REPLACE TABLE reviews(
	-- Add unique identifier 
    review_id NUMBER(10,0) PRIMARY KEY,
  	-- Add other attributes 
    review VARCHAR(255)
);

SELECT
	-- Create a sequential number
	row_number() OVER (order by  TRIM(f.value)),
	TRIM(f.value)
FROM productqualityrating,
LATERAL FLATTEN(INPUT => SPLIT(productqualityrating.ingredients, ';')) f
-- Group the data
group by TRIM(f.value);

-- Add command to insert data
INSERT INTO ingredients (ingredient_id, ingredient)
SELECT
	ROW_NUMBER() OVER (ORDER BY TRIM(f.value)),
	TRIM(f.value)
FROM productqualityrating,
LATERAL FLATTEN(INPUT => SPLIT(productqualityrating.ingredients, ';')) f
GROUP BY TRIM(f.value);

-- Modify script for review
INSERT INTO reviews (review_id, review)
SELECT
	ROW_NUMBER() OVER (ORDER BY TRIM(f.value)),
	TRIM(f.value)
FROM productqualityrating,
LATERAL FLATTEN (INPUT => SPLIT(productqualityrating.review, ';')) f
GROUP BY TRIM(f.value);
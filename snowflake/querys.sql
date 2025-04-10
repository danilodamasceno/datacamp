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


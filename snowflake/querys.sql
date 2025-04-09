DESC TABLE ecommerceonlineretail;

-- Create a new products entity
create or replace table products(
	-- List the entity's attributes
	stockcode VARCHAR(255),
    description VARCHAR(255) 
);

-- Create a new orders entity
create or replace table orders(
	-- List the invoice attributes
	invoiceno VARCHAR(10),
  	invoicedate TIMESTAMP_NTZ(9),
  	-- List the attributes related to price and quantity
  	unitprice NUMBER(10,2),
  	quantity NUMBER(38)
);
-- Create a new hub entity
CREATE OR REPLACE TABLE hub_employee (
	-- Assign automated values to the hub key
	hub_employee_key NUMBER(10,0) AUTOINCREMENT PRIMARY KEY,
	employee_id NUMBER(38,0),
	-- Add attributes for historical tracking
	load_date TIMESTAMP,
	record_source VARCHAR(255)
);

CREATE OR REPLACE TABLE hub_department (
	-- Assign automated values to the hub key
	hub_department_id NUMBER(10,0) AUTOINCREMENT PRIMARY KEY,
  	-- Add hubs key reference
	department_id NUMBER(38,0),
	-- Add attributes for historical tracking
	load_date TIMESTAMP,
	record_source VARCHAR(255)
);

CREATE OR REPLACE TABLE hub_training (
	-- Add hub key
	hub_training_key NUMBER(10,0) AUTOINCREMENT PRIMARY KEY,
    -- Add the key attribute of trainings
	training_id NUMBER(38,0),
	-- Add history tracking attributes
	load_date TIMESTAMP,
	record_source VARCHAR(255)
);

-- Create a new satellite
CREATE OR REPLACE TABLE sat_employee (
	sat_employee_key NUMBER(10,0) AUTOINCREMENT PRIMARY KEY,
	hub_employee_key NUMBER(10,0),
   	employee_name VARCHAR(255),
    gender CHAR(1),
    age NUMBER(3,0),
	-- Add history tracking attributes
	load_date TIMESTAMP,
    record_source VARCHAR(255),
	-- Add a reference to foreign hub
    FOREIGN KEY (hub_employee_key) REFERENCES hub_employee(hub_employee_key)
);

CREATE OR REPLACE TABLE sat_department (
	-- Add the satellite's unique identifier
	sat_department_key NUMBER(10,0) AUTOINCREMENT PRIMARY KEY,
	-- Add the hub's key attribute
	hub_department_key NUMBER(10,0),
	department_name VARCHAR(255),
	region VARCHAR(255),
    -- Add history tracking attributes
    load_date TIMESTAMP,
    record_source VARCHAR(255),
	-- Add a reference to foreign hub
	FOREIGN KEY (hub_department_key) REFERENCES hub_department(hub_department_key)
);

CREATE OR REPLACE TABLE link_all (
	link_key NUMBER(10,0) AUTOINCREMENT PRIMARY KEY,
	hub_employee_key NUMBER(10,0),
  	-- Add the hub's key attributes
  	hub_training_key NUMBER(10,0),
  	hub_department_key NUMBER(10,0),
  	load_date TIMESTAMP,
    record_source VARCHAR(255),
	FOREIGN KEY (hub_employee_key) REFERENCES hub_employee(hub_employee_key),
  	-- Add a relationship with the foreign hubs
  	FOREIGN KEY (hub_training_key) REFERENCES hub_training(hub_training_key),
  	FOREIGN KEY (hub_department_key) REFERENCES hub_department(hub_department_key)
);

SELECT
    hub_e.hub_employee_key,
    sat_d.department_name,
    -- Aggregate the attribute
    MAX(sat_t.avg_training_score) AS average_training
FROM hub_employee hub_e
	JOIN link_all AS li 
    ON hub_e.hub_employee_key = li.hub_employee_key
    JOIN sat_department AS sat_d 
    ON li.hub_department_key = sat_d.hub_department_key
    LEFT JOIN sat_training AS sat_t 
    ON li.hub_training_key = sat_t.hub_training_key
WHERE sat_t.awards_won = 1
-- Group the results 
GROUP BY  hub_e.hub_employee_key, sat_d.department_name;
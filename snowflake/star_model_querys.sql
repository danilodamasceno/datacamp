--Modify entity
ALTER TABLE IF EXISTS employees
RENAME TO dim_employees;

ALTER TABLE IF EXISTS trainings
RENAME TO dim_trainings;

-- Create new entity
CREATE OR REPLACE TABLE dim_date (
  	-- Add unique identifier
    date_id NUMBER(10,0) PRIMARY KEY,
  	-- Add new attributes to register date
    year NUMBER(4,0),
    month NUMBER(2,0)
);

SELECT
	-- Retrieve all attributes from the dimension
	dim_employees.*
FROM fact_employee_trainings 
	-- Merge fact table with dimension
	JOIN dim_employees
    ON fact_employee_trainings.employee_id = dim_employees.employee_id;

SELECT 
	dim_employees.*,
    dim_trainings.avg_training_score
    ,dim_departments.department_name
FROM fact_employee_trainings 
	JOIN dim_employees
    ON fact_employee_trainings.employee_id = dim_employees.employee_id
    JOIN dim_trainings 
    ON fact_employee_trainings.training_id = dim_trainings.training_id
    JOIN dim_departments 
    ON fact_employee_trainings.department_id = dim_departments.department_id
    -- Add dimension needed
    JOIN dim_date
    ON fact_employee_trainings.date_id = dim_date.date_id
WHERE dim_trainings.avg_training_score < 100   
    -- Add extra filter
    AND dim_date.year = 2023;
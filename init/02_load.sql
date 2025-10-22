SET search_path TO public;
SET client_encoding = 'UTF8';
SET session_replication_role = 'replica';

COPY branch (branch_code, region, branch_name)
FROM '/docker-entrypoint-initdb.d/data/branch.csv'
WITH (FORMAT csv, HEADER true);

COPY product (product_code, product_name, category, f_sku, spec_code_size, unit)
FROM '/docker-entrypoint-initdb.d/data/product.csv'
WITH (FORMAT csv, HEADER true);

-- INVENTORY: đã thêm branch_code
COPY inventory (product_code, branch_code, product_name, unit, quantity)
FROM '/docker-entrypoint-initdb.d/data/inventory.csv'
WITH (FORMAT csv, HEADER true);

COPY sales (date, branch_code, customer_code, product_code, quantity, square_meters, unit)
FROM '/docker-entrypoint-initdb.d/data/sales.csv'
WITH (FORMAT csv, HEADER true);

SET session_replication_role = 'origin';

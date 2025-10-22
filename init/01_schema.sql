SET search_path TO public;
SET client_encoding = 'UTF8';

-- BRANCH
DROP TABLE IF EXISTS branch CASCADE;
CREATE TABLE branch (
  branch_code   INTEGER PRIMARY KEY,
  region        TEXT    NOT NULL,
  branch_name   TEXT    NOT NULL
);

-- PRODUCT MASTER
DROP TABLE IF EXISTS product CASCADE;
CREATE TABLE product (
  product_code     VARCHAR(128) PRIMARY KEY,
  product_name     TEXT NOT NULL,
  category         TEXT,
  f_sku            TEXT,
  spec_code_size   TEXT,
  unit             TEXT NOT NULL
);

-- INVENTORY THEO CHI NHÁNH (ĐÃ THÊM branch_code)
DROP TABLE IF EXISTS inventory CASCADE;
CREATE TABLE inventory (
  product_code   VARCHAR(128) NOT NULL
                 REFERENCES product(product_code)
                 ON UPDATE CASCADE ON DELETE CASCADE,
  branch_code    INTEGER NOT NULL
                 REFERENCES branch(branch_code)
                 ON UPDATE CASCADE ON DELETE CASCADE,
  product_name   TEXT,      -- giữ để tiện xem, không bắt buộc
  unit           TEXT,
  quantity       INTEGER NOT NULL CHECK (quantity >= 0),
  PRIMARY KEY (product_code, branch_code)
);

-- SALES (giữ nguyên như trước)
DROP TABLE IF EXISTS sales CASCADE;
CREATE TABLE sales (
  id             BIGSERIAL PRIMARY KEY,
  date           DATE NOT NULL,
  branch_code    INTEGER NOT NULL
                 REFERENCES branch(branch_code)
                 ON UPDATE CASCADE ON DELETE RESTRICT,
  customer_code  VARCHAR(64) NOT NULL,
  product_code   VARCHAR(128) NOT NULL
                 REFERENCES product(product_code)
                 ON UPDATE CASCADE ON DELETE RESTRICT,
  quantity       INTEGER NOT NULL CHECK (quantity >= 0),
  square_meters  NUMERIC(12,2) NOT NULL,
  unit           TEXT NOT NULL
);


-- MindTrace NG · MySQL schema (write-up 3.12.6, class diagram Fig. 3.8)
-- The application creates these tables automatically on first run.
-- Run this file only if you prefer to create the database by hand:
--   mysql -u root -p < database/schema.sql

CREATE DATABASE IF NOT EXISTS mindtrace_ng CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE mindtrace_ng;

CREATE TABLE IF NOT EXISTS users (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  username      VARCHAR(40)  NOT NULL UNIQUE,
  full_name     VARCHAR(120) NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  role          VARCHAR(40)  NOT NULL,
  facility      VARCHAR(160) NOT NULL,
  state         VARCHAR(40)  NOT NULL,
  created_at    DATETIME     NOT NULL,
  last_login    DATETIME     NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS mri_images (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  user_id       INT          NOT NULL,
  original_name VARCHAR(255) NOT NULL,
  stored_name   VARCHAR(80)  NOT NULL UNIQUE,
  scan_ref      VARCHAR(60)  NULL,
  width         INT          NOT NULL,
  height        INT          NOT NULL,
  size_bytes    INT          NOT NULL,
  status        VARCHAR(20)  NOT NULL DEFAULT 'analysed',
  uploaded_at   DATETIME     NOT NULL,
  INDEX ix_mri_images_user_id (user_id),
  CONSTRAINT fk_image_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS predictions (
  id              INT AUTO_INCREMENT PRIMARY KEY,
  image_id        INT         NOT NULL UNIQUE,
  user_id         INT         NOT NULL,
  predicted_class VARCHAR(10) NOT NULL,          -- 'AD' or 'No AD'
  probability     FLOAT       NOT NULL,          -- probability of the predicted class
  prob_ad         FLOAT       NOT NULL,
  features_json   TEXT        NOT NULL,          -- the 69 extracted feature values
  processing_ms   FLOAT       NOT NULL,
  timing_json     TEXT        NOT NULL,
  created_at      DATETIME    NOT NULL,
  INDEX ix_predictions_user_id (user_id),
  INDEX ix_predictions_created_at (created_at),
  CONSTRAINT fk_pred_image FOREIGN KEY (image_id) REFERENCES mri_images (id) ON DELETE CASCADE,
  CONSTRAINT fk_pred_user  FOREIGN KEY (user_id)  REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS timeseries_db;

USE timeseries_db;

-- Metrics table for storing time series metrics
CREATE TABLE IF NOT EXISTS metrics (
    timestamp DateTime64(3) DEFAULT now64(3),
    organization_id String,
    team_id String,
    user_id String,
    metric_name String,
    metric_value Float64,
    metric_type String DEFAULT 'gauge', -- gauge, counter, histogram, summary
    labels Map(String, String) DEFAULT {},
    created_at DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (organization_id, team_id, metric_name, timestamp)
TTL timestamp + INTERVAL 1 YEAR;

-- Events table for storing application events
CREATE TABLE IF NOT EXISTS events (
    timestamp DateTime64(3) DEFAULT now64(3),
    organization_id String,
    team_id String,
    user_id String,
    event_type String,
    event_name String,
    event_data String, -- JSON string
    session_id String,
    ip_address String,
    user_agent String,
    created_at DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (organization_id, event_type, event_name, timestamp)
TTL timestamp + INTERVAL 6 MONTH;

-- Logs table for storing application logs
CREATE TABLE IF NOT EXISTS logs (
    timestamp DateTime64(3) DEFAULT now64(3),
    organization_id String,
    team_id String,
    user_id String,
    level String, -- DEBUG, INFO, WARN, ERROR, FATAL
    logger String,
    message String,
    context String, -- JSON string
    request_id String,
    created_at DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (organization_id, level, logger, timestamp)
TTL timestamp + INTERVAL 3 MONTH;

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    timestamp DateTime64(3) DEFAULT now64(3),
    organization_id String,
    team_id String,
    user_id String,
    endpoint String,
    method String,
    status_code UInt16,
    response_time_ms UInt32,
    request_size_bytes UInt32,
    response_size_bytes UInt32,
    user_agent String,
    ip_address String,
    created_at DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (organization_id, endpoint, method, timestamp)
TTL timestamp + INTERVAL 6 MONTH;

-- Business metrics table for KPIs
CREATE TABLE IF NOT EXISTS business_metrics (
    timestamp DateTime64(3) DEFAULT now64(3),
    organization_id String,
    team_id String,
    metric_category String, -- sales, users, engagement, etc.
    metric_name String,
    metric_value Float64,
    dimensions Map(String, String) DEFAULT {},
    created_at DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (organization_id, metric_category, metric_name, timestamp)
TTL timestamp + INTERVAL 2 YEAR;

-- Data sync logs table
CREATE TABLE IF NOT EXISTS data_sync_logs (
    timestamp DateTime64(3) DEFAULT now64(3),
    organization_id String,
    sync_id String,
    source_system String,
    target_system String,
    sync_type String, -- full, incremental, real-time
    records_processed UInt64,
    records_success UInt64,
    records_failed UInt64,
    sync_duration_ms UInt32,
    status String, -- success, failed, partial
    error_message String,
    created_at DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (organization_id, source_system, target_system, timestamp)
TTL timestamp + INTERVAL 1 YEAR;

-- Training logs table for ML/AI models
CREATE TABLE IF NOT EXISTS training_logs (
    timestamp DateTime64(3) DEFAULT now64(3),
    organization_id String,
    model_id String,
    model_name String,
    model_type String,
    training_status String, -- pending, running, completed, failed
    training_duration_ms UInt32,
    dataset_size UInt64,
    accuracy Float64,
    loss Float64,
    f1_score Float64,
    precision Float64,
    recall Float64,
    hyperparameters String, -- JSON string
    created_at DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (organization_id, model_name, model_type, timestamp)
TTL timestamp + INTERVAL 2 YEAR;

-- User activity table
CREATE TABLE IF NOT EXISTS user_activity (
    timestamp DateTime64(3) DEFAULT now64(3),
    organization_id String,
    team_id String,
    user_id String,
    activity_type String, -- login, logout, page_view, action, etc.
    activity_name String,
    activity_data String, -- JSON string
    session_id String,
    ip_address String,
    user_agent String,
    created_at DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (organization_id, user_id, activity_type, timestamp)
TTL timestamp + INTERVAL 1 YEAR;

-- Create materialized views for common aggregations

-- Daily metrics summary
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_metrics_summary
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (organization_id, team_id, metric_name, date)
AS SELECT
    toDate(timestamp) as date,
    organization_id,
    team_id,
    metric_name,
    count() as count,
    sum(metric_value) as total_value,
    avg(metric_value) as avg_value,
    min(metric_value) as min_value,
    max(metric_value) as max_value
FROM metrics
GROUP BY date, organization_id, team_id, metric_name;

-- Hourly performance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_performance_summary
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(hour)
ORDER BY (organization_id, endpoint, method, hour)
AS SELECT
    toStartOfHour(timestamp) as hour,
    organization_id,
    endpoint,
    method,
    count() as request_count,
    sum(response_time_ms) as total_response_time,
    avg(response_time_ms) as avg_response_time,
    max(response_time_ms) as max_response_time,
    countIf(status_code >= 400) as error_count
FROM performance_metrics
GROUP BY hour, organization_id, endpoint, method;

-- Daily user activity summary
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_user_activity_summary
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (organization_id, team_id, user_id, date)
AS SELECT
    toDate(timestamp) as date,
    organization_id,
    team_id,
    user_id,
    count() as activity_count,
    uniq(activity_type) as unique_activity_types,
    uniq(session_id) as unique_sessions
FROM user_activity
GROUP BY date, organization_id, team_id, user_id;

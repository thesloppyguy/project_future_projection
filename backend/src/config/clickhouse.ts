import { createClient } from '@clickhouse/client';
import { config } from './env';

export const clickhouse = createClient({
  url: config.CLICKHOUSE_URL,
  username: config.CLICKHOUSE_USERNAME,
  password: config.CLICKHOUSE_PASSWORD,
  database: config.CLICKHOUSE_DATABASE,
  request_timeout: 30000,
  max_open_connections: 10,
});

// Test connection
clickhouse.ping()
  .then(() => {
    console.log('✅ ClickHouse connected');
  })
  .catch((error) => {
    console.error('❌ ClickHouse connection error:', error);
  });

export default clickhouse;

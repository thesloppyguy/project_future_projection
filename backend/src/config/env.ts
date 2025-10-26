import { z } from 'zod';
import { envSchema } from 'env-schema';

const schema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: z.coerce.number().default(3000),
  HOST: z.string().default('0.0.0.0'),
  
  // Database
  DATABASE_URL: z.string().min(1),
  DBMATE_DATABASE_URL: z.string().min(1),
  
  // Redis
  REDIS_URL: z.string().default('redis://localhost:6379'),
  
  // ClickHouse
  CLICKHOUSE_URL: z.string().default('http://localhost:8123'),
  CLICKHOUSE_USERNAME: z.string().default('default'),
  CLICKHOUSE_PASSWORD: z.string().default('clickhouse_password'),
  CLICKHOUSE_DATABASE: z.string().default('timeseries_db'),
  
  // Auth
  BETTER_AUTH_SECRET: z.string().min(32),
  BETTER_AUTH_URL: z.string().url(),
  JWT_SECRET: z.string().min(32),
  
  // Email
  RESEND_API_KEY: z.string().min(1),
  
  // CORS
  CORS_ORIGIN: z.string().default('http://localhost:3000,http://localhost:3001'),
  
  // File uploads
  UPLOAD_DIR: z.string().default('./uploads'),
  MAX_FILE_SIZE: z.coerce.number().default(10 * 1024 * 1024), // 10MB
  
  // Logging
  LOG_LEVEL: z.enum(['fatal', 'error', 'warn', 'info', 'debug', 'trace']).default('info'),
  
  // API
  API_KEY: z.string().optional(),
});

export const config = envSchema({
  schema,
  dotenv: true,
});

export type Config = z.infer<typeof schema>;

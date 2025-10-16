import { defineConfig } from 'drizzle-kit';
import { config } from './src/config/env';

export default defineConfig({
  schema: './db/schema/schema.ts',
  out: './db/migrations',
  dialect: 'postgresql',
  dbCredentials: {
    url: process.env.DBMATE_DATABASE_URL || 'postgres://username:password@localhost:5432/sashflow_db',
  },
  verbose: true,
  strict: true,
});

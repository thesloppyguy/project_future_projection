import { defineConfig } from 'drizzle-kit';

export default defineConfig({
  schema: './src/db/schema/index.ts',
  out: './db/migrations',
  dialect: 'postgresql',
  dbCredentials: {
    url: process.env.DBMATE_DATABASE_URL || 'postgres://username:password@localhost:5432/sashflow_db',
  },
  verbose: true,
  strict: true,
});

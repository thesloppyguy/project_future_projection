import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { config } from '../config/env';
import { schema } from './schema';

// Create the connection
const connection = postgres(config.DATABASE_URL, {
  max: 10,
  idle_timeout: 20,
  connect_timeout: 10,
});

// Create the database instance
export const db = drizzle(connection, { schema });

// Export the connection for migrations
export { connection };

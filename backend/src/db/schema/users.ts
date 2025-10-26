import { pgTable, text, timestamp, boolean, uuid, varchar } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';

export const users = pgTable('users', {
  id: uuid('id').primaryKey().defaultRandom(),
  email: varchar('email', { length: 255 }).notNull().unique(),
  name: varchar('name', { length: 255 }),
  password: text('password'), // Better Auth handles this
  emailVerified: boolean('email_verified').default(false),
  status: varchar('status', { length: 50 }).default('pending'), // pending, active, suspended
  role: varchar('role', { length: 50 }).default('team_user'), // maintainer, org_admin, team_admin, team_user
  avatar: text('avatar'),
  lastLoginAt: timestamp('last_login_at'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});

export const insertUserSchema = createInsertSchema(users, {
  email: z.string().email(),
  name: z.string().min(1).max(255),
  status: z.enum(['pending', 'active', 'suspended']),
  role: z.enum(['maintainer', 'org_admin', 'team_admin', 'team_user']),
});

export const selectUserSchema = createSelectSchema(users);

export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;

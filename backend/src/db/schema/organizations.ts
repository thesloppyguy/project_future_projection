import { pgTable, text, timestamp, uuid, varchar, jsonb } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';

export const organizations = pgTable('organizations', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 255 }).notNull(),
  slug: varchar('slug', { length: 100 }).notNull().unique(),
  description: text('description'),
  settings: jsonb('settings').$type<{
    maxUsers?: number;
    maxTeams?: number;
    maxFileSize?: number;
    allowedFileTypes?: string[];
    rateLimitPerHour?: number;
    features?: string[];
  }>(),
  status: varchar('status', { length: 50 }).default('active'), // active, suspended, deleted
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});

export const insertOrganizationSchema = createInsertSchema(organizations, {
  name: z.string().min(1).max(255),
  slug: z.string().min(1).max(100).regex(/^[a-z0-9-]+$/),
  status: z.enum(['active', 'suspended', 'deleted']),
});

export const selectOrganizationSchema = createSelectSchema(organizations);

export type Organization = typeof organizations.$inferSelect;
export type NewOrganization = typeof organizations.$inferInsert;

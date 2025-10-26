import { pgTable, text, timestamp, uuid, varchar, jsonb } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';
import { organizations } from './organizations';

export const teams = pgTable('teams', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 255 }).notNull(),
  description: text('description'),
  organizationId: uuid('organization_id').notNull().references(() => organizations.id, { onDelete: 'cascade' }),
  settings: jsonb('settings').$type<{
    maxMembers?: number;
    permissions?: string[];
  }>(),
  status: varchar('status', { length: 50 }).default('active'), // active, archived
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});

export const insertTeamSchema = createInsertSchema(teams, {
  name: z.string().min(1).max(255),
  status: z.enum(['active', 'archived']),
});

export const selectTeamSchema = createSelectSchema(teams);

export type Team = typeof teams.$inferSelect;
export type NewTeam = typeof teams.$inferInsert;

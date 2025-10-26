import { pgTable, timestamp, uuid, varchar, unique } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';
import { users } from './users';
import { teams } from './teams';

export const teamMembers = pgTable('team_members', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  teamId: uuid('team_id').notNull().references(() => teams.id, { onDelete: 'cascade' }),
  role: varchar('role', { length: 50 }).notNull(), // team_admin, team_user
  status: varchar('status', { length: 50 }).default('active'), // active, suspended
  joinedAt: timestamp('joined_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
}, (table) => ({
  userTeamUnique: unique().on(table.userId, table.teamId),
}));

export const insertTeamMemberSchema = createInsertSchema(teamMembers, {
  role: z.enum(['team_admin', 'team_user']),
  status: z.enum(['active', 'suspended']),
});

export const selectTeamMemberSchema = createSelectSchema(teamMembers);

export type TeamMember = typeof teamMembers.$inferSelect;
export type NewTeamMember = typeof teamMembers.$inferInsert;

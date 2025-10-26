import { pgTable, timestamp, uuid, varchar, unique } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';
import { users } from './users';
import { organizations } from './organizations';

export const organizationMembers = pgTable('organization_members', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  organizationId: uuid('organization_id').notNull().references(() => organizations.id, { onDelete: 'cascade' }),
  role: varchar('role', { length: 50 }).notNull(), // org_admin, team_admin, team_user
  status: varchar('status', { length: 50 }).default('active'), // active, suspended
  joinedAt: timestamp('joined_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
}, (table) => ({
  userOrgUnique: unique().on(table.userId, table.organizationId),
}));

export const insertOrganizationMemberSchema = createInsertSchema(organizationMembers, {
  role: z.enum(['org_admin', 'team_admin', 'team_user']),
  status: z.enum(['active', 'suspended']),
});

export const selectOrganizationMemberSchema = createSelectSchema(organizationMembers);

export type OrganizationMember = typeof organizationMembers.$inferSelect;
export type NewOrganizationMember = typeof organizationMembers.$inferInsert;

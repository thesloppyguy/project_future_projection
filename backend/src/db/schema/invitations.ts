import { pgTable, text, timestamp, uuid, varchar, boolean } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';
import { organizations } from './organizations';
import { teams } from './teams';

export const invitations = pgTable('invitations', {
  id: uuid('id').primaryKey().defaultRandom(),
  email: varchar('email', { length: 255 }).notNull(),
  token: varchar('token', { length: 255 }).notNull().unique(),
  role: varchar('role', { length: 50 }).notNull(), // org_admin, team_admin, team_user
  organizationId: uuid('organization_id').notNull().references(() => organizations.id, { onDelete: 'cascade' }),
  teamId: uuid('team_id').references(() => teams.id, { onDelete: 'cascade' }),
  invitedBy: uuid('invited_by').notNull(), // user ID
  status: varchar('status', { length: 50 }).default('pending'), // pending, accepted, expired, cancelled
  expiresAt: timestamp('expires_at').notNull(),
  acceptedAt: timestamp('accepted_at'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});

export const insertInvitationSchema = createInsertSchema(invitations, {
  email: z.string().email(),
  role: z.enum(['org_admin', 'team_admin', 'team_user']),
  status: z.enum(['pending', 'accepted', 'expired', 'cancelled']),
});

export const selectInvitationSchema = createSelectSchema(invitations);

export type Invitation = typeof invitations.$inferSelect;
export type NewInvitation = typeof invitations.$inferInsert;

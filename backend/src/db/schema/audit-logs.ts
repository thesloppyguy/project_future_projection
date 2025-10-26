import { pgTable, text, timestamp, uuid, varchar, jsonb } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';
import { users } from './users';
import { organizations } from './organizations';

export const auditLogs = pgTable('audit_logs', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => users.id, { onDelete: 'set null' }),
  organizationId: uuid('organization_id').references(() => organizations.id, { onDelete: 'cascade' }),
  action: varchar('action', { length: 100 }).notNull(), // user.created, org.updated, team.deleted, etc.
  resource: varchar('resource', { length: 100 }).notNull(), // user, organization, team, etc.
  resourceId: uuid('resource_id'),
  metadata: jsonb('metadata').$type<Record<string, any>>(),
  ipAddress: varchar('ip_address', { length: 45 }),
  userAgent: text('user_agent'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export const insertAuditLogSchema = createInsertSchema(auditLogs, {
  action: z.string().min(1).max(100),
  resource: z.string().min(1).max(100),
});

export const selectAuditLogSchema = createSelectSchema(auditLogs);

export type AuditLog = typeof auditLogs.$inferSelect;
export type NewAuditLog = typeof auditLogs.$inferInsert;

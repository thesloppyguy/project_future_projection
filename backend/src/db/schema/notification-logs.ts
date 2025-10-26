import { pgTable, text, timestamp, uuid, varchar, jsonb } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';
import { users } from './users';
import { organizations } from './organizations';

export const notificationLogs = pgTable('notification_logs', {
  id: uuid('id').primaryKey().defaultRandom(),
  type: varchar('type', { length: 50 }).notNull(), // email, in_app, sms, webhook
  channel: varchar('channel', { length: 100 }).notNull(), // specific channel identifier
  recipient: varchar('recipient', { length: 255 }).notNull(), // email, user_id, phone, etc.
  subject: varchar('subject', { length: 255 }),
  content: text('content'),
  metadata: jsonb('metadata').$type<{
    templateId?: string;
    variables?: Record<string, any>;
    attachments?: string[];
  }>(),
  status: varchar('status', { length: 50 }).default('pending'), // pending, sent, delivered, failed
  error: text('error'),
  userId: uuid('user_id').references(() => users.id, { onDelete: 'set null' }),
  organizationId: uuid('organization_id').references(() => organizations.id, { onDelete: 'cascade' }),
  sentAt: timestamp('sent_at'),
  deliveredAt: timestamp('delivered_at'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export const insertNotificationLogSchema = createInsertSchema(notificationLogs, {
  type: z.enum(['email', 'in_app', 'sms', 'webhook']),
  status: z.enum(['pending', 'sent', 'delivered', 'failed']),
});

export const selectNotificationLogSchema = createSelectSchema(notificationLogs);

export type NotificationLog = typeof notificationLogs.$inferSelect;
export type NewNotificationLog = typeof notificationLogs.$inferInsert;

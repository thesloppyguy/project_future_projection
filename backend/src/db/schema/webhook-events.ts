import { pgTable, text, timestamp, jsonb, integer } from 'drizzle-orm/pg-core';
import { createId } from '@paralleldrive/cuid2';

export const webhookEvents = pgTable('webhook_events', {
  id: text('id').primaryKey().$defaultFn(() => createId()),
  webhookId: text('webhook_id').notNull().references(() => webhooks.id, { onDelete: 'cascade' }),
  event: text('event').notNull(),
  payload: jsonb('payload').notNull(),
  status: text('status', { enum: ['pending', 'delivered', 'failed'] }).notNull().default('pending'),
  responseCode: text('response_code'),
  responseBody: text('response_body'),
  attempts: text('attempts').notNull().default('1'),
  deliveredAt: timestamp('delivered_at'),
  error: text('error'),
  createdAt: timestamp('created_at').notNull().defaultNow(),
  updatedAt: timestamp('updated_at').notNull().defaultNow(),
});

export type WebhookEvent = typeof webhookEvents.$inferSelect;
export type NewWebhookEvent = typeof webhookEvents.$inferInsert;

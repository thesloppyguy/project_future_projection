import { pgTable, text, timestamp, uuid, varchar, boolean, jsonb } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';
import { organizations } from './organizations';

export const webhooks = pgTable('webhooks', {
  id: uuid('id').primaryKey().defaultRandom(),
  organizationId: uuid('organization_id').notNull().references(() => organizations.id, { onDelete: 'cascade' }),
  name: varchar('name', { length: 255 }).notNull(),
  url: text('url').notNull(),
  events: jsonb('events').$type<string[]>().notNull(), // ['user.created', 'team.updated', etc.]
  secret: varchar('secret', { length: 255 }),
  isActive: boolean('is_active').default(true),
  retryCount: varchar('retry_count', { length: 10 }).default('3'),
  timeout: varchar('timeout', { length: 10 }).default('30'), // seconds
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});

export const webhookEvents = pgTable('webhook_events', {
  id: uuid('id').primaryKey().defaultRandom(),
  webhookId: uuid('webhook_id').notNull().references(() => webhooks.id, { onDelete: 'cascade' }),
  event: varchar('event', { length: 100 }).notNull(),
  payload: jsonb('payload').notNull(),
  status: varchar('status', { length: 50 }).default('pending'), // pending, delivered, failed
  responseCode: varchar('response_code', { length: 10 }),
  responseBody: text('response_body'),
  attempts: varchar('attempts', { length: 10 }).default('0'),
  nextRetryAt: timestamp('next_retry_at'),
  deliveredAt: timestamp('delivered_at'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export const insertWebhookSchema = createInsertSchema(webhooks, {
  name: z.string().min(1).max(255),
  url: z.string().url(),
  events: z.array(z.string()),
  retryCount: z.string().regex(/^\d+$/),
  timeout: z.string().regex(/^\d+$/),
});

export const insertWebhookEventSchema = createInsertSchema(webhookEvents, {
  event: z.string().min(1).max(100),
  status: z.enum(['pending', 'delivered', 'failed']),
  attempts: z.string().regex(/^\d+$/),
});

export const selectWebhookSchema = createSelectSchema(webhooks);
export const selectWebhookEventSchema = createSelectSchema(webhookEvents);

export type Webhook = typeof webhooks.$inferSelect;
export type NewWebhook = typeof webhooks.$inferInsert;
export type WebhookEvent = typeof webhookEvents.$inferSelect;
export type NewWebhookEvent = typeof webhookEvents.$inferInsert;

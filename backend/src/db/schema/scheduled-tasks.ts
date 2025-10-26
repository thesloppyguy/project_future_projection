import { pgTable, text, timestamp, uuid, varchar, boolean, jsonb } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';
import { organizations } from './organizations';

export const scheduledTasks = pgTable('scheduled_tasks', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 255 }).notNull(),
  description: text('description'),
  type: varchar('type', { length: 100 }).notNull(), // email, notification, webhook, etc.
  schedule: varchar('schedule', { length: 100 }).notNull(), // cron expression
  payload: jsonb('payload').notNull(),
  organizationId: uuid('organization_id').references(() => organizations.id, { onDelete: 'cascade' }),
  isActive: boolean('is_active').default(true),
  lastRunAt: timestamp('last_run_at'),
  nextRunAt: timestamp('next_run_at'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});

export const taskQueue = pgTable('task_queue', {
  id: uuid('id').primaryKey().defaultRandom(),
  taskId: uuid('task_id').references(() => scheduledTasks.id, { onDelete: 'cascade' }),
  type: varchar('type', { length: 100 }).notNull(),
  payload: jsonb('payload').notNull(),
  status: varchar('status', { length: 50 }).default('pending'), // pending, active, completed, failed
  attempts: varchar('attempts', { length: 10 }).default('0'),
  maxAttempts: varchar('max_attempts', { length: 10 }).default('3'),
  error: text('error'),
  result: jsonb('result'),
  startedAt: timestamp('started_at'),
  completedAt: timestamp('completed_at'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export const insertScheduledTaskSchema = createInsertSchema(scheduledTasks, {
  name: z.string().min(1).max(255),
  type: z.string().min(1).max(100),
  schedule: z.string().min(1).max(100),
  status: z.enum(['pending', 'active', 'completed', 'failed']),
});

export const insertTaskQueueSchema = createInsertSchema(taskQueue, {
  type: z.string().min(1).max(100),
  status: z.enum(['pending', 'active', 'completed', 'failed']),
  attempts: z.string().regex(/^\d+$/),
  maxAttempts: z.string().regex(/^\d+$/),
});

export const selectScheduledTaskSchema = createSelectSchema(scheduledTasks);
export const selectTaskQueueSchema = createSelectSchema(taskQueue);

export type ScheduledTask = typeof scheduledTasks.$inferSelect;
export type NewScheduledTask = typeof scheduledTasks.$inferInsert;
export type TaskQueue = typeof taskQueue.$inferSelect;
export type NewTaskQueue = typeof taskQueue.$inferInsert;

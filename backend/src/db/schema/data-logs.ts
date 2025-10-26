import { pgTable, text, timestamp, uuid, varchar, jsonb } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';
import { users } from './users';
import { organizations } from './organizations';

export const dataSyncLogs = pgTable('data_sync_logs', {
  id: uuid('id').primaryKey().defaultRandom(),
  source: varchar('source', { length: 100 }).notNull(), // api, file_upload, webhook, etc.
  operation: varchar('operation', { length: 50 }).notNull(), // create, update, delete, sync
  resource: varchar('resource', { length: 100 }).notNull(), // user, product, order, etc.
  resourceId: uuid('resource_id'),
  status: varchar('status', { length: 50 }).default('pending'), // pending, success, failed
  recordsProcessed: varchar('records_processed', { length: 20 }).default('0'),
  recordsFailed: varchar('records_failed', { length: 20 }).default('0'),
  error: text('error'),
  metadata: jsonb('metadata').$type<{
    fileSize?: number;
    duration?: number;
    sourceUrl?: string;
  }>(),
  userId: uuid('user_id').references(() => users.id, { onDelete: 'set null' }),
  organizationId: uuid('organization_id').references(() => organizations.id, { onDelete: 'cascade' }),
  startedAt: timestamp('started_at').defaultNow().notNull(),
  completedAt: timestamp('completed_at'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export const dataTrainingLogs = pgTable('data_training_logs', {
  id: uuid('id').primaryKey().defaultRandom(),
  modelName: varchar('model_name', { length: 255 }).notNull(),
  modelType: varchar('model_type', { length: 100 }).notNull(), // ml, ai, forecasting, etc.
  status: varchar('status', { length: 50 }).default('pending'), // pending, training, completed, failed
  progress: varchar('progress', { length: 10 }).default('0'), // percentage
  metrics: jsonb('metrics').$type<{
    accuracy?: number;
    loss?: number;
    f1Score?: number;
    precision?: number;
    recall?: number;
  }>(),
  error: text('error'),
  metadata: jsonb('metadata').$type<{
    datasetSize?: number;
    trainingTime?: number;
    hyperparameters?: Record<string, any>;
  }>(),
  userId: uuid('user_id').references(() => users.id, { onDelete: 'set null' }),
  organizationId: uuid('organization_id').references(() => organizations.id, { onDelete: 'cascade' }),
  startedAt: timestamp('started_at').defaultNow().notNull(),
  completedAt: timestamp('completed_at'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export const insertDataSyncLogSchema = createInsertSchema(dataSyncLogs, {
  source: z.string().min(1).max(100),
  operation: z.enum(['create', 'update', 'delete', 'sync']),
  resource: z.string().min(1).max(100),
  status: z.enum(['pending', 'success', 'failed']),
  recordsProcessed: z.string().regex(/^\d+$/),
  recordsFailed: z.string().regex(/^\d+$/),
});

export const insertDataTrainingLogSchema = createInsertSchema(dataTrainingLogs, {
  modelName: z.string().min(1).max(255),
  modelType: z.string().min(1).max(100),
  status: z.enum(['pending', 'training', 'completed', 'failed']),
  progress: z.string().regex(/^\d+$/),
});

export const selectDataSyncLogSchema = createSelectSchema(dataSyncLogs);
export const selectDataTrainingLogSchema = createSelectSchema(dataTrainingLogs);

export type DataSyncLog = typeof dataSyncLogs.$inferSelect;
export type NewDataSyncLog = typeof dataSyncLogs.$inferInsert;
export type DataTrainingLog = typeof dataTrainingLogs.$inferSelect;
export type NewDataTrainingLog = typeof dataTrainingLogs.$inferInsert;

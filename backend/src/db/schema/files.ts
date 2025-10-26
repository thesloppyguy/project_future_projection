import { pgTable, text, timestamp, uuid, varchar, bigint, jsonb } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';
import { users } from './users';
import { organizations } from './organizations';

export const fileUploads = pgTable('file_uploads', {
  id: uuid('id').primaryKey().defaultRandom(),
  filename: varchar('filename', { length: 255 }).notNull(),
  originalName: varchar('original_name', { length: 255 }).notNull(),
  mimeType: varchar('mime_type', { length: 100 }).notNull(),
  size: bigint('size', { mode: 'number' }).notNull(),
  path: text('path').notNull(),
  url: text('url'),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  organizationId: uuid('organization_id').notNull().references(() => organizations.id, { onDelete: 'cascade' }),
  metadata: jsonb('metadata').$type<{
    width?: number;
    height?: number;
    duration?: number;
    tags?: string[];
  }>(),
  status: varchar('status', { length: 50 }).default('active'), // active, deleted
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});

export const insertFileUploadSchema = createInsertSchema(fileUploads, {
  filename: z.string().min(1).max(255),
  originalName: z.string().min(1).max(255),
  mimeType: z.string().min(1).max(100),
  size: z.number().positive(),
  status: z.enum(['active', 'deleted']),
});

export const selectFileUploadSchema = createSelectSchema(fileUploads);

export type FileUpload = typeof fileUploads.$inferSelect;
export type NewFileUpload = typeof fileUploads.$inferInsert;

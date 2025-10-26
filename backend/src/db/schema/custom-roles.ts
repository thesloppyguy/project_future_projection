import { pgTable, text, timestamp, uuid, varchar, jsonb } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';
import { organizations } from './organizations';

export const customRoleGroups = pgTable('custom_role_groups', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 255 }).notNull(),
  description: text('description'),
  permissions: jsonb('permissions').$type<string[]>().notNull(),
  organizationId: uuid('organization_id').references(() => organizations.id, { onDelete: 'cascade' }),
  isActive: boolean('is_active').default(true),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});

export const rolePermissions = pgTable('role_permissions', {
  id: uuid('id').primaryKey().defaultRandom(),
  role: varchar('role', { length: 50 }).notNull(), // maintainer, org_admin, team_admin, team_user
  permission: varchar('permission', { length: 100 }).notNull(), // org:manage, user:invite, etc.
  resource: varchar('resource', { length: 100 }).notNull(), // org, user, team, etc.
  action: varchar('action', { length: 50 }).notNull(), // create, read, update, delete, manage
  conditions: jsonb('conditions').$type<{
    ownOnly?: boolean;
    teamOnly?: boolean;
    orgOnly?: boolean;
  }>(),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export const insertCustomRoleGroupSchema = createInsertSchema(customRoleGroups, {
  name: z.string().min(1).max(255),
  permissions: z.array(z.string()),
});

export const insertRolePermissionSchema = createInsertSchema(rolePermissions, {
  role: z.enum(['maintainer', 'org_admin', 'team_admin', 'team_user']),
  permission: z.string().min(1).max(100),
  resource: z.string().min(1).max(100),
  action: z.enum(['create', 'read', 'update', 'delete', 'manage']),
});

export const selectCustomRoleGroupSchema = createSelectSchema(customRoleGroups);
export const selectRolePermissionSchema = createSelectSchema(rolePermissions);

export type CustomRoleGroup = typeof customRoleGroups.$inferSelect;
export type NewCustomRoleGroup = typeof customRoleGroups.$inferInsert;
export type RolePermission = typeof rolePermissions.$inferSelect;
export type NewRolePermission = typeof rolePermissions.$inferInsert;

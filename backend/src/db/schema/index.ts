// Export all schemas
export * from './users';
export * from './organizations';
export * from './teams';
export * from './organization-members';
export * from './team-members';
export * from './invitations';
export * from './audit-logs';
export * from './webhooks';
export * from './webhook-events';
export * from './files';
export * from './scheduled-tasks';
export * from './notification-logs';
export * from './data-logs';
export * from './custom-roles';

// Import all tables for Drizzle
import { users } from './users';
import { organizations } from './organizations';
import { teams } from './teams';
import { organizationMembers } from './organization-members';
import { teamMembers } from './team-members';
import { invitations } from './invitations';
import { auditLogs } from './audit-logs';
import { webhooks } from './webhooks';
import { webhookEvents } from './webhook-events';
import { fileUploads } from './files';
import { scheduledTasks, taskQueue } from './scheduled-tasks';
import { notificationLogs } from './notification-logs';
import { dataSyncLogs, dataTrainingLogs } from './data-logs';
import { customRoleGroups, rolePermissions } from './custom-roles';

export const schema = {
  users,
  organizations,
  teams,
  organizationMembers,
  teamMembers,
  invitations,
  auditLogs,
  webhooks,
  webhookEvents,
  fileUploads,
  scheduledTasks,
  taskQueue,
  notificationLogs,
  dataSyncLogs,
  dataTrainingLogs,
  customRoleGroups,
  rolePermissions,
};

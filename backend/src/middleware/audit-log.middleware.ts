import { FastifyRequest, FastifyReply } from 'fastify';
import { db } from '../db';
import { auditLogs } from '../db/schema';
import { AuthenticatedRequest } from './auth.middleware';

export interface AuditLogRequest extends AuthenticatedRequest {
  auditContext?: {
    action: string;
    resource: string;
    resourceId?: string;
    metadata?: Record<string, any>;
  };
}

export async function auditLog(
  action: string,
  resource: string,
  getResourceId?: (request: AuditLogRequest) => string | undefined,
  getMetadata?: (request: AuditLogRequest) => Record<string, any>
) {
  return async (request: AuditLogRequest, reply: FastifyReply) => {
    // Store audit context for use in response handler
    request.auditContext = {
      action,
      resource,
      resourceId: getResourceId?.(request),
      metadata: getMetadata?.(request),
    };
  };
}

export async function auditLogResponse(
  request: AuditLogRequest,
  reply: FastifyReply
) {
  if (!request.auditContext || !request.user) {
    return;
  }

  try {
    await db.insert(auditLogs).values({
      userId: request.user.id,
      organizationId: (request as any).organization?.id,
      action: request.auditContext.action,
      resource: request.auditContext.resource,
      resourceId: request.auditContext.resourceId,
      metadata: request.auditContext.metadata,
      ipAddress: request.ip,
      userAgent: request.headers['user-agent'],
    });
  } catch (error) {
    // Don't fail the request if audit logging fails
    console.error('Failed to log audit event:', error);
  }
}

// Helper functions for common audit scenarios
export const auditHelpers = {
  userCreated: (request: AuditLogRequest) => ({
    action: 'user.created',
    resource: 'user',
    resourceId: (request.body as any)?.id,
    metadata: { email: (request.body as any)?.email },
  }),

  userUpdated: (request: AuditLogRequest) => ({
    action: 'user.updated',
    resource: 'user',
    resourceId: (request.params as any)?.id,
    metadata: { changes: (request.body as any) },
  }),

  organizationCreated: (request: AuditLogRequest) => ({
    action: 'organization.created',
    resource: 'organization',
    resourceId: (request.body as any)?.id,
    metadata: { name: (request.body as any)?.name },
  }),

  teamCreated: (request: AuditLogRequest) => ({
    action: 'team.created',
    resource: 'team',
    resourceId: (request.body as any)?.id,
    metadata: { 
      name: (request.body as any)?.name,
      organizationId: (request.params as any)?.organizationId,
    },
  }),

  invitationSent: (request: AuditLogRequest) => ({
    action: 'invitation.sent',
    resource: 'invitation',
    resourceId: (request.body as any)?.id,
    metadata: { 
      email: (request.body as any)?.email,
      role: (request.body as any)?.role,
    },
  }),

  fileUploaded: (request: AuditLogRequest) => ({
    action: 'file.uploaded',
    resource: 'file',
    resourceId: (request.body as any)?.id,
    metadata: { 
      filename: (request.body as any)?.filename,
      size: (request.body as any)?.size,
    },
  }),
};

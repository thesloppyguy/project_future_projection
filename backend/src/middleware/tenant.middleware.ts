import { FastifyRequest, FastifyReply } from 'fastify';
import { db } from '../db';
import { organizations, organizationMembers } from '../db/schema';
import { eq, and } from 'drizzle-orm';
import { AuthenticatedRequest } from './auth.middleware';

export interface TenantRequest extends AuthenticatedRequest {
  organization?: {
    id: string;
    name: string;
    slug: string;
    settings: any;
  };
  organizationMember?: {
    id: string;
    role: string;
    status: string;
  };
}

export async function requireOrganization(
  request: TenantRequest,
  reply: FastifyReply
) {
  const organizationId = (request.params as any)?.organizationId;

  if (!organizationId) {
    return reply.status(400).send({
      error: 'Bad Request',
      message: 'Organization ID is required',
    });
  }

  if (!request.user) {
    return reply.status(401).send({
      error: 'Unauthorized',
      message: 'Authentication required',
    });
  }

  try {
    // Get organization
    const org = await db
      .select()
      .from(organizations)
      .where(eq(organizations.id, organizationId))
      .limit(1);

    if (!org.length) {
      return reply.status(404).send({
        error: 'Not Found',
        message: 'Organization not found',
      });
    }

    if (org[0].status !== 'active') {
      return reply.status(403).send({
        error: 'Forbidden',
        message: 'Organization is not active',
      });
    }

    // Check if user is member of organization (unless maintainer)
    if (request.user.role !== 'maintainer') {
      const member = await db
        .select()
        .from(organizationMembers)
        .where(and(
          eq(organizationMembers.userId, request.user.id),
          eq(organizationMembers.organizationId, organizationId),
          eq(organizationMembers.status, 'active')
        ))
        .limit(1);

      if (!member.length) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'You are not a member of this organization',
        });
      }

      request.organizationMember = member[0];
    }

    request.organization = org[0];
  } catch (error) {
    return reply.status(500).send({
      error: 'Internal Server Error',
      message: 'Failed to validate organization access',
    });
  }
}

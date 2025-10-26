import { FastifyRequest, FastifyReply } from 'fastify';
import { auth } from '../config/auth';
import { PermissionService } from '../utils/permissions';

export interface AuthenticatedRequest extends FastifyRequest {
  user?: {
    id: string;
    email: string;
    name?: string;
    role: string;
    status: string;
  };
  session?: {
    id: string;
    userId: string;
  };
}

export async function requireAuth(
  request: AuthenticatedRequest,
  reply: FastifyReply
) {
  try {
    const session = await auth.api.getSession({
      headers: request.headers as any,
    });

    if (!session) {
      return reply.status(401).send({
        error: 'Unauthorized',
        message: 'Authentication required',
      });
    }

    // Get user details
    const user = await auth.api.getUser({
      headers: request.headers as any,
    });

    if (!user) {
      return reply.status(401).send({
        error: 'Unauthorized',
        message: 'User not found',
      });
    }

    // Check if user is active
    if (user.status !== 'active') {
      return reply.status(403).send({
        error: 'Forbidden',
        message: 'Account is not active',
      });
    }

    request.user = {
      id: user.id,
      email: user.email,
      name: user.name,
      role: user.role || 'team_user',
      status: user.status || 'pending',
    };

    request.session = {
      id: session.id,
      userId: session.userId,
    };
  } catch (error) {
    return reply.status(401).send({
      error: 'Unauthorized',
      message: 'Invalid session',
    });
  }
}

export async function requirePermission(
  permission: string,
  resource?: string,
  action?: string
) {
  return async (request: AuthenticatedRequest, reply: FastifyReply) => {
    if (!request.user) {
      return reply.status(401).send({
        error: 'Unauthorized',
        message: 'Authentication required',
      });
    }

    const context = {
      userId: request.user.id,
      organizationId: (request.params as any)?.organizationId,
      teamId: (request.params as any)?.teamId,
      resourceId: (request.params as any)?.id,
    };

    const hasPermission = await PermissionService.hasPermission(
      context,
      permission,
      resource,
      action
    );

    if (!hasPermission) {
      return reply.status(403).send({
        error: 'Forbidden',
        message: 'Insufficient permissions',
      });
    }
  };
}

export async function requireMaintainer(
  request: AuthenticatedRequest,
  reply: FastifyReply
) {
  if (!request.user) {
    return reply.status(401).send({
      error: 'Unauthorized',
      message: 'Authentication required',
    });
  }

  const isMaintainer = await PermissionService.isMaintainer(request.user.id);

  if (!isMaintainer) {
    return reply.status(403).send({
      error: 'Forbidden',
      message: 'Maintainer access required',
    });
  }
}

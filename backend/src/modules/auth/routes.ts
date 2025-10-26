import { FastifyInstance } from 'fastify';
import { auth } from '../../config/auth';
import { requireAuth, AuthenticatedRequest } from '../../middleware/auth.middleware';
import { auditLog, auditLogResponse } from '../../middleware/audit-log.middleware';
import { db } from '../../db';
import { invitations, users, organizationMembers, teamMembers } from '../../db/schema';
import { eq, and, gt } from 'drizzle-orm';

export async function authRoutes(fastify: FastifyInstance) {
  // Better Auth API routes
  fastify.all('/api/auth/*', async (request, reply) => {
    return auth.handler(request, reply);
  });

  // Get current user
  fastify.get('/api/auth/me', {
    preHandler: [requireAuth],
  }, async (request: AuthenticatedRequest, reply) => {
    return {
      user: request.user,
      session: request.session,
    };
  });

  // Logout
  fastify.post('/api/auth/logout', {
    preHandler: [requireAuth],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      await auth.api.signOut({
        headers: request.headers as any,
      });

      return { message: 'Logged out successfully' };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to logout',
      });
    }
  });

  // Check if user can register (invite-only)
  fastify.get('/api/auth/can-register/:token', async (request, reply) => {
    const { token } = request.params as { token: string };

    try {
      // Check if invitation exists and is valid
      const invitation = await db.query.invitations.findFirst({
        where: (invitations, { eq, and, gt }) => and(
          eq(invitations.token, token),
          eq(invitations.status, 'pending'),
          gt(invitations.expiresAt, new Date())
        ),
      });

      if (!invitation) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Invalid or expired invitation',
        });
      }

      return {
        valid: true,
        email: invitation.email,
        role: invitation.role,
        organizationId: invitation.organizationId,
        teamId: invitation.teamId,
      };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to validate invitation',
      });
    }
  });

  // Register with invitation token
  fastify.post('/api/auth/register', {
    preHandler: [
      auditLog('user.registered', 'user', (req) => (req.body as any)?.id),
    ],
  }, async (request, reply) => {
    const { email, password, name, token } = request.body as {
      email: string;
      password: string;
      name: string;
      token: string;
    };

    try {
      // Validate invitation
      const invitation = await db.query.invitations.findFirst({
        where: (invitations, { eq, and, gt }) => and(
          eq(invitations.token, token),
          eq(invitations.email, email),
          eq(invitations.status, 'pending'),
          gt(invitations.expiresAt, new Date())
        ),
      });

      if (!invitation) {
        return reply.status(400).send({
          error: 'Bad Request',
          message: 'Invalid or expired invitation',
        });
      }

      // Register user with Better Auth
      const result = await auth.api.signUpEmail({
        body: {
          email,
          password,
          name,
        },
        headers: request.headers as any,
      });

      if (result.error) {
        return reply.status(400).send({
          error: 'Bad Request',
          message: result.error.message,
        });
      }

      // Update user role and status
      await db.update(users)
        .set({
          role: invitation.role,
          status: 'active',
          emailVerified: true,
        })
        .where(eq(users.id, result.data.user.id));

      // Add user to organization
      await db.insert(organizationMembers).values({
        userId: result.data.user.id,
        organizationId: invitation.organizationId,
        role: invitation.role,
        status: 'active',
      });

      // Add user to team if specified
      if (invitation.teamId) {
        await db.insert(teamMembers).values({
          userId: result.data.user.id,
          teamId: invitation.teamId,
          role: invitation.role === 'org_admin' ? 'team_admin' : 'team_user',
          status: 'active',
        });
      }

      // Mark invitation as accepted
      await db.update(invitations)
        .set({
          status: 'accepted',
          acceptedAt: new Date(),
        })
        .where(eq(invitations.id, invitation.id));

      return {
        user: result.data.user,
        session: result.data.session,
      };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to register user',
      });
    }
  });

  // Forgot password
  fastify.post('/api/auth/forgot-password', async (request, reply) => {
    const { email } = request.body as { email: string };

    try {
      const result = await auth.api.forgetPassword({
        body: { email },
        headers: request.headers as any,
      });

      if (result.error) {
        return reply.status(400).send({
          error: 'Bad Request',
          message: result.error.message,
        });
      }

      return { message: 'Password reset email sent' };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to send password reset email',
      });
    }
  });

  // Reset password
  fastify.post('/api/auth/reset-password', async (request, reply) => {
    const { token, password } = request.body as {
      token: string;
      password: string;
    };

    try {
      const result = await auth.api.resetPassword({
        body: { token, password },
        headers: request.headers as any,
      });

      if (result.error) {
        return reply.status(400).send({
          error: 'Bad Request',
          message: result.error.message,
        });
      }

      return { message: 'Password reset successfully' };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to reset password',
      });
    }
  });

  // Activate account
  fastify.post('/api/auth/activate', async (request, reply) => {
    const { token } = request.body as { token: string };

    try {
      const result = await auth.api.verifyEmail({
        body: { token },
        headers: request.headers as any,
      });

      if (result.error) {
        return reply.status(400).send({
          error: 'Bad Request',
          message: result.error.message,
        });
      }

      return { message: 'Account activated successfully' };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to activate account',
      });
    }
  });

  // Add audit logging to all routes
  fastify.addHook('onSend', auditLogResponse);
}

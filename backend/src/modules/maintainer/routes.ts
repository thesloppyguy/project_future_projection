import { FastifyInstance } from 'fastify';
import { requireMaintainer, AuthenticatedRequest } from '../../middleware/auth.middleware';
import { auditLog, auditLogResponse } from '../../middleware/audit-log.middleware';
import { db } from '../../db';
import { 
  users, 
  organizations, 
  teams, 
  organizationMembers, 
  teamMembers,
  invitations,
  auditLogs,
  webhooks,
  scheduledTasks,
  taskQueue,
  notificationLogs,
  dataSyncLogs,
  dataTrainingLogs
} from '../../db/schema';
import { eq, desc, count, sql } from 'drizzle-orm';
import { nanoid } from 'nanoid';
import { ClickHouseMigrationService } from '../../services/clickhouse-migration.service';

export async function maintainerRoutes(fastify: FastifyInstance) {
  // All maintainer routes require maintainer role
  fastify.addHook('preHandler', requireMaintainer);

  // Platform overview
  fastify.get('/api/maintainer/overview', async (request: AuthenticatedRequest, reply) => {
    try {
      const [
        totalUsers,
        totalOrgs,
        totalTeams,
        activeUsers,
        pendingInvitations,
        recentAuditLogs
      ] = await Promise.all([
        db.select({ count: count() }).from(users),
        db.select({ count: count() }).from(organizations),
        db.select({ count: count() }).from(teams),
        db.select({ count: count() }).from(users).where(eq(users.status, 'active')),
        db.select({ count: count() }).from(invitations).where(eq(invitations.status, 'pending')),
        db.select().from(auditLogs).orderBy(desc(auditLogs.createdAt)).limit(10)
      ]);

      return {
        stats: {
          totalUsers: totalUsers[0].count,
          totalOrgs: totalOrgs[0].count,
          totalTeams: totalTeams[0].count,
          activeUsers: activeUsers[0].count,
          pendingInvitations: pendingInvitations[0].count,
        },
        recentActivity: recentAuditLogs,
      };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to fetch platform overview',
      });
    }
  });

  // Manage organizations
  fastify.get('/api/maintainer/organizations', async (request: AuthenticatedRequest, reply) => {
    try {
      const orgs = await db
        .select({
          id: organizations.id,
          name: organizations.name,
          slug: organizations.slug,
          status: organizations.status,
          createdAt: organizations.createdAt,
          userCount: sql<number>`(
            SELECT COUNT(*) FROM ${organizationMembers} 
            WHERE ${organizationMembers.organizationId} = ${organizations.id}
          )`,
        })
        .from(organizations)
        .orderBy(desc(organizations.createdAt));

      return { organizations: orgs };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to fetch organizations',
      });
    }
  });

  // Create organization
  fastify.post('/api/maintainer/organizations', {
    preHandler: [
      auditLog('organization.created', 'organization', (req) => (req.body as any)?.id),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    const { name, slug, description, settings } = request.body as {
      name: string;
      slug: string;
      description?: string;
      settings?: any;
    };

    try {
      const [org] = await db.insert(organizations).values({
        name,
        slug,
        description,
        settings: settings || {
          maxUsers: 100,
          maxTeams: 10,
          maxFileSize: 10 * 1024 * 1024,
          allowedFileTypes: ['image/jpeg', 'image/png', 'application/pdf'],
          rateLimitPerHour: 1000,
          features: ['basic'],
        },
      }).returning();

      return { organization: org };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to create organization',
      });
    }
  });

  // Manage users
  fastify.get('/api/maintainer/users', async (request: AuthenticatedRequest, reply) => {
    try {
      const userList = await db
        .select({
          id: users.id,
          email: users.email,
          name: users.name,
          role: users.role,
          status: users.status,
          createdAt: users.createdAt,
          lastLoginAt: users.lastLoginAt,
        })
        .from(users)
        .orderBy(desc(users.createdAt));

      return { users: userList };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to fetch users',
      });
    }
  });

  // Create maintainer user
  fastify.post('/api/maintainer/users/maintainer', {
    preHandler: [
      auditLog('maintainer.created', 'user', (req) => (req.body as any)?.id),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    const { email, name, password } = request.body as {
      email: string;
      name: string;
      password: string;
    };

    try {
      // Check if user already exists
      const existingUser = await db
        .select()
        .from(users)
        .where(eq(users.email, email))
        .limit(1);

      if (existingUser.length > 0) {
        return reply.status(400).send({
          error: 'Bad Request',
          message: 'User with this email already exists',
        });
      }

      // Create maintainer user
      const [user] = await db.insert(users).values({
        email,
        name,
        role: 'maintainer',
        status: 'active',
        emailVerified: true,
        password, // In production, this should be hashed
      }).returning();

      return { user };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to create maintainer user',
      });
    }
  });

  // System logs
  fastify.get('/api/maintainer/logs/audit', async (request: AuthenticatedRequest, reply) => {
    const { page = 1, limit = 50, action, resource } = request.query as {
      page?: number;
      limit?: number;
      action?: string;
      resource?: string;
    };

    try {
      let query = db.select().from(auditLogs);

      if (action) {
        query = query.where(eq(auditLogs.action, action));
      }
      if (resource) {
        query = query.where(eq(auditLogs.resource, resource));
      }

      const logs = await query
        .orderBy(desc(auditLogs.createdAt))
        .limit(limit)
        .offset((page - 1) * limit);

      return { logs };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to fetch audit logs',
      });
    }
  });

  // System monitoring
  fastify.get('/api/maintainer/system/status', async (request: AuthenticatedRequest, reply) => {
    try {
      const [
        queueStats,
        webhookStats,
        scheduledTaskStats
      ] = await Promise.all([
        db.select({
          pending: sql<number>`COUNT(CASE WHEN status = 'pending' THEN 1 END)`,
          active: sql<number>`COUNT(CASE WHEN status = 'active' THEN 1 END)`,
          completed: sql<number>`COUNT(CASE WHEN status = 'completed' THEN 1 END)`,
          failed: sql<number>`COUNT(CASE WHEN status = 'failed' THEN 1 END)`,
        }).from(taskQueue),
        db.select({
          total: count(),
          active: sql<number>`COUNT(CASE WHEN is_active = true THEN 1 END)`,
        }).from(webhooks),
        db.select({
          total: count(),
          active: sql<number>`COUNT(CASE WHEN is_active = true THEN 1 END)`,
        }).from(scheduledTasks),
      ]);

      return {
        queues: queueStats[0],
        webhooks: webhookStats[0],
        scheduledTasks: scheduledTaskStats[0],
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to fetch system status',
      });
    }
  });

  // Impersonate user (for debugging)
  fastify.post('/api/maintainer/impersonate/:userId', {
    preHandler: [
      auditLog('user.impersonated', 'user', (req) => (req.params as any)?.userId),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    const { userId } = request.params as { userId: string };

    try {
      const user = await db
        .select()
        .from(users)
        .where(eq(users.id, userId))
        .limit(1);

      if (!user.length) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'User not found',
        });
      }

      // Create a temporary session for impersonation
      const impersonationToken = nanoid(32);
      
      // Store impersonation token in Redis (expires in 1 hour)
      await fastify.redis.setex(
        `impersonate:${impersonationToken}`,
        3600,
        JSON.stringify({
          originalUserId: request.user!.id,
          impersonatedUserId: userId,
          createdAt: new Date().toISOString(),
        })
      );

      return {
        token: impersonationToken,
        user: user[0],
        expiresIn: 3600,
      };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to create impersonation session',
      });
    }
  });

  // ClickHouse management endpoints
  fastify.get('/api/maintainer/clickhouse/status', async (request: AuthenticatedRequest, reply) => {
    try {
      const isConnected = await ClickHouseMigrationService.checkConnection();
      const version = await ClickHouseMigrationService.getVersion();
      const dbInfo = await ClickHouseMigrationService.getDatabaseInfo();

      return {
        connected: isConnected,
        version,
        database: dbInfo,
      };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get ClickHouse status',
      });
    }
  });

  fastify.get('/api/maintainer/clickhouse/tables', async (request: AuthenticatedRequest, reply) => {
    try {
      const dbInfo = await ClickHouseMigrationService.getDatabaseInfo();
      const tableStats = [];

      for (const table of dbInfo.tables) {
        const stats = await ClickHouseMigrationService.getTableStats(table);
        tableStats.push({
          name: table,
          ...stats,
        });
      }

      return { tables: tableStats };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get ClickHouse table information',
      });
    }
  });

  fastify.post('/api/maintainer/clickhouse/optimize', {
    preHandler: [
      auditLog('clickhouse.optimized', 'clickhouse', () => 'all tables'),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      await ClickHouseMigrationService.optimizeTables();

      return { success: true, message: 'ClickHouse tables optimized successfully' };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to optimize ClickHouse tables',
      });
    }
  });

  fastify.post('/api/maintainer/clickhouse/cleanup', {
    preHandler: [
      auditLog('clickhouse.cleanup', 'clickhouse', () => 'old data'),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      await ClickHouseMigrationService.cleanupOldData();

      return { success: true, message: 'ClickHouse old data cleanup completed' };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to cleanup old ClickHouse data',
      });
    }
  });

  // Add audit logging to all routes
  fastify.addHook('onSend', auditLogResponse);
}

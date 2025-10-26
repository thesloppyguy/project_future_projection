import { FastifyInstance } from 'fastify';
import { requireAuth, AuthenticatedRequest } from '../../middleware/auth.middleware';
import { requireOrganization, TenantRequest } from '../../middleware/tenant.middleware';
import { auditLog, auditLogResponse } from '../../middleware/audit-log.middleware';
import { AnalyticsService } from '../../services/analytics.service';
import { DailyIngestionService } from '../../services/daily-ingestion.service';
import { z } from 'zod';

// Validation schemas
const AnalyticsQuerySchema = z.object({
  organizationId: z.string(),
  teamId: z.string().optional(),
  startDate: z.string().transform(str => new Date(str)),
  endDate: z.string().transform(str => new Date(str)),
  granularity: z.enum(['hour', 'day', 'week', 'month']).default('day'),
  filters: z.record(z.any()).optional(),
});

const TimeSeriesQuerySchema = z.object({
  organizationId: z.string(),
  teamId: z.string().optional(),
  startDate: z.string().transform(str => new Date(str)),
  endDate: z.string().transform(str => new Date(str)),
  granularity: z.enum(['hour', 'day', 'week', 'month']).default('day'),
  metricName: z.string(),
  aggregation: z.enum(['sum', 'avg', 'count', 'max', 'min']).default('sum'),
});

const IngestionConfigSchema = z.object({
  organizationId: z.string(),
  teamId: z.string().optional(),
  date: z.string().transform(str => new Date(str)),
  metrics: z.object({
    enabled: z.boolean(),
    categories: z.array(z.string()),
  }).optional(),
  events: z.object({
    enabled: z.boolean(),
    types: z.array(z.string()),
  }).optional(),
  performance: z.object({
    enabled: z.boolean(),
    endpoints: z.array(z.string()),
  }).optional(),
  business: z.object({
    enabled: z.boolean(),
    categories: z.array(z.string()),
  }).optional(),
  userActivity: z.object({
    enabled: z.boolean(),
    types: z.array(z.string()),
  }).optional(),
});

export async function analyticsRoutes(fastify: FastifyInstance) {
  // All routes require authentication
  fastify.addHook('preHandler', requireAuth);

  // Get dashboard metrics
  fastify.get('/api/analytics/dashboard', {
    preHandler: [
      auditLog('analytics.dashboard_viewed', 'analytics', (req) => (req.query as any)?.organizationId),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const query = AnalyticsQuerySchema.parse({
        ...request.query,
        organizationId: (request as any).organization?.id || (request.query as any)?.organizationId,
      });

      if (request.user?.role !== 'maintainer' && query.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      const metrics = await AnalyticsService.getDashboardMetrics(query);

      return { data: metrics };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid query parameters',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get dashboard metrics',
      });
    }
  });

  // Get time series data
  fastify.get('/api/analytics/timeseries', {
    preHandler: [
      auditLog('analytics.timeseries_viewed', 'analytics', (req) => (req.query as any)?.metricName),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const query = TimeSeriesQuerySchema.parse({
        ...request.query,
        organizationId: (request as any).organization?.id || (request.query as any)?.organizationId,
      });

      if (request.user?.role !== 'maintainer' && query.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      const data = await AnalyticsService.getTimeSeriesData(
        query,
        query.metricName,
        query.aggregation
      );

      return { data };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid query parameters',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get time series data',
      });
    }
  });

  // Get performance metrics
  fastify.get('/api/analytics/performance', {
    preHandler: [
      auditLog('analytics.performance_viewed', 'analytics', (req) => (req.query as any)?.organizationId),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const query = AnalyticsQuerySchema.parse({
        ...request.query,
        organizationId: (request as any).organization?.id || (request.query as any)?.organizationId,
      });

      if (request.user?.role !== 'maintainer' && query.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      const metrics = await AnalyticsService.getPerformanceMetrics(query);

      return { data: metrics };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid query parameters',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get performance metrics',
      });
    }
  });

  // Get user engagement metrics
  fastify.get('/api/analytics/engagement', {
    preHandler: [
      auditLog('analytics.engagement_viewed', 'analytics', (req) => (req.query as any)?.organizationId),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const query = AnalyticsQuerySchema.parse({
        ...request.query,
        organizationId: (request as any).organization?.id || (request.query as any)?.organizationId,
      });

      if (request.user?.role !== 'maintainer' && query.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      const metrics = await AnalyticsService.getUserEngagementMetrics(query);

      return { data: metrics };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid query parameters',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get engagement metrics',
      });
    }
  });

  // Get business metrics
  fastify.get('/api/analytics/business', {
    preHandler: [
      auditLog('analytics.business_viewed', 'analytics', (req) => (req.query as any)?.organizationId),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const query = AnalyticsQuerySchema.parse({
        ...request.query,
        organizationId: (request as any).organization?.id || (request.query as any)?.organizationId,
      });

      if (request.user?.role !== 'maintainer' && query.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      const metrics = await AnalyticsService.getBusinessMetrics(query);

      return { data: metrics };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid query parameters',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get business metrics',
      });
    }
  });

  // Get system health metrics
  fastify.get('/api/analytics/system', {
    preHandler: [
      auditLog('analytics.system_viewed', 'analytics', (req) => (req.query as any)?.organizationId),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const query = AnalyticsQuerySchema.parse({
        ...request.query,
        organizationId: (request as any).organization?.id || (request.query as any)?.organizationId,
      });

      if (request.user?.role !== 'maintainer' && query.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      const metrics = await AnalyticsService.getSystemHealthMetrics(query);

      return { data: metrics };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid query parameters',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get system health metrics',
      });
    }
  });

  // Trigger daily data ingestion
  fastify.post('/api/analytics/ingestion/trigger', {
    preHandler: [
      auditLog('analytics.ingestion_triggered', 'analytics', (req) => (req.body as any)?.organizationId),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const config = IngestionConfigSchema.parse(request.body);

      if (request.user?.role !== 'maintainer' && config.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      await DailyIngestionService.runDailyIngestion(config);

      return { success: true, message: 'Daily ingestion triggered successfully' };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid ingestion configuration',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to trigger daily ingestion',
      });
    }
  });

  // Get ingestion status
  fastify.get('/api/analytics/ingestion/status', async (request: AuthenticatedRequest, reply) => {
    try {
      const { organizationId, date } = request.query as {
        organizationId: string;
        date: string;
      };

      if (request.user?.role !== 'maintainer' && organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      const status = await DailyIngestionService.getIngestionStatus(
        organizationId,
        new Date(date)
      );

      return { data: status };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get ingestion status',
      });
    }
  });

  // Schedule daily ingestion for all organizations (maintainer only)
  fastify.post('/api/analytics/ingestion/schedule-all', {
    preHandler: [
      requireAuth,
      async (request: AuthenticatedRequest, reply) => {
        if (request.user?.role !== 'maintainer') {
          return reply.status(403).send({
            error: 'Forbidden',
            message: 'Maintainer access required',
          });
        }
      },
      auditLog('analytics.ingestion_scheduled_all', 'analytics', () => 'all organizations'),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const { date } = request.body as { date: string };

      await DailyIngestionService.scheduleDailyIngestion(new Date(date));

      return { success: true, message: 'Daily ingestion scheduled for all organizations' };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to schedule daily ingestion',
      });
    }
  });

  // Add audit logging to all routes
  fastify.addHook('onSend', auditLogResponse);
}

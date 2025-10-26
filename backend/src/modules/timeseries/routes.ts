import { FastifyInstance } from 'fastify';
import { requireAuth, AuthenticatedRequest } from '../../middleware/auth.middleware';
import { requireOrganization, TenantRequest } from '../../middleware/tenant.middleware';
import { auditLog, auditLogResponse } from '../../middleware/audit-log.middleware';
import { TimeSeriesService } from '../../services/timeseries.service';
import { z } from 'zod';

// Validation schemas
const MetricInsertSchema = z.object({
  organizationId: z.string(),
  teamId: z.string().optional(),
  userId: z.string().optional(),
  metricName: z.string(),
  metricValue: z.number(),
  metricType: z.enum(['gauge', 'counter', 'histogram', 'summary']).default('gauge'),
  labels: z.record(z.string()).default({}),
});

const EventInsertSchema = z.object({
  organizationId: z.string(),
  teamId: z.string().optional(),
  userId: z.string().optional(),
  eventType: z.string(),
  eventName: z.string(),
  eventData: z.record(z.any()).default({}),
  sessionId: z.string().optional(),
  ipAddress: z.string().optional(),
  userAgent: z.string().optional(),
});

const LogInsertSchema = z.object({
  organizationId: z.string(),
  teamId: z.string().optional(),
  userId: z.string().optional(),
  level: z.enum(['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL']),
  logger: z.string(),
  message: z.string(),
  context: z.record(z.any()).default({}),
  requestId: z.string().optional(),
});

const PerformanceMetricInsertSchema = z.object({
  organizationId: z.string(),
  teamId: z.string().optional(),
  userId: z.string().optional(),
  endpoint: z.string(),
  method: z.string(),
  statusCode: z.number(),
  responseTimeMs: z.number(),
  requestSizeBytes: z.number().optional(),
  responseSizeBytes: z.number().optional(),
  userAgent: z.string().optional(),
  ipAddress: z.string().optional(),
});

const BusinessMetricInsertSchema = z.object({
  organizationId: z.string(),
  teamId: z.string().optional(),
  metricCategory: z.string(),
  metricName: z.string(),
  metricValue: z.number(),
  dimensions: z.record(z.string()).default({}),
});

const UserActivityInsertSchema = z.object({
  organizationId: z.string(),
  teamId: z.string().optional(),
  userId: z.string(),
  activityType: z.string(),
  activityName: z.string(),
  activityData: z.record(z.any()).default({}),
  sessionId: z.string().optional(),
  ipAddress: z.string().optional(),
  userAgent: z.string().optional(),
});

const QueryParamsSchema = z.object({
  organizationId: z.string(),
  teamId: z.string().optional(),
  startTime: z.string().transform(str => new Date(str)),
  endTime: z.string().transform(str => new Date(str)),
  limit: z.coerce.number().optional().default(1000),
});

export async function timeseriesRoutes(fastify: FastifyInstance) {
  // All routes require authentication
  fastify.addHook('preHandler', requireAuth);

  // Insert single metric
  fastify.post('/api/timeseries/metrics', {
    preHandler: [
      auditLog('metric.inserted', 'timeseries', (req) => (req.body as any)?.metricName),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const metric = MetricInsertSchema.parse(request.body);
      
      // Ensure user has access to the organization
      if (request.user?.role !== 'maintainer' && metric.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      await TimeSeriesService.insertMetric(metric);

      return { success: true, message: 'Metric inserted successfully' };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid metric data',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to insert metric',
      });
    }
  });

  // Insert multiple metrics in batch
  fastify.post('/api/timeseries/metrics/batch', {
    preHandler: [
      auditLog('metrics.batch_inserted', 'timeseries', (req) => `${(req.body as any)?.length || 0} metrics`),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const metrics = z.array(MetricInsertSchema).parse(request.body);
      
      // Ensure user has access to all organizations in the batch
      if (request.user?.role !== 'maintainer') {
        const userOrgId = (request as any).organization?.id;
        const hasInvalidOrg = metrics.some(metric => metric.organizationId !== userOrgId);
        
        if (hasInvalidOrg) {
          return reply.status(403).send({
            error: 'Forbidden',
            message: 'Access denied to one or more organizations',
          });
        }
      }

      await TimeSeriesService.insertMetrics(metrics);

      return { success: true, message: `${metrics.length} metrics inserted successfully` };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid metrics data',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to insert metrics',
      });
    }
  });

  // Insert event
  fastify.post('/api/timeseries/events', {
    preHandler: [
      auditLog('event.inserted', 'timeseries', (req) => (req.body as any)?.eventName),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const event = EventInsertSchema.parse(request.body);
      
      if (request.user?.role !== 'maintainer' && event.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      await TimeSeriesService.insertEvent(event);

      return { success: true, message: 'Event inserted successfully' };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid event data',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to insert event',
      });
    }
  });

  // Insert log
  fastify.post('/api/timeseries/logs', {
    preHandler: [
      auditLog('log.inserted', 'timeseries', (req) => (req.body as any)?.logger),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const log = LogInsertSchema.parse(request.body);
      
      if (request.user?.role !== 'maintainer' && log.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      await TimeSeriesService.insertLog(log);

      return { success: true, message: 'Log inserted successfully' };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid log data',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to insert log',
      });
    }
  });

  // Insert performance metric
  fastify.post('/api/timeseries/performance', {
    preHandler: [
      auditLog('performance_metric.inserted', 'timeseries', (req) => (req.body as any)?.endpoint),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const metric = PerformanceMetricInsertSchema.parse(request.body);
      
      if (request.user?.role !== 'maintainer' && metric.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      await TimeSeriesService.insertPerformanceMetric(metric);

      return { success: true, message: 'Performance metric inserted successfully' };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid performance metric data',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to insert performance metric',
      });
    }
  });

  // Insert business metric
  fastify.post('/api/timeseries/business', {
    preHandler: [
      auditLog('business_metric.inserted', 'timeseries', (req) => (req.body as any)?.metricName),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const metric = BusinessMetricInsertSchema.parse(request.body);
      
      if (request.user?.role !== 'maintainer' && metric.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      await TimeSeriesService.insertBusinessMetric(metric);

      return { success: true, message: 'Business metric inserted successfully' };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid business metric data',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to insert business metric',
      });
    }
  });

  // Insert user activity
  fastify.post('/api/timeseries/activity', {
    preHandler: [
      auditLog('user_activity.inserted', 'timeseries', (req) => (req.body as any)?.activityName),
    ],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const activity = UserActivityInsertSchema.parse(request.body);
      
      if (request.user?.role !== 'maintainer' && activity.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      await TimeSeriesService.insertUserActivity(activity);

      return { success: true, message: 'User activity inserted successfully' };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid user activity data',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to insert user activity',
      });
    }
  });

  // Query metrics
  fastify.get('/api/timeseries/metrics', async (request: AuthenticatedRequest, reply) => {
    try {
      const params = QueryParamsSchema.parse({
        ...request.query,
        organizationId: (request as any).organization?.id || (request.query as any)?.organizationId,
      });

      if (request.user?.role !== 'maintainer' && params.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      const results = await TimeSeriesService.queryMetrics({
        organizationId: params.organizationId,
        teamId: params.teamId,
        metricName: (request.query as any)?.metricName,
        startTime: params.startTime,
        endTime: params.endTime,
        limit: params.limit,
      });

      return { data: results };
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
        message: 'Failed to query metrics',
      });
    }
  });

  // Get daily metrics summary
  fastify.get('/api/timeseries/metrics/summary', async (request: AuthenticatedRequest, reply) => {
    try {
      const params = z.object({
        organizationId: z.string(),
        teamId: z.string().optional(),
        startDate: z.string().transform(str => new Date(str)),
        endDate: z.string().transform(str => new Date(str)),
      }).parse({
        ...request.query,
        organizationId: (request as any).organization?.id || (request.query as any)?.organizationId,
      });

      if (request.user?.role !== 'maintainer' && params.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      const results = await TimeSeriesService.getDailyMetricsSummary(params);

      return { data: results };
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
        message: 'Failed to get metrics summary',
      });
    }
  });

  // Get performance summary
  fastify.get('/api/timeseries/performance/summary', async (request: AuthenticatedRequest, reply) => {
    try {
      const params = z.object({
        organizationId: z.string(),
        startTime: z.string().transform(str => new Date(str)),
        endTime: z.string().transform(str => new Date(str)),
        endpoint: z.string().optional(),
      }).parse({
        ...request.query,
        organizationId: (request as any).organization?.id || (request.query as any)?.organizationId,
      });

      if (request.user?.role !== 'maintainer' && params.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      const results = await TimeSeriesService.getPerformanceSummary(params);

      return { data: results };
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
        message: 'Failed to get performance summary',
      });
    }
  });

  // Get user activity summary
  fastify.get('/api/timeseries/activity/summary', async (request: AuthenticatedRequest, reply) => {
    try {
      const params = z.object({
        organizationId: z.string(),
        teamId: z.string().optional(),
        userId: z.string().optional(),
        startDate: z.string().transform(str => new Date(str)),
        endDate: z.string().transform(str => new Date(str)),
      }).parse({
        ...request.query,
        organizationId: (request as any).organization?.id || (request.query as any)?.organizationId,
      });

      if (request.user?.role !== 'maintainer' && params.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      const results = await TimeSeriesService.getUserActivitySummary(params);

      return { data: results };
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
        message: 'Failed to get user activity summary',
      });
    }
  });

  // Get top metrics
  fastify.get('/api/timeseries/metrics/top', async (request: AuthenticatedRequest, reply) => {
    try {
      const params = QueryParamsSchema.parse({
        ...request.query,
        organizationId: (request as any).organization?.id || (request.query as any)?.organizationId,
      });

      if (request.user?.role !== 'maintainer' && params.organizationId !== (request as any).organization?.id) {
        return reply.status(403).send({
          error: 'Forbidden',
          message: 'Access denied to this organization',
        });
      }

      const results = await TimeSeriesService.getTopMetrics({
        organizationId: params.organizationId,
        teamId: params.teamId,
        metricName: (request.query as any)?.metricName,
        startTime: params.startTime,
        endTime: params.endTime,
        limit: (request.query as any)?.limit || 10,
      });

      return { data: results };
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
        message: 'Failed to get top metrics',
      });
    }
  });

  // Add audit logging to all routes
  fastify.addHook('onSend', auditLogResponse);
}

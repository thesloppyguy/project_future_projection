import { FastifyInstance } from 'fastify';
import { requireAuth, AuthenticatedRequest } from '../../middleware/auth.middleware';
import { requireOrganization, TenantRequest } from '../../middleware/tenant.middleware';
import { auditLog, auditLogResponse } from '../../middleware/audit-log.middleware';
import { WebhookService } from '../../services/webhook.service';
import { db } from '../../db';
import { webhooks, webhookEvents } from '../../db/schema';
import { eq, and, desc } from 'drizzle-orm';
import { z } from 'zod';
import crypto from 'crypto';

// Validation schemas
const WebhookCreateSchema = z.object({
  name: z.string().min(1).max(255),
  url: z.string().url(),
  events: z.array(z.string()).min(1),
  secret: z.string().optional(),
  isActive: z.boolean().default(true),
  retryCount: z.number().min(0).max(10).default(3),
  timeout: z.number().min(1).max(300).default(30),
});

const WebhookUpdateSchema = z.object({
  name: z.string().min(1).max(255).optional(),
  url: z.string().url().optional(),
  events: z.array(z.string()).min(1).optional(),
  secret: z.string().optional(),
  isActive: z.boolean().optional(),
  retryCount: z.number().min(0).max(10).optional(),
  timeout: z.number().min(1).max(300).optional(),
});

const WebhookTestSchema = z.object({
  webhookId: z.string(),
  eventType: z.string(),
  eventName: z.string(),
  payload: z.record(z.any()).optional(),
});

export async function webhookRoutes(fastify: FastifyInstance) {
  // Webhook endpoint for receiving external webhooks (no auth required)
  fastify.post('/webhook/:webhookId', async (request, reply) => {
    const { webhookId } = request.params as { webhookId: string };
    const headers = request.headers;
    const body = request.body;

    try {
      // Get webhook configuration
      const webhook = await db
        .select()
        .from(webhooks)
        .where(and(
          eq(webhooks.id, webhookId),
          eq(webhooks.isActive, true)
        ))
        .limit(1);

      if (!webhook.length) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Webhook not found or inactive',
        });
      }

      const webhookConfig = webhook[0];

      // Validate webhook signature if secret is configured
      if (webhookConfig.secret) {
        const signature = headers['x-webhook-signature'] || headers['x-hub-signature-256'];
        if (!signature || !WebhookService.validateSignature(body, webhookConfig.secret, signature as string)) {
          return reply.status(401).send({
            error: 'Unauthorized',
            message: 'Invalid webhook signature',
          });
        }
      }

      // Process the webhook
      await WebhookService.processIncomingWebhook({
        webhookId: webhookConfig.id,
        organizationId: webhookConfig.organizationId,
        headers: headers as Record<string, string>,
        body: body as Record<string, any>,
        ipAddress: request.ip,
        userAgent: headers['user-agent'] as string,
      });

      return { success: true, message: 'Webhook processed successfully' };
    } catch (error) {
      fastify.log.error('Webhook processing error:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to process webhook',
      });
    }
  });

  // Generic webhook endpoint for testing
  fastify.post('/webhook/test/:webhookId', async (request, reply) => {
    const { webhookId } = request.params as { webhookId: string };
    const body = request.body;

    try {
      // Get webhook configuration
      const webhook = await db
        .select()
        .from(webhooks)
        .where(eq(webhooks.id, webhookId))
        .limit(1);

      if (!webhook.length) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Webhook not found',
        });
      }

      const webhookConfig = webhook[0];

      // Log the test webhook
      await WebhookService.logWebhookEvent({
        webhookId: webhookConfig.id,
        event: 'test.webhook',
        payload: body as Record<string, any>,
        status: 'delivered',
        responseCode: '200',
        responseBody: JSON.stringify({ success: true, message: 'Test webhook received' }),
        attempts: 1,
      });

      return { 
        success: true, 
        message: 'Test webhook received',
        webhookId: webhookConfig.id,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      fastify.log.error('Test webhook error:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to process test webhook',
      });
    }
  });

  // All other routes require authentication
  fastify.addHook('preHandler', requireAuth);

  // Create webhook
  fastify.post('/api/webhooks', {
    preHandler: [
      requireOrganization,
      auditLog('webhook.created', 'webhook', (req) => (req.body as any)?.name),
    ],
  }, async (request: TenantRequest, reply) => {
    try {
      const webhookData = WebhookCreateSchema.parse(request.body);

      const [webhook] = await db.insert(webhooks).values({
        organizationId: request.organization!.id,
        name: webhookData.name,
        url: webhookData.url,
        events: webhookData.events,
        secret: webhookData.secret,
        isActive: webhookData.isActive,
        retryCount: webhookData.retryCount.toString(),
        timeout: webhookData.timeout.toString(),
      }).returning();

      return { webhook };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid webhook data',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to create webhook',
      });
    }
  });

  // List webhooks
  fastify.get('/api/webhooks', {
    preHandler: [requireOrganization],
  }, async (request: TenantRequest, reply) => {
    try {
      const webhooksList = await db
        .select()
        .from(webhooks)
        .where(eq(webhooks.organizationId, request.organization!.id))
        .orderBy(desc(webhooks.createdAt));

      return { webhooks: webhooksList };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to fetch webhooks',
      });
    }
  });

  // Get webhook by ID
  fastify.get('/api/webhooks/:webhookId', {
    preHandler: [requireOrganization],
  }, async (request: TenantRequest, reply) => {
    try {
      const { webhookId } = request.params as { webhookId: string };

      const webhook = await db
        .select()
        .from(webhooks)
        .where(and(
          eq(webhooks.id, webhookId),
          eq(webhooks.organizationId, request.organization!.id)
        ))
        .limit(1);

      if (!webhook.length) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Webhook not found',
        });
      }

      return { webhook: webhook[0] };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to fetch webhook',
      });
    }
  });

  // Update webhook
  fastify.put('/api/webhooks/:webhookId', {
    preHandler: [
      requireOrganization,
      auditLog('webhook.updated', 'webhook', (req) => (req.params as any)?.webhookId),
    ],
  }, async (request: TenantRequest, reply) => {
    try {
      const { webhookId } = request.params as { webhookId: string };
      const updateData = WebhookUpdateSchema.parse(request.body);

      const [webhook] = await db
        .update(webhooks)
        .set({
          ...updateData,
          retryCount: updateData.retryCount?.toString(),
          timeout: updateData.timeout?.toString(),
          updatedAt: new Date(),
        })
        .where(and(
          eq(webhooks.id, webhookId),
          eq(webhooks.organizationId, request.organization!.id)
        ))
        .returning();

      if (!webhook) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Webhook not found',
        });
      }

      return { webhook };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid webhook data',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to update webhook',
      });
    }
  });

  // Delete webhook
  fastify.delete('/api/webhooks/:webhookId', {
    preHandler: [
      requireOrganization,
      auditLog('webhook.deleted', 'webhook', (req) => (req.params as any)?.webhookId),
    ],
  }, async (request: TenantRequest, reply) => {
    try {
      const { webhookId } = request.params as { webhookId: string };

      const result = await db
        .delete(webhooks)
        .where(and(
          eq(webhooks.id, webhookId),
          eq(webhooks.organizationId, request.organization!.id)
        ));

      return { success: true, message: 'Webhook deleted successfully' };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to delete webhook',
      });
    }
  });

  // Test webhook
  fastify.post('/api/webhooks/:webhookId/test', {
    preHandler: [
      requireOrganization,
      auditLog('webhook.tested', 'webhook', (req) => (req.params as any)?.webhookId),
    ],
  }, async (request: TenantRequest, reply) => {
    try {
      const { webhookId } = request.params as { webhookId: string };
      const testData = WebhookTestSchema.parse({
        webhookId,
        ...request.body,
      });

      const result = await WebhookService.testWebhook(testData);

      return { 
        success: true, 
        message: 'Webhook test completed',
        result,
      };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid test data',
          details: error.errors,
        });
      }

      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to test webhook',
      });
    }
  });

  // Get webhook events
  fastify.get('/api/webhooks/:webhookId/events', {
    preHandler: [requireOrganization],
  }, async (request: TenantRequest, reply) => {
    try {
      const { webhookId } = request.params as { webhookId: string };
      const { page = 1, limit = 50 } = request.query as { page?: number; limit?: number };

      // Verify webhook belongs to organization
      const webhook = await db
        .select()
        .from(webhooks)
        .where(and(
          eq(webhooks.id, webhookId),
          eq(webhooks.organizationId, request.organization!.id)
        ))
        .limit(1);

      if (!webhook.length) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Webhook not found',
        });
      }

      const events = await db
        .select()
        .from(webhookEvents)
        .where(eq(webhookEvents.webhookId, webhookId))
        .orderBy(desc(webhookEvents.createdAt))
        .limit(limit)
        .offset((page - 1) * limit);

      return { events };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to fetch webhook events',
      });
    }
  });

  // Retry failed webhook
  fastify.post('/api/webhooks/:webhookId/events/:eventId/retry', {
    preHandler: [
      requireOrganization,
      auditLog('webhook.retried', 'webhook', (req) => (req.params as any)?.eventId),
    ],
  }, async (request: TenantRequest, reply) => {
    try {
      const { webhookId, eventId } = request.params as { webhookId: string; eventId: string };

      // Verify webhook belongs to organization
      const webhook = await db
        .select()
        .from(webhooks)
        .where(and(
          eq(webhooks.id, webhookId),
          eq(webhooks.organizationId, request.organization!.id)
        ))
        .limit(1);

      if (!webhook.length) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Webhook not found',
        });
      }

      const result = await WebhookService.retryWebhookEvent(eventId);

      return { 
        success: true, 
        message: 'Webhook retry initiated',
        result,
      };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to retry webhook',
      });
    }
  });

  // Get webhook statistics
  fastify.get('/api/webhooks/:webhookId/stats', {
    preHandler: [requireOrganization],
  }, async (request: TenantRequest, reply) => {
    try {
      const { webhookId } = request.params as { webhookId: string };
      const { startDate, endDate } = request.query as { 
        startDate?: string; 
        endDate?: string; 
      };

      // Verify webhook belongs to organization
      const webhook = await db
        .select()
        .from(webhooks)
        .where(and(
          eq(webhooks.id, webhookId),
          eq(webhooks.organizationId, request.organization!.id)
        ))
        .limit(1);

      if (!webhook.length) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Webhook not found',
        });
      }

      const stats = await WebhookService.getWebhookStats(webhookId, {
        startDate: startDate ? new Date(startDate) : undefined,
        endDate: endDate ? new Date(endDate) : undefined,
      });

      return { stats };
    } catch (error) {
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to fetch webhook statistics',
      });
    }
  });

  // Add audit logging to all routes
  fastify.addHook('onSend', auditLogResponse);
}

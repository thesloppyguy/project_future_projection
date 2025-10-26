import { db } from '../db';
import { webhooks, webhookEvents } from '../db/schema';
import { eq, and, gte, lte, count, sql } from 'drizzle-orm';
import crypto from 'crypto';
import { TimeSeriesService } from './timeseries.service';

export interface WebhookPayload {
  webhookId: string;
  organizationId: string;
  headers: Record<string, string>;
  body: Record<string, any>;
  ipAddress?: string;
  userAgent?: string;
}

export interface WebhookTestData {
  webhookId: string;
  eventType: string;
  eventName: string;
  payload?: Record<string, any>;
}

export interface WebhookStats {
  totalEvents: number;
  successfulDeliveries: number;
  failedDeliveries: number;
  successRate: number;
  averageResponseTime: number;
  lastDelivery: string | null;
  eventsByStatus: Record<string, number>;
  eventsByDay: Array<{ date: string; count: number; success: number; failed: number }>;
}

export class WebhookService {
  /**
   * Validate webhook signature
   */
  static validateSignature(
    payload: any,
    secret: string,
    signature: string
  ): boolean {
    try {
      const body = typeof payload === 'string' ? payload : JSON.stringify(payload);
      const expectedSignature = crypto
        .createHmac('sha256', secret)
        .update(body)
        .digest('hex');

      // Handle different signature formats
      const cleanSignature = signature.replace(/^sha256=/, '');
      
      return crypto.timingSafeEqual(
        Buffer.from(expectedSignature, 'hex'),
        Buffer.from(cleanSignature, 'hex')
      );
    } catch (error) {
      return false;
    }
  }

  /**
   * Process incoming webhook
   */
  static async processIncomingWebhook(payload: WebhookPayload): Promise<void> {
    const { webhookId, organizationId, headers, body, ipAddress, userAgent } = payload;

    try {
      // Log the webhook event
      await this.logWebhookEvent({
        webhookId,
        event: 'webhook.received',
        payload: body,
        status: 'delivered',
        responseCode: '200',
        responseBody: JSON.stringify({ success: true }),
        attempts: 1,
      });

      // Store in time series for analytics
      await TimeSeriesService.insertEvent({
        organizationId,
        eventType: 'webhook',
        eventName: 'webhook.received',
        eventData: {
          webhookId,
          headers: this.sanitizeHeaders(headers),
          bodySize: JSON.stringify(body).length,
          ipAddress,
          userAgent,
        },
      });

      // Process the webhook based on event type
      await this.routeWebhookEvent(webhookId, body);

    } catch (error) {
      console.error('Failed to process incoming webhook:', error);
      
      // Log failed event
      await this.logWebhookEvent({
        webhookId,
        event: 'webhook.received',
        payload: body,
        status: 'failed',
        responseCode: '500',
        responseBody: JSON.stringify({ error: 'Processing failed' }),
        attempts: 1,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  /**
   * Route webhook event to appropriate handlers
   */
  private static async routeWebhookEvent(webhookId: string, payload: any): Promise<void> {
    const eventType = payload.type || payload.event || 'unknown';
    const eventName = payload.name || payload.action || 'unknown';

    // Route to different handlers based on event type
    switch (eventType) {
      case 'user':
        await this.handleUserEvent(webhookId, eventName, payload);
        break;
      case 'order':
      case 'payment':
        await this.handleBusinessEvent(webhookId, eventName, payload);
        break;
      case 'system':
        await this.handleSystemEvent(webhookId, eventName, payload);
        break;
      default:
        await this.handleGenericEvent(webhookId, eventType, eventName, payload);
    }
  }

  /**
   * Handle user-related events
   */
  private static async handleUserEvent(webhookId: string, eventName: string, payload: any): Promise<void> {
    const webhook = await db.select().from(webhooks).where(eq(webhooks.id, webhookId)).limit(1);
    if (!webhook.length) return;

    const organizationId = webhook[0].organizationId;

    // Log user activity
    await TimeSeriesService.insertUserActivity({
      organizationId,
      userId: payload.userId || 'unknown',
      activityType: 'webhook',
      activityName: `user.${eventName}`,
      activityData: {
        webhookId,
        eventData: payload,
      },
    });

    // Store as business metric if it's a significant event
    if (['user.created', 'user.activated', 'user.deleted'].includes(eventName)) {
      await TimeSeriesService.insertBusinessMetric({
        organizationId,
        metricCategory: 'users',
        metricName: eventName,
        metricValue: 1,
        dimensions: {
          source: 'webhook',
          webhookId,
        },
      });
    }
  }

  /**
   * Handle business-related events
   */
  private static async handleBusinessEvent(webhookId: string, eventName: string, payload: any): Promise<void> {
    const webhook = await db.select().from(webhooks).where(eq(webhooks.id, webhookId)).limit(1);
    if (!webhook.length) return;

    const organizationId = webhook[0].organizationId;

    // Store business metrics
    if (eventName === 'order.created' || eventName === 'payment.completed') {
      await TimeSeriesService.insertBusinessMetric({
        organizationId,
        metricCategory: 'sales',
        metricName: eventName,
        metricValue: payload.amount || 1,
        dimensions: {
          source: 'webhook',
          webhookId,
          currency: payload.currency || 'USD',
        },
      });
    }
  }

  /**
   * Handle system events
   */
  private static async handleSystemEvent(webhookId: string, eventName: string, payload: any): Promise<void> {
    const webhook = await db.select().from(webhooks).where(eq(webhooks.id, webhookId)).limit(1);
    if (!webhook.length) return;

    const organizationId = webhook[0].organizationId;

    // Store system metrics
    await TimeSeriesService.insertMetric({
      organizationId,
      metricName: `system.${eventName}`,
      metricValue: 1,
      metricType: 'counter',
      labels: {
        source: 'webhook',
        webhookId,
        eventType: 'system',
      },
    });
  }

  /**
   * Handle generic events
   */
  private static async handleGenericEvent(
    webhookId: string, 
    eventType: string, 
    eventName: string, 
    payload: any
  ): Promise<void> {
    const webhook = await db.select().from(webhooks).where(eq(webhooks.id, webhookId)).limit(1);
    if (!webhook.length) return;

    const organizationId = webhook[0].organizationId;

    // Store as generic event
    await TimeSeriesService.insertEvent({
      organizationId,
      eventType,
      eventName,
      eventData: {
        webhookId,
        originalPayload: payload,
      },
    });
  }

  /**
   * Test webhook delivery
   */
  static async testWebhook(testData: WebhookTestData): Promise<{
    success: boolean;
    responseCode?: number;
    responseBody?: string;
    responseTime?: number;
    error?: string;
  }> {
    const { webhookId, eventType, eventName, payload = {} } = testData;

    try {
      const webhook = await db.select().from(webhooks).where(eq(webhooks.id, webhookId)).limit(1);
      if (!webhook.length) {
        throw new Error('Webhook not found');
      }

      const webhookConfig = webhook[0];
      const startTime = Date.now();

      // Prepare test payload
      const testPayload = {
        type: eventType,
        name: eventName,
        timestamp: new Date().toISOString(),
        test: true,
        ...payload,
      };

      // Make HTTP request to webhook URL
      const response = await fetch(webhookConfig.url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'WebhookService/1.0',
          'X-Webhook-Event': eventName,
          'X-Webhook-Type': eventType,
          ...(webhookConfig.secret && {
            'X-Webhook-Signature': this.generateSignature(testPayload, webhookConfig.secret),
          }),
        },
        body: JSON.stringify(testPayload),
        signal: AbortSignal.timeout(parseInt(webhookConfig.timeout) * 1000),
      });

      const responseTime = Date.now() - startTime;
      const responseBody = await response.text();

      const result = {
        success: response.ok,
        responseCode: response.status,
        responseBody,
        responseTime,
      };

      // Log the test event
      await this.logWebhookEvent({
        webhookId,
        event: `test.${eventName}`,
        payload: testPayload,
        status: response.ok ? 'delivered' : 'failed',
        responseCode: response.status.toString(),
        responseBody,
        attempts: 1,
      });

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      // Log failed test
      await this.logWebhookEvent({
        webhookId,
        event: `test.${eventName}`,
        payload: payload,
        status: 'failed',
        responseCode: '0',
        responseBody: '',
        attempts: 1,
        error: errorMessage,
      });

      return {
        success: false,
        error: errorMessage,
      };
    }
  }

  /**
   * Retry failed webhook event
   */
  static async retryWebhookEvent(eventId: string): Promise<{
    success: boolean;
    message: string;
  }> {
    try {
      const event = await db
        .select()
        .from(webhookEvents)
        .where(eq(webhookEvents.id, eventId))
        .limit(1);

      if (!event.length) {
        throw new Error('Webhook event not found');
      }

      const webhookEvent = event[0];
      const webhook = await db
        .select()
        .from(webhooks)
        .where(eq(webhooks.id, webhookEvent.webhookId))
        .limit(1);

      if (!webhook.length) {
        throw new Error('Webhook not found');
      }

      const webhookConfig = webhook[0];
      const startTime = Date.now();

      // Make retry request
      const response = await fetch(webhookConfig.url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'WebhookService/1.0',
          'X-Webhook-Event': webhookEvent.event,
          'X-Webhook-Retry': 'true',
          ...(webhookConfig.secret && {
            'X-Webhook-Signature': this.generateSignature(webhookEvent.payload, webhookConfig.secret),
          }),
        },
        body: JSON.stringify(webhookEvent.payload),
        signal: AbortSignal.timeout(parseInt(webhookConfig.timeout) * 1000),
      });

      const responseTime = Date.now() - startTime;
      const responseBody = await response.text();

      // Update event status
      await db
        .update(webhookEvents)
        .set({
          status: response.ok ? 'delivered' : 'failed',
          responseCode: response.status.toString(),
          responseBody,
          attempts: (parseInt(webhookEvent.attempts) + 1).toString(),
          deliveredAt: response.ok ? new Date() : null,
        })
        .where(eq(webhookEvents.id, eventId));

      return {
        success: response.ok,
        message: response.ok ? 'Webhook retry successful' : 'Webhook retry failed',
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      // Update event with error
      await db
        .update(webhookEvents)
        .set({
          status: 'failed',
          error: errorMessage,
          attempts: sql`${webhookEvents.attempts} + 1`,
        })
        .where(eq(webhookEvents.id, eventId));

      return {
        success: false,
        message: `Retry failed: ${errorMessage}`,
      };
    }
  }

  /**
   * Get webhook statistics
   */
  static async getWebhookStats(
    webhookId: string,
    options: { startDate?: Date; endDate?: Date } = {}
  ): Promise<WebhookStats> {
    const { startDate, endDate } = options;
    const now = new Date();
    const defaultStartDate = startDate || new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000); // 30 days ago
    const defaultEndDate = endDate || now;

    // Get total events
    const totalEventsResult = await db
      .select({ count: count() })
      .from(webhookEvents)
      .where(and(
        eq(webhookEvents.webhookId, webhookId),
        gte(webhookEvents.createdAt, defaultStartDate),
        lte(webhookEvents.createdAt, defaultEndDate)
      ));

    const totalEvents = totalEventsResult[0]?.count || 0;

    // Get successful deliveries
    const successfulResult = await db
      .select({ count: count() })
      .from(webhookEvents)
      .where(and(
        eq(webhookEvents.webhookId, webhookId),
        eq(webhookEvents.status, 'delivered'),
        gte(webhookEvents.createdAt, defaultStartDate),
        lte(webhookEvents.createdAt, defaultEndDate)
      ));

    const successfulDeliveries = successfulResult[0]?.count || 0;
    const failedDeliveries = totalEvents - successfulDeliveries;
    const successRate = totalEvents > 0 ? (successfulDeliveries / totalEvents) * 100 : 0;

    // Get average response time
    const avgResponseTimeResult = await db
      .select({
        avgTime: sql<number>`avg(
          CASE 
            WHEN ${webhookEvents.deliveredAt} IS NOT NULL 
            THEN extract(epoch from (${webhookEvents.deliveredAt} - ${webhookEvents.createdAt})) * 1000
            ELSE NULL 
          END
        )`,
      })
      .from(webhookEvents)
      .where(and(
        eq(webhookEvents.webhookId, webhookId),
        gte(webhookEvents.createdAt, defaultStartDate),
        lte(webhookEvents.createdAt, defaultEndDate)
      ));

    const averageResponseTime = avgResponseTimeResult[0]?.avgTime || 0;

    // Get last delivery
    const lastDeliveryResult = await db
      .select({ lastDelivery: webhookEvents.deliveredAt })
      .from(webhookEvents)
      .where(and(
        eq(webhookEvents.webhookId, webhookId),
        eq(webhookEvents.status, 'delivered')
      ))
      .orderBy(desc(webhookEvents.deliveredAt))
      .limit(1);

    const lastDelivery = lastDeliveryResult[0]?.lastDelivery?.toISOString() || null;

    // Get events by status
    const eventsByStatusResult = await db
      .select({
        status: webhookEvents.status,
        count: count(),
      })
      .from(webhookEvents)
      .where(and(
        eq(webhookEvents.webhookId, webhookId),
        gte(webhookEvents.createdAt, defaultStartDate),
        lte(webhookEvents.createdAt, defaultEndDate)
      ))
      .groupBy(webhookEvents.status);

    const eventsByStatus = eventsByStatusResult.reduce((acc, row) => {
      acc[row.status] = row.count;
      return acc;
    }, {} as Record<string, number>);

    // Get events by day
    const eventsByDayResult = await db
      .select({
        date: sql<string>`toDate(${webhookEvents.createdAt})`,
        count: count(),
        success: sql<number>`countIf(${webhookEvents.status} = 'delivered')`,
        failed: sql<number>`countIf(${webhookEvents.status} = 'failed')`,
      })
      .from(webhookEvents)
      .where(and(
        eq(webhookEvents.webhookId, webhookId),
        gte(webhookEvents.createdAt, defaultStartDate),
        lte(webhookEvents.createdAt, defaultEndDate)
      ))
      .groupBy(sql`toDate(${webhookEvents.createdAt})`)
      .orderBy(sql`toDate(${webhookEvents.createdAt})`);

    const eventsByDay = eventsByDayResult.map(row => ({
      date: row.date,
      count: row.count,
      success: row.success,
      failed: row.failed,
    }));

    return {
      totalEvents,
      successfulDeliveries,
      failedDeliveries,
      successRate,
      averageResponseTime,
      lastDelivery,
      eventsByStatus,
      eventsByDay,
    };
  }

  /**
   * Log webhook event
   */
  static async logWebhookEvent(data: {
    webhookId: string;
    event: string;
    payload: any;
    status: 'pending' | 'delivered' | 'failed';
    responseCode?: string;
    responseBody?: string;
    attempts?: number;
    error?: string;
  }): Promise<void> {
    await db.insert(webhookEvents).values({
      webhookId: data.webhookId,
      event: data.event,
      payload: data.payload,
      status: data.status,
      responseCode: data.responseCode,
      responseBody: data.responseBody,
      attempts: data.attempts?.toString() || '1',
      deliveredAt: data.status === 'delivered' ? new Date() : null,
      error: data.error,
    });
  }

  /**
   * Generate webhook signature
   */
  private static generateSignature(payload: any, secret: string): string {
    const body = typeof payload === 'string' ? payload : JSON.stringify(payload);
    return crypto
      .createHmac('sha256', secret)
      .update(body)
      .digest('hex');
  }

  /**
   * Sanitize headers for logging
   */
  private static sanitizeHeaders(headers: Record<string, string>): Record<string, string> {
    const sanitized = { ...headers };
    
    // Remove sensitive headers
    delete sanitized.authorization;
    delete sanitized.cookie;
    delete sanitized['x-api-key'];
    delete sanitized['x-auth-token'];
    
    return sanitized;
  }
}

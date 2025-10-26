import { TimeSeriesService } from './timeseries.service';
import { db } from '../db';
import { organizations, users, organizationMembers, teamMembers } from '../db/schema';
import { eq, and, gte, lte } from 'drizzle-orm';

export interface DailyIngestionConfig {
  organizationId: string;
  teamId?: string;
  date: Date;
  metrics?: {
    enabled: boolean;
    categories: string[];
  };
  events?: {
    enabled: boolean;
    types: string[];
  };
  performance?: {
    enabled: boolean;
    endpoints: string[];
  };
  business?: {
    enabled: boolean;
    categories: string[];
  };
  userActivity?: {
    enabled: boolean;
    types: string[];
  };
}

export class DailyIngestionService {
  /**
   * Run daily data ingestion for an organization
   */
  static async runDailyIngestion(config: DailyIngestionConfig): Promise<void> {
    const { organizationId, teamId, date } = config;
    
    console.log(`Starting daily ingestion for org ${organizationId} on ${date.toISOString()}`);

    try {
      // Run ingestion tasks in parallel
      const tasks = [];

      if (config.metrics?.enabled) {
        tasks.push(this.ingestDailyMetrics(organizationId, teamId, date, config.metrics.categories));
      }

      if (config.events?.enabled) {
        tasks.push(this.ingestDailyEvents(organizationId, teamId, date, config.events.types));
      }

      if (config.performance?.enabled) {
        tasks.push(this.ingestDailyPerformance(organizationId, teamId, date, config.performance.endpoints));
      }

      if (config.business?.enabled) {
        tasks.push(this.ingestDailyBusinessMetrics(organizationId, teamId, date, config.business.categories));
      }

      if (config.userActivity?.enabled) {
        tasks.push(this.ingestDailyUserActivity(organizationId, teamId, date, config.userActivity.types));
      }

      await Promise.all(tasks);

      console.log(`Daily ingestion completed for org ${organizationId}`);
    } catch (error) {
      console.error(`Daily ingestion failed for org ${organizationId}:`, error);
      throw error;
    }
  }

  /**
   * Ingest daily metrics from various sources
   */
  private static async ingestDailyMetrics(
    organizationId: string,
    teamId: string | undefined,
    date: Date,
    categories: string[]
  ): Promise<void> {
    const startOfDay = new Date(date);
    startOfDay.setHours(0, 0, 0, 0);
    
    const endOfDay = new Date(date);
    endOfDay.setHours(23, 59, 59, 999);

    const metrics = [];

    // Generate sample metrics for demonstration
    // In a real implementation, this would fetch from actual data sources
    for (const category of categories) {
      switch (category) {
        case 'system':
          metrics.push(
            {
              organizationId,
              teamId,
              metricName: 'cpu_usage_percent',
              metricValue: Math.random() * 100,
              metricType: 'gauge' as const,
              labels: { category, source: 'system' },
            },
            {
              organizationId,
              teamId,
              metricName: 'memory_usage_percent',
              metricValue: Math.random() * 100,
              metricType: 'gauge' as const,
              labels: { category, source: 'system' },
            },
            {
              organizationId,
              teamId,
              metricName: 'disk_usage_percent',
              metricValue: Math.random() * 100,
              metricType: 'gauge' as const,
              labels: { category, source: 'system' },
            }
          );
          break;

        case 'application':
          metrics.push(
            {
              organizationId,
              teamId,
              metricName: 'active_users',
              metricValue: Math.floor(Math.random() * 1000) + 100,
              metricType: 'gauge' as const,
              labels: { category, source: 'application' },
            },
            {
              organizationId,
              teamId,
              metricName: 'api_requests_total',
              metricValue: Math.floor(Math.random() * 10000) + 1000,
              metricType: 'counter' as const,
              labels: { category, source: 'application' },
            },
            {
              organizationId,
              teamId,
              metricName: 'error_rate_percent',
              metricValue: Math.random() * 5,
              metricType: 'gauge' as const,
              labels: { category, source: 'application' },
            }
          );
          break;

        case 'business':
          metrics.push(
            {
              organizationId,
              teamId,
              metricName: 'revenue_daily',
              metricValue: Math.floor(Math.random() * 10000) + 1000,
              metricType: 'counter' as const,
              labels: { category, source: 'business' },
            },
            {
              organizationId,
              teamId,
              metricName: 'conversion_rate_percent',
              metricValue: Math.random() * 20 + 5,
              metricType: 'gauge' as const,
              labels: { category, source: 'business' },
            }
          );
          break;
      }
    }

    if (metrics.length > 0) {
      await TimeSeriesService.insertMetrics(metrics);
      console.log(`Ingested ${metrics.length} metrics for org ${organizationId}`);
    }
  }

  /**
   * Ingest daily events
   */
  private static async ingestDailyEvents(
    organizationId: string,
    teamId: string | undefined,
    date: Date,
    types: string[]
  ): Promise<void> {
    const events = [];

    // Generate sample events
    for (const type of types) {
      switch (type) {
        case 'user':
          events.push(
            {
              organizationId,
              teamId,
              eventType: 'user',
              eventName: 'user_login',
              eventData: { source: 'web', timestamp: date.toISOString() },
            },
            {
              organizationId,
              teamId,
              eventType: 'user',
              eventName: 'user_logout',
              eventData: { source: 'web', timestamp: date.toISOString() },
            }
          );
          break;

        case 'system':
          events.push(
            {
              organizationId,
              teamId,
              eventType: 'system',
              eventName: 'backup_completed',
              eventData: { size: '2.5GB', duration: '15min' },
            },
            {
              organizationId,
              teamId,
              eventType: 'system',
              eventName: 'maintenance_started',
              eventData: { type: 'scheduled', duration: '30min' },
            }
          );
          break;

        case 'business':
          events.push(
            {
              organizationId,
              teamId,
              eventType: 'business',
              eventName: 'order_created',
              eventData: { amount: 99.99, currency: 'USD' },
            },
            {
              organizationId,
              teamId,
              eventType: 'business',
              eventName: 'payment_processed',
              eventData: { amount: 99.99, method: 'credit_card' },
            }
          );
          break;
      }
    }

    for (const event of events) {
      await TimeSeriesService.insertEvent(event);
    }

    console.log(`Ingested ${events.length} events for org ${organizationId}`);
  }

  /**
   * Ingest daily performance metrics
   */
  private static async ingestDailyPerformance(
    organizationId: string,
    teamId: string | undefined,
    date: Date,
    endpoints: string[]
  ): Promise<void> {
    const performanceMetrics = [];

    // Generate sample performance metrics
    for (const endpoint of endpoints) {
      const requestCount = Math.floor(Math.random() * 1000) + 100;
      const errorRate = Math.random() * 0.05; // 0-5% error rate
      const avgResponseTime = Math.random() * 500 + 50; // 50-550ms

      for (let i = 0; i < requestCount; i++) {
        const isError = Math.random() < errorRate;
        const responseTime = avgResponseTime + (Math.random() - 0.5) * 200;

        performanceMetrics.push({
          organizationId,
          teamId,
          endpoint,
          method: 'GET',
          statusCode: isError ? 500 : 200,
          responseTimeMs: Math.max(10, responseTime),
          requestSizeBytes: Math.floor(Math.random() * 1000) + 100,
          responseSizeBytes: Math.floor(Math.random() * 5000) + 500,
        });
      }
    }

    for (const metric of performanceMetrics) {
      await TimeSeriesService.insertPerformanceMetric(metric);
    }

    console.log(`Ingested ${performanceMetrics.length} performance metrics for org ${organizationId}`);
  }

  /**
   * Ingest daily business metrics
   */
  private static async ingestDailyBusinessMetrics(
    organizationId: string,
    teamId: string | undefined,
    date: Date,
    categories: string[]
  ): Promise<void> {
    const businessMetrics = [];

    for (const category of categories) {
      switch (category) {
        case 'sales':
          businessMetrics.push(
            {
              organizationId,
              teamId,
              metricCategory: 'sales',
              metricName: 'daily_revenue',
              metricValue: Math.floor(Math.random() * 50000) + 10000,
              dimensions: { currency: 'USD', region: 'US' },
            },
            {
              organizationId,
              teamId,
              metricCategory: 'sales',
              metricName: 'orders_count',
              metricValue: Math.floor(Math.random() * 500) + 50,
              dimensions: { status: 'completed' },
            }
          );
          break;

        case 'users':
          businessMetrics.push(
            {
              organizationId,
              teamId,
              metricCategory: 'users',
              metricName: 'new_registrations',
              metricValue: Math.floor(Math.random() * 100) + 10,
              dimensions: { source: 'organic' },
            },
            {
              organizationId,
              teamId,
              metricCategory: 'users',
              metricName: 'active_users',
              metricValue: Math.floor(Math.random() * 1000) + 100,
              dimensions: { period: 'daily' },
            }
          );
          break;

        case 'engagement':
          businessMetrics.push(
            {
              organizationId,
              teamId,
              metricCategory: 'engagement',
              metricName: 'page_views',
              metricValue: Math.floor(Math.random() * 10000) + 1000,
              dimensions: { source: 'web' },
            },
            {
              organizationId,
              teamId,
              metricCategory: 'engagement',
              metricName: 'session_duration_avg',
              metricValue: Math.random() * 600 + 60, // 1-11 minutes
              dimensions: { platform: 'web' },
            }
          );
          break;
      }
    }

    for (const metric of businessMetrics) {
      await TimeSeriesService.insertBusinessMetric(metric);
    }

    console.log(`Ingested ${businessMetrics.length} business metrics for org ${organizationId}`);
  }

  /**
   * Ingest daily user activity
   */
  private static async ingestDailyUserActivity(
    organizationId: string,
    teamId: string | undefined,
    date: Date,
    types: string[]
  ): Promise<void> {
    // Get users from the organization
    const orgUsers = await db
      .select({ userId: users.id })
      .from(users)
      .innerJoin(organizationMembers, eq(users.id, organizationMembers.userId))
      .where(
        and(
          eq(organizationMembers.organizationId, organizationId),
          eq(organizationMembers.status, 'active')
        )
      )
      .limit(100); // Limit for demo purposes

    const activities = [];

    for (const user of orgUsers) {
      for (const type of types) {
        switch (type) {
          case 'navigation':
            activities.push(
              {
                organizationId,
                teamId,
                userId: user.userId,
                activityType: 'navigation',
                activityName: 'page_view',
                activityData: { page: '/dashboard', duration: Math.random() * 300 },
              },
              {
                organizationId,
                teamId,
                userId: user.userId,
                activityType: 'navigation',
                activityName: 'page_view',
                activityData: { page: '/profile', duration: Math.random() * 120 },
              }
            );
            break;

          case 'interaction':
            activities.push(
              {
                organizationId,
                teamId,
                userId: user.userId,
                activityType: 'interaction',
                activityName: 'button_click',
                activityData: { button: 'save', page: '/settings' },
              },
              {
                organizationId,
                teamId,
                userId: user.userId,
                activityType: 'interaction',
                activityName: 'form_submit',
                activityData: { form: 'profile_update', success: true },
              }
            );
            break;

          case 'feature':
            activities.push(
              {
                organizationId,
                teamId,
                userId: user.userId,
                activityType: 'feature',
                activityName: 'feature_used',
                activityData: { feature: 'export_data', success: true },
              }
            );
            break;
        }
      }
    }

    for (const activity of activities) {
      await TimeSeriesService.insertUserActivity(activity);
    }

    console.log(`Ingested ${activities.length} user activities for org ${organizationId}`);
  }

  /**
   * Get ingestion status for an organization
   */
  static async getIngestionStatus(organizationId: string, date: Date): Promise<{
    date: string;
    organizationId: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    metrics: number;
    events: number;
    performance: number;
    business: number;
    userActivity: number;
    lastUpdated: string;
  }> {
    // In a real implementation, this would check actual ingestion status
    // For now, return a mock status
    return {
      date: date.toISOString().split('T')[0],
      organizationId,
      status: 'completed',
      metrics: Math.floor(Math.random() * 100) + 50,
      events: Math.floor(Math.random() * 50) + 20,
      performance: Math.floor(Math.random() * 1000) + 100,
      business: Math.floor(Math.random() * 20) + 10,
      userActivity: Math.floor(Math.random() * 200) + 50,
      lastUpdated: new Date().toISOString(),
    };
  }

  /**
   * Schedule daily ingestion for all organizations
   */
  static async scheduleDailyIngestion(date: Date): Promise<void> {
    const organizations = await db
      .select({ id: organizations.id, name: organizations.name })
      .from(organizations)
      .where(eq(organizations.status, 'active'));

    console.log(`Scheduling daily ingestion for ${organizations.length} organizations`);

    for (const org of organizations) {
      const config: DailyIngestionConfig = {
        organizationId: org.id,
        date,
        metrics: {
          enabled: true,
          categories: ['system', 'application', 'business'],
        },
        events: {
          enabled: true,
          types: ['user', 'system', 'business'],
        },
        performance: {
          enabled: true,
          endpoints: ['/api/users', '/api/orders', '/api/dashboard'],
        },
        business: {
          enabled: true,
          categories: ['sales', 'users', 'engagement'],
        },
        userActivity: {
          enabled: true,
          types: ['navigation', 'interaction', 'feature'],
        },
      };

      try {
        await this.runDailyIngestion(config);
      } catch (error) {
        console.error(`Failed to ingest data for organization ${org.id}:`, error);
        // Continue with other organizations
      }
    }
  }
}

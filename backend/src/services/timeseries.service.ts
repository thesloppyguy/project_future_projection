import { clickhouse } from '../config/clickhouse';
import { z } from 'zod';

// Schemas for data validation
export const MetricSchema = z.object({
  organizationId: z.string(),
  teamId: z.string().optional(),
  userId: z.string().optional(),
  metricName: z.string(),
  metricValue: z.number(),
  metricType: z.enum(['gauge', 'counter', 'histogram', 'summary']).default('gauge'),
  labels: z.record(z.string()).default({}),
});

export const EventSchema = z.object({
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

export const LogSchema = z.object({
  organizationId: z.string(),
  teamId: z.string().optional(),
  userId: z.string().optional(),
  level: z.enum(['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL']),
  logger: z.string(),
  message: z.string(),
  context: z.record(z.any()).default({}),
  requestId: z.string().optional(),
});

export const PerformanceMetricSchema = z.object({
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

export const BusinessMetricSchema = z.object({
  organizationId: z.string(),
  teamId: z.string().optional(),
  metricCategory: z.string(),
  metricName: z.string(),
  metricValue: z.number(),
  dimensions: z.record(z.string()).default({}),
});

export const UserActivitySchema = z.object({
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

export type Metric = z.infer<typeof MetricSchema>;
export type Event = z.infer<typeof EventSchema>;
export type Log = z.infer<typeof LogSchema>;
export type PerformanceMetric = z.infer<typeof PerformanceMetricSchema>;
export type BusinessMetric = z.infer<typeof BusinessMetricSchema>;
export type UserActivity = z.infer<typeof UserActivitySchema>;

export class TimeSeriesService {
  /**
   * Insert a single metric
   */
  static async insertMetric(metric: Metric): Promise<void> {
    const validatedMetric = MetricSchema.parse(metric);
    
    await clickhouse.insert({
      table: 'metrics',
      values: [{
        organization_id: validatedMetric.organizationId,
        team_id: validatedMetric.teamId || '',
        user_id: validatedMetric.userId || '',
        metric_name: validatedMetric.metricName,
        metric_value: validatedMetric.metricValue,
        metric_type: validatedMetric.metricType,
        labels: validatedMetric.labels,
      }],
    });
  }

  /**
   * Insert multiple metrics in batch
   */
  static async insertMetrics(metrics: Metric[]): Promise<void> {
    const validatedMetrics = metrics.map(metric => MetricSchema.parse(metric));
    
    const values = validatedMetrics.map(metric => ({
      organization_id: metric.organizationId,
      team_id: metric.teamId || '',
      user_id: metric.userId || '',
      metric_name: metric.metricName,
      metric_value: metric.metricValue,
      metric_type: metric.metricType,
      labels: metric.labels,
    }));

    await clickhouse.insert({
      table: 'metrics',
      values,
    });
  }

  /**
   * Insert an event
   */
  static async insertEvent(event: Event): Promise<void> {
    const validatedEvent = EventSchema.parse(event);
    
    await clickhouse.insert({
      table: 'events',
      values: [{
        organization_id: validatedEvent.organizationId,
        team_id: validatedEvent.teamId || '',
        user_id: validatedEvent.userId || '',
        event_type: validatedEvent.eventType,
        event_name: validatedEvent.eventName,
        event_data: JSON.stringify(validatedEvent.eventData),
        session_id: validatedEvent.sessionId || '',
        ip_address: validatedEvent.ipAddress || '',
        user_agent: validatedEvent.userAgent || '',
      }],
    });
  }

  /**
   * Insert a log entry
   */
  static async insertLog(log: Log): Promise<void> {
    const validatedLog = LogSchema.parse(log);
    
    await clickhouse.insert({
      table: 'logs',
      values: [{
        organization_id: validatedLog.organizationId,
        team_id: validatedLog.teamId || '',
        user_id: validatedLog.userId || '',
        level: validatedLog.level,
        logger: validatedLog.logger,
        message: validatedLog.message,
        context: JSON.stringify(validatedLog.context),
        request_id: validatedLog.requestId || '',
      }],
    });
  }

  /**
   * Insert performance metric
   */
  static async insertPerformanceMetric(metric: PerformanceMetric): Promise<void> {
    const validatedMetric = PerformanceMetricSchema.parse(metric);
    
    await clickhouse.insert({
      table: 'performance_metrics',
      values: [{
        organization_id: validatedMetric.organizationId,
        team_id: validatedMetric.teamId || '',
        user_id: validatedMetric.userId || '',
        endpoint: validatedMetric.endpoint,
        method: validatedMetric.method,
        status_code: validatedMetric.statusCode,
        response_time_ms: validatedMetric.responseTimeMs,
        request_size_bytes: validatedMetric.requestSizeBytes || 0,
        response_size_bytes: validatedMetric.responseSizeBytes || 0,
        user_agent: validatedMetric.userAgent || '',
        ip_address: validatedMetric.ipAddress || '',
      }],
    });
  }

  /**
   * Insert business metric
   */
  static async insertBusinessMetric(metric: BusinessMetric): Promise<void> {
    const validatedMetric = BusinessMetricSchema.parse(metric);
    
    await clickhouse.insert({
      table: 'business_metrics',
      values: [{
        organization_id: validatedMetric.organizationId,
        team_id: validatedMetric.teamId || '',
        metric_category: validatedMetric.metricCategory,
        metric_name: validatedMetric.metricName,
        metric_value: validatedMetric.metricValue,
        dimensions: validatedMetric.dimensions,
      }],
    });
  }

  /**
   * Insert user activity
   */
  static async insertUserActivity(activity: UserActivity): Promise<void> {
    const validatedActivity = UserActivitySchema.parse(activity);
    
    await clickhouse.insert({
      table: 'user_activity',
      values: [{
        organization_id: validatedActivity.organizationId,
        team_id: validatedActivity.teamId || '',
        user_id: validatedActivity.userId,
        activity_type: validatedActivity.activityType,
        activity_name: validatedActivity.activityName,
        activity_data: JSON.stringify(validatedActivity.activityData),
        session_id: validatedActivity.sessionId || '',
        ip_address: validatedActivity.ipAddress || '',
        user_agent: validatedActivity.userAgent || '',
      }],
    });
  }

  /**
   * Query metrics with filters
   */
  static async queryMetrics(params: {
    organizationId: string;
    teamId?: string;
    metricName?: string;
    startTime: Date;
    endTime: Date;
    limit?: number;
  }) {
    const { organizationId, teamId, metricName, startTime, endTime, limit = 1000 } = params;

    let query = `
      SELECT 
        timestamp,
        organization_id,
        team_id,
        user_id,
        metric_name,
        metric_value,
        metric_type,
        labels
      FROM metrics
      WHERE organization_id = {organizationId:String}
        AND timestamp >= {startTime:DateTime64}
        AND timestamp <= {endTime:DateTime64}
    `;

    const queryParams: any = {
      organizationId,
      startTime: startTime.toISOString(),
      endTime: endTime.toISOString(),
    };

    if (teamId) {
      query += ' AND team_id = {teamId:String}';
      queryParams.teamId = teamId;
    }

    if (metricName) {
      query += ' AND metric_name = {metricName:String}';
      queryParams.metricName = metricName;
    }

    query += ' ORDER BY timestamp DESC';

    if (limit) {
      query += ' LIMIT {limit:UInt32}';
      queryParams.limit = limit;
    }

    const result = await clickhouse.query({
      query,
      query_params: queryParams,
      format: 'JSONEachRow',
    });

    return await result.json();
  }

  /**
   * Get daily metrics summary
   */
  static async getDailyMetricsSummary(params: {
    organizationId: string;
    teamId?: string;
    startDate: Date;
    endDate: Date;
  }) {
    const { organizationId, teamId, startDate, endDate } = params;

    let query = `
      SELECT 
        date,
        organization_id,
        team_id,
        metric_name,
        count,
        total_value,
        avg_value,
        min_value,
        max_value
      FROM daily_metrics_summary
      WHERE organization_id = {organizationId:String}
        AND date >= {startDate:Date}
        AND date <= {endDate:Date}
    `;

    const queryParams: any = {
      organizationId,
      startDate: startDate.toISOString().split('T')[0],
      endDate: endDate.toISOString().split('T')[0],
    };

    if (teamId) {
      query += ' AND team_id = {teamId:String}';
      queryParams.teamId = teamId;
    }

    query += ' ORDER BY date DESC, metric_name';

    const result = await clickhouse.query({
      query,
      query_params: queryParams,
      format: 'JSONEachRow',
    });

    return await result.json();
  }

  /**
   * Get performance metrics summary
   */
  static async getPerformanceSummary(params: {
    organizationId: string;
    startTime: Date;
    endTime: Date;
    endpoint?: string;
  }) {
    const { organizationId, startTime, endTime, endpoint } = params;

    let query = `
      SELECT 
        hour,
        organization_id,
        endpoint,
        method,
        request_count,
        total_response_time,
        avg_response_time,
        max_response_time,
        error_count
      FROM hourly_performance_summary
      WHERE organization_id = {organizationId:String}
        AND hour >= {startTime:DateTime}
        AND hour <= {endTime:DateTime}
    `;

    const queryParams: any = {
      organizationId,
      startTime: startTime.toISOString(),
      endTime: endTime.toISOString(),
    };

    if (endpoint) {
      query += ' AND endpoint = {endpoint:String}';
      queryParams.endpoint = endpoint;
    }

    query += ' ORDER BY hour DESC';

    const result = await clickhouse.query({
      query,
      query_params: queryParams,
      format: 'JSONEachRow',
    });

    return await result.json();
  }

  /**
   * Get user activity summary
   */
  static async getUserActivitySummary(params: {
    organizationId: string;
    teamId?: string;
    userId?: string;
    startDate: Date;
    endDate: Date;
  }) {
    const { organizationId, teamId, userId, startDate, endDate } = params;

    let query = `
      SELECT 
        date,
        organization_id,
        team_id,
        user_id,
        activity_count,
        unique_activity_types,
        unique_sessions
      FROM daily_user_activity_summary
      WHERE organization_id = {organizationId:String}
        AND date >= {startDate:Date}
        AND date <= {endDate:Date}
    `;

    const queryParams: any = {
      organizationId,
      startDate: startDate.toISOString().split('T')[0],
      endDate: endDate.toISOString().split('T')[0],
    };

    if (teamId) {
      query += ' AND team_id = {teamId:String}';
      queryParams.teamId = teamId;
    }

    if (userId) {
      query += ' AND user_id = {userId:String}';
      queryParams.userId = userId;
    }

    query += ' ORDER BY date DESC, user_id';

    const result = await clickhouse.query({
      query,
      query_params: queryParams,
      format: 'JSONEachRow',
    });

    return await result.json();
  }

  /**
   * Get top metrics by value
   */
  static async getTopMetrics(params: {
    organizationId: string;
    teamId?: string;
    metricName?: string;
    startTime: Date;
    endTime: Date;
    limit?: number;
  }) {
    const { organizationId, teamId, metricName, startTime, endTime, limit = 10 } = params;

    let query = `
      SELECT 
        metric_name,
        avg(metric_value) as avg_value,
        max(metric_value) as max_value,
        count() as count
      FROM metrics
      WHERE organization_id = {organizationId:String}
        AND timestamp >= {startTime:DateTime64}
        AND timestamp <= {endTime:DateTime64}
    `;

    const queryParams: any = {
      organizationId,
      startTime: startTime.toISOString(),
      endTime: endTime.toISOString(),
    };

    if (teamId) {
      query += ' AND team_id = {teamId:String}';
      queryParams.teamId = teamId;
    }

    if (metricName) {
      query += ' AND metric_name = {metricName:String}';
      queryParams.metricName = metricName;
    }

    query += `
      GROUP BY metric_name
      ORDER BY avg_value DESC
      LIMIT {limit:UInt32}
    `;
    queryParams.limit = limit;

    const result = await clickhouse.query({
      query,
      query_params: queryParams,
      format: 'JSONEachRow',
    });

    return await result.json();
  }
}

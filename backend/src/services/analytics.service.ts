import { clickhouse } from '../config/clickhouse';

export interface AnalyticsQuery {
  organizationId: string;
  teamId?: string;
  startDate: Date;
  endDate: Date;
  granularity: 'hour' | 'day' | 'week' | 'month';
  filters?: Record<string, any>;
}

export interface DashboardMetrics {
  totalUsers: number;
  activeUsers: number;
  totalRevenue: number;
  conversionRate: number;
  avgSessionDuration: number;
  errorRate: number;
  topPages: Array<{ page: string; views: number }>;
  topEvents: Array<{ event: string; count: number }>;
}

export interface TimeSeriesData {
  timestamp: string;
  value: number;
  label?: string;
}

export interface PerformanceMetrics {
  avgResponseTime: number;
  p95ResponseTime: number;
  p99ResponseTime: number;
  totalRequests: number;
  errorRate: number;
  throughput: number;
}

export class AnalyticsService {
  /**
   * Get dashboard metrics for an organization
   */
  static async getDashboardMetrics(query: AnalyticsQuery): Promise<DashboardMetrics> {
    const { organizationId, teamId, startDate, endDate } = query;

    // Get total users
    const totalUsersResult = await clickhouse.query({
      query: `
        SELECT count() as total_users
        FROM user_activity
        WHERE organization_id = {organizationId:String}
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const totalUsers = (await totalUsersResult.json())[0]?.total_users || 0;

    // Get active users (unique users with activity)
    const activeUsersResult = await clickhouse.query({
      query: `
        SELECT uniq(user_id) as active_users
        FROM user_activity
        WHERE organization_id = {organizationId:String}
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const activeUsers = (await activeUsersResult.json())[0]?.active_users || 0;

    // Get total revenue
    const revenueResult = await clickhouse.query({
      query: `
        SELECT sum(metric_value) as total_revenue
        FROM business_metrics
        WHERE organization_id = {organizationId:String}
          AND metric_category = 'sales'
          AND metric_name = 'daily_revenue'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const totalRevenue = (await revenueResult.json())[0]?.total_revenue || 0;

    // Get conversion rate
    const conversionResult = await clickhouse.query({
      query: `
        SELECT avg(metric_value) as conversion_rate
        FROM business_metrics
        WHERE organization_id = {organizationId:String}
          AND metric_category = 'business'
          AND metric_name = 'conversion_rate_percent'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const conversionRate = (await conversionResult.json())[0]?.conversion_rate || 0;

    // Get average session duration
    const sessionDurationResult = await clickhouse.query({
      query: `
        SELECT avg(metric_value) as avg_session_duration
        FROM business_metrics
        WHERE organization_id = {organizationId:String}
          AND metric_category = 'engagement'
          AND metric_name = 'session_duration_avg'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const avgSessionDuration = (await sessionDurationResult.json())[0]?.avg_session_duration || 0;

    // Get error rate
    const errorRateResult = await clickhouse.query({
      query: `
        SELECT avg(metric_value) as error_rate
        FROM metrics
        WHERE organization_id = {organizationId:String}
          AND metric_name = 'error_rate_percent'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const errorRate = (await errorRateResult.json())[0]?.error_rate || 0;

    // Get top pages
    const topPagesResult = await clickhouse.query({
      query: `
        SELECT 
          JSONExtractString(activity_data, 'page') as page,
          count() as views
        FROM user_activity
        WHERE organization_id = {organizationId:String}
          AND activity_type = 'navigation'
          AND activity_name = 'page_view'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        GROUP BY page
        ORDER BY views DESC
        LIMIT 10
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const topPages = await topPagesResult.json();

    // Get top events
    const topEventsResult = await clickhouse.query({
      query: `
        SELECT 
          event_name as event,
          count() as count
        FROM events
        WHERE organization_id = {organizationId:String}
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        GROUP BY event_name
        ORDER BY count DESC
        LIMIT 10
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const topEvents = await topEventsResult.json();

    return {
      totalUsers,
      activeUsers,
      totalRevenue,
      conversionRate,
      avgSessionDuration,
      errorRate,
      topPages,
      topEvents,
    };
  }

  /**
   * Get time series data for charts
   */
  static async getTimeSeriesData(
    query: AnalyticsQuery,
    metricName: string,
    aggregation: 'sum' | 'avg' | 'count' | 'max' | 'min' = 'sum'
  ): Promise<TimeSeriesData[]> {
    const { organizationId, teamId, startDate, endDate, granularity } = query;

    let timeGrouping: string;
    switch (granularity) {
      case 'hour':
        timeGrouping = 'toStartOfHour(timestamp)';
        break;
      case 'day':
        timeGrouping = 'toDate(timestamp)';
        break;
      case 'week':
        timeGrouping = 'toMonday(timestamp)';
        break;
      case 'month':
        timeGrouping = 'toStartOfMonth(timestamp)';
        break;
      default:
        timeGrouping = 'toDate(timestamp)';
    }

    const result = await clickhouse.query({
      query: `
        SELECT 
          ${timeGrouping} as timestamp,
          ${aggregation}(metric_value) as value
        FROM metrics
        WHERE organization_id = {organizationId:String}
          AND metric_name = {metricName:String}
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        ${teamId ? 'AND team_id = {teamId:String}' : ''}
        GROUP BY timestamp
        ORDER BY timestamp
      `,
      query_params: {
        organizationId,
        metricName,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
        ...(teamId && { teamId }),
      },
      format: 'JSONEachRow',
    });

    const data = await result.json();
    return data.map((row: any) => ({
      timestamp: row.timestamp,
      value: parseFloat(row.value) || 0,
    }));
  }

  /**
   * Get performance metrics
   */
  static async getPerformanceMetrics(query: AnalyticsQuery): Promise<PerformanceMetrics> {
    const { organizationId, startDate, endDate } = query;

    const result = await clickhouse.query({
      query: `
        SELECT 
          avg(response_time_ms) as avg_response_time,
          quantile(0.95)(response_time_ms) as p95_response_time,
          quantile(0.99)(response_time_ms) as p99_response_time,
          count() as total_requests,
          countIf(status_code >= 400) / count() as error_rate,
          count() / (dateDiff('second', {startDate:DateTime}, {endDate:DateTime}) / 3600) as throughput
        FROM performance_metrics
        WHERE organization_id = {organizationId:String}
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const data = (await result.json())[0];
    return {
      avgResponseTime: parseFloat(data?.avg_response_time) || 0,
      p95ResponseTime: parseFloat(data?.p95_response_time) || 0,
      p99ResponseTime: parseFloat(data?.p99_response_time) || 0,
      totalRequests: parseInt(data?.total_requests) || 0,
      errorRate: parseFloat(data?.error_rate) || 0,
      throughput: parseFloat(data?.throughput) || 0,
    };
  }

  /**
   * Get user engagement metrics
   */
  static async getUserEngagementMetrics(query: AnalyticsQuery): Promise<{
    dailyActiveUsers: TimeSeriesData[];
    sessionDuration: TimeSeriesData[];
    pageViews: TimeSeriesData[];
    bounceRate: TimeSeriesData[];
  }> {
    const { organizationId, startDate, endDate, granularity } = query;

    let timeGrouping: string;
    switch (granularity) {
      case 'hour':
        timeGrouping = 'toStartOfHour(timestamp)';
        break;
      case 'day':
        timeGrouping = 'toDate(timestamp)';
        break;
      case 'week':
        timeGrouping = 'toMonday(timestamp)';
        break;
      case 'month':
        timeGrouping = 'toStartOfMonth(timestamp)';
        break;
      default:
        timeGrouping = 'toDate(timestamp)';
    }

    // Daily Active Users
    const dauResult = await clickhouse.query({
      query: `
        SELECT 
          ${timeGrouping} as timestamp,
          uniq(user_id) as value
        FROM user_activity
        WHERE organization_id = {organizationId:String}
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        GROUP BY timestamp
        ORDER BY timestamp
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const dailyActiveUsers = (await dauResult.json()).map((row: any) => ({
      timestamp: row.timestamp,
      value: parseInt(row.value) || 0,
    }));

    // Session Duration
    const sessionDurationResult = await clickhouse.query({
      query: `
        SELECT 
          ${timeGrouping} as timestamp,
          avg(JSONExtractFloat(activity_data, 'duration')) as value
        FROM user_activity
        WHERE organization_id = {organizationId:String}
          AND activity_type = 'navigation'
          AND activity_name = 'page_view'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        GROUP BY timestamp
        ORDER BY timestamp
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const sessionDuration = (await sessionDurationResult.json()).map((row: any) => ({
      timestamp: row.timestamp,
      value: parseFloat(row.value) || 0,
    }));

    // Page Views
    const pageViewsResult = await clickhouse.query({
      query: `
        SELECT 
          ${timeGrouping} as timestamp,
          count() as value
        FROM user_activity
        WHERE organization_id = {organizationId:String}
          AND activity_type = 'navigation'
          AND activity_name = 'page_view'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        GROUP BY timestamp
        ORDER BY timestamp
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const pageViews = (await pageViewsResult.json()).map((row: any) => ({
      timestamp: row.timestamp,
      value: parseInt(row.value) || 0,
    }));

    // Bounce Rate (simplified calculation)
    const bounceRateResult = await clickhouse.query({
      query: `
        SELECT 
          ${timeGrouping} as timestamp,
          countIf(JSONExtractFloat(activity_data, 'duration') < 30) / count() as value
        FROM user_activity
        WHERE organization_id = {organizationId:String}
          AND activity_type = 'navigation'
          AND activity_name = 'page_view'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        GROUP BY timestamp
        ORDER BY timestamp
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const bounceRate = (await bounceRateResult.json()).map((row: any) => ({
      timestamp: row.timestamp,
      value: parseFloat(row.value) || 0,
    }));

    return {
      dailyActiveUsers,
      sessionDuration,
      pageViews,
      bounceRate,
    };
  }

  /**
   * Get business metrics
   */
  static async getBusinessMetrics(query: AnalyticsQuery): Promise<{
    revenue: TimeSeriesData[];
    orders: TimeSeriesData[];
    conversionRate: TimeSeriesData[];
    customerAcquisition: TimeSeriesData[];
  }> {
    const { organizationId, startDate, endDate, granularity } = query;

    let timeGrouping: string;
    switch (granularity) {
      case 'hour':
        timeGrouping = 'toStartOfHour(timestamp)';
        break;
      case 'day':
        timeGrouping = 'toDate(timestamp)';
        break;
      case 'week':
        timeGrouping = 'toMonday(timestamp)';
        break;
      case 'month':
        timeGrouping = 'toStartOfMonth(timestamp)';
        break;
      default:
        timeGrouping = 'toDate(timestamp)';
    }

    // Revenue
    const revenueResult = await clickhouse.query({
      query: `
        SELECT 
          ${timeGrouping} as timestamp,
          sum(metric_value) as value
        FROM business_metrics
        WHERE organization_id = {organizationId:String}
          AND metric_category = 'sales'
          AND metric_name = 'daily_revenue'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        GROUP BY timestamp
        ORDER BY timestamp
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const revenue = (await revenueResult.json()).map((row: any) => ({
      timestamp: row.timestamp,
      value: parseFloat(row.value) || 0,
    }));

    // Orders
    const ordersResult = await clickhouse.query({
      query: `
        SELECT 
          ${timeGrouping} as timestamp,
          sum(metric_value) as value
        FROM business_metrics
        WHERE organization_id = {organizationId:String}
          AND metric_category = 'sales'
          AND metric_name = 'orders_count'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        GROUP BY timestamp
        ORDER BY timestamp
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const orders = (await ordersResult.json()).map((row: any) => ({
      timestamp: row.timestamp,
      value: parseFloat(row.value) || 0,
    }));

    // Conversion Rate
    const conversionRateResult = await clickhouse.query({
      query: `
        SELECT 
          ${timeGrouping} as timestamp,
          avg(metric_value) as value
        FROM business_metrics
        WHERE organization_id = {organizationId:String}
          AND metric_category = 'business'
          AND metric_name = 'conversion_rate_percent'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        GROUP BY timestamp
        ORDER BY timestamp
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const conversionRate = (await conversionRateResult.json()).map((row: any) => ({
      timestamp: row.timestamp,
      value: parseFloat(row.value) || 0,
    }));

    // Customer Acquisition
    const customerAcquisitionResult = await clickhouse.query({
      query: `
        SELECT 
          ${timeGrouping} as timestamp,
          sum(metric_value) as value
        FROM business_metrics
        WHERE organization_id = {organizationId:String}
          AND metric_category = 'users'
          AND metric_name = 'new_registrations'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        GROUP BY timestamp
        ORDER BY timestamp
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const customerAcquisition = (await customerAcquisitionResult.json()).map((row: any) => ({
      timestamp: row.timestamp,
      value: parseFloat(row.value) || 0,
    }));

    return {
      revenue,
      orders,
      conversionRate,
      customerAcquisition,
    };
  }

  /**
   * Get system health metrics
   */
  static async getSystemHealthMetrics(query: AnalyticsQuery): Promise<{
    cpuUsage: TimeSeriesData[];
    memoryUsage: TimeSeriesData[];
    diskUsage: TimeSeriesData[];
    errorRate: TimeSeriesData[];
  }> {
    const { organizationId, startDate, endDate, granularity } = query;

    let timeGrouping: string;
    switch (granularity) {
      case 'hour':
        timeGrouping = 'toStartOfHour(timestamp)';
        break;
      case 'day':
        timeGrouping = 'toDate(timestamp)';
        break;
      case 'week':
        timeGrouping = 'toMonday(timestamp)';
        break;
      case 'month':
        timeGrouping = 'toStartOfMonth(timestamp)';
        break;
      default:
        timeGrouping = 'toDate(timestamp)';
    }

    // CPU Usage
    const cpuResult = await clickhouse.query({
      query: `
        SELECT 
          ${timeGrouping} as timestamp,
          avg(metric_value) as value
        FROM metrics
        WHERE organization_id = {organizationId:String}
          AND metric_name = 'cpu_usage_percent'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        GROUP BY timestamp
        ORDER BY timestamp
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const cpuUsage = (await cpuResult.json()).map((row: any) => ({
      timestamp: row.timestamp,
      value: parseFloat(row.value) || 0,
    }));

    // Memory Usage
    const memoryResult = await clickhouse.query({
      query: `
        SELECT 
          ${timeGrouping} as timestamp,
          avg(metric_value) as value
        FROM metrics
        WHERE organization_id = {organizationId:String}
          AND metric_name = 'memory_usage_percent'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        GROUP BY timestamp
        ORDER BY timestamp
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const memoryUsage = (await memoryResult.json()).map((row: any) => ({
      timestamp: row.timestamp,
      value: parseFloat(row.value) || 0,
    }));

    // Disk Usage
    const diskResult = await clickhouse.query({
      query: `
        SELECT 
          ${timeGrouping} as timestamp,
          avg(metric_value) as value
        FROM metrics
        WHERE organization_id = {organizationId:String}
          AND metric_name = 'disk_usage_percent'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        GROUP BY timestamp
        ORDER BY timestamp
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const diskUsage = (await diskResult.json()).map((row: any) => ({
      timestamp: row.timestamp,
      value: parseFloat(row.value) || 0,
    }));

    // Error Rate
    const errorRateResult = await clickhouse.query({
      query: `
        SELECT 
          ${timeGrouping} as timestamp,
          avg(metric_value) as value
        FROM metrics
        WHERE organization_id = {organizationId:String}
          AND metric_name = 'error_rate_percent'
          AND timestamp >= {startDate:DateTime64}
          AND timestamp <= {endDate:DateTime64}
        GROUP BY timestamp
        ORDER BY timestamp
      `,
      query_params: {
        organizationId,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
      },
      format: 'JSONEachRow',
    });

    const errorRate = (await errorRateResult.json()).map((row: any) => ({
      timestamp: row.timestamp,
      value: parseFloat(row.value) || 0,
    }));

    return {
      cpuUsage,
      memoryUsage,
      diskUsage,
      errorRate,
    };
  }
}

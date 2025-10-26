import Fastify from 'fastify';
import { config } from './config/env';
import { db } from './db';
import { authRoutes } from './modules/auth/routes';
import { maintainerRoutes } from './modules/maintainer/routes';
import { webhookRoutes } from './modules/webhooks/routes';
import { internalRoutes } from './modules/internal/routes';
import { mlRoutes } from './modules/ml/routes';
import { WorkerClientService } from './services/worker-client.service';
import { timeseriesRoutes } from './modules/timeseries/routes';
import { analyticsRoutes } from './modules/analytics/routes';
import { seed } from './db/seed';
import { ClickHouseMigrationService } from './services/clickhouse-migration.service';

const fastify = Fastify({
  logger: {
    level: config.LOG_LEVEL,
  },
});

// Register plugins
await fastify.register(import('@fastify/cors'), {
  origin: config.CORS_ORIGIN.split(','),
  credentials: true,
});

await fastify.register(import('@fastify/helmet'), {
  contentSecurityPolicy: false,
});

await fastify.register(import('@fastify/rate-limit'), {
  max: 100,
  timeWindow: '1 minute',
});

await fastify.register(import('@fastify/multipart'), {
  limits: {
    fileSize: config.MAX_FILE_SIZE,
  },
});

await fastify.register(import('@fastify/static'), {
  root: config.UPLOAD_DIR,
  prefix: '/uploads/',
});

// Add database to fastify instance
fastify.decorate('db', db);

  // Add Redis to fastify instance
  fastify.decorate('redis', (await import('./config/redis')).default);

  // Add ClickHouse to fastify instance
  fastify.decorate('clickhouse', (await import('./config/clickhouse')).default);

  // Add Worker Client Service to fastify instance
  const workerClient = new WorkerClientService(fastify);
  fastify.decorate('workerClient', workerClient);

  // Register routes
  await fastify.register(authRoutes);
  await fastify.register(maintainerRoutes);
  await fastify.register(webhookRoutes);
  await fastify.register(internalRoutes);
  await fastify.register(mlRoutes);
  await fastify.register(timeseriesRoutes);
  await fastify.register(analyticsRoutes);

// Health check
fastify.get('/api/v1/health', async (request, reply) => {
  return { status: 'ok', timestamp: new Date().toISOString() };
});

// Error handler
fastify.setErrorHandler((error, request, reply) => {
  fastify.log.error(error);
  
  reply.status(500).send({
    error: 'Internal Server Error',
    message: config.NODE_ENV === 'development' ? error.message : 'Something went wrong',
  });
});

// Start server
const start = async () => {
  try {
    // Run database migrations
    console.log('🔄 Running database migrations...');
    // Note: In production, run migrations separately
    
    // Run ClickHouse migrations
    console.log('🔄 Running ClickHouse migrations...');
    await ClickHouseMigrationService.runMigrations();
    
    // Seed database if needed
    if (config.NODE_ENV === 'development') {
      console.log('🌱 Seeding database...');
      await seed();
    }

    await fastify.listen({
      port: config.PORT,
      host: config.HOST,
    });

    console.log(`🚀 Server running on http://${config.HOST}:${config.PORT}`);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

start();

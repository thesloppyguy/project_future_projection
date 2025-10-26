import { FastifyInstance } from 'fastify';
import { requireAuth, AuthenticatedRequest } from '../../middleware/auth.middleware';
import { requireOrganization, TenantRequest } from '../../middleware/tenant.middleware';
import { auditLog, auditLogResponse } from '../../middleware/audit-log.middleware';
import { WorkerClientService } from '../../services/worker-client.service';
import { db } from '../../db';
import { dataTrainingLogs } from '../../db/schema';
import { eq, and, desc } from 'drizzle-orm';
import { z } from 'zod';
import { nanoid } from 'nanoid';

// Validation schemas
const TrainingRequestSchema = z.object({
  modelType: z.enum(['classification', 'regression', 'clustering', 'neural_network']),
  trainingData: z.object({
    dataSource: z.string(),
    features: z.array(z.string()),
    target: z.string(),
    testSize: z.number().min(0).max(1).default(0.2),
  }),
  hyperparameters: z.record(z.any()).optional(),
  name: z.string().min(1).max(255).optional(),
  description: z.string().optional(),
});

const PredictionRequestSchema = z.object({
  modelId: z.string(),
  inputData: z.record(z.any()),
  batchSize: z.number().min(1).max(1000).default(1).optional(),
});

const ModelListQuerySchema = z.object({
  page: z.coerce.number().min(1).default(1),
  limit: z.coerce.number().min(1).max(100).default(20),
  status: z.enum(['pending', 'running', 'completed', 'failed']).optional(),
});

export async function mlRoutes(fastify: FastifyInstance) {
  // Get worker client service
  const workerClient = fastify.workerClient as WorkerClientService;

  // Start model training
  fastify.post('/api/ml/training/start', {
    preHandler: [
      requireAuth,
      requireOrganization,
      auditLog('ml.training.started', 'training', (req) => (req.body as any)?.name || 'Unnamed Training'),
    ],
  }, async (request: TenantRequest, reply) => {
    try {
      const trainingData = TrainingRequestSchema.parse(request.body);

      // Get available training workers
      const workers = workerClient.getWorkersByCapability('training');
      if (workers.length === 0) {
        return reply.status(503).send({
          error: 'Service Unavailable',
          message: 'No training workers available',
        });
      }

      // Select worker (round-robin or based on load)
      const selectedWorker = workers[0]; // Simple selection for now

      // Create training job
      const jobId = nanoid();
      
      // Log training job start
      await db.insert(dataTrainingLogs).values({
        organizationId: request.organization!.id,
        jobId,
        jobType: 'model_training',
        status: 'pending',
        progress: 0,
        metadata: {
          modelType: trainingData.modelType,
          trainingData: trainingData.trainingData,
          hyperparameters: trainingData.hyperparameters,
          name: trainingData.name,
          description: trainingData.description,
          workerId: selectedWorker.workerId,
        },
      });

      // Submit job to worker
      const jobResponse = await workerClient.submitTrainingJob(selectedWorker.workerId, {
        organizationId: request.organization!.id,
        modelType: trainingData.modelType,
        trainingData: trainingData.trainingData,
        hyperparameters: trainingData.hyperparameters,
        callbackUrl: `${fastify.config.BACKEND_URL}/api/internal/workers/job-update`,
      });

      return {
        success: true,
        jobId,
        status: jobResponse.status,
        workerId: selectedWorker.workerId,
        message: 'Training job started successfully',
      };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid training request data',
          details: error.errors,
        });
      }

      fastify.log.error('Failed to start training:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to start training job',
      });
    }
  });

  // Get training job status
  fastify.get('/api/ml/training/:jobId/status', {
    preHandler: [requireAuth, requireOrganization],
  }, async (request: TenantRequest, reply) => {
    try {
      const { jobId } = request.params as { jobId: string };

      // Get training log from database
      const trainingLog = await db.query.dataTrainingLogs.findFirst({
        where: and(
          eq(dataTrainingLogs.jobId, jobId),
          eq(dataTrainingLogs.organizationId, request.organization!.id)
        ),
      });

      if (!trainingLog) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Training job not found',
        });
      }

      // If job is still running, get status from worker
      if (trainingLog.status === 'running' || trainingLog.status === 'pending') {
        const workerId = trainingLog.metadata?.workerId;
        if (workerId) {
          try {
            const workerStatus = await workerClient.getJobStatus(workerId, jobId);
            return {
              jobId,
              status: workerStatus.status,
              progress: workerStatus.progress,
              result: workerStatus.result,
              error: workerStatus.error,
              startedAt: trainingLog.createdAt,
              updatedAt: trainingLog.updatedAt,
            };
          } catch (error) {
            fastify.log.warn('Failed to get worker status, using cached status', { jobId, error });
          }
        }
      }

      return {
        jobId,
        status: trainingLog.status,
        progress: trainingLog.progress,
        result: trainingLog.result,
        error: trainingLog.error,
        startedAt: trainingLog.createdAt,
        updatedAt: trainingLog.updatedAt,
      };
    } catch (error) {
      fastify.log.error('Failed to get training status:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get training status',
      });
    }
  });

  // List training jobs
  fastify.get('/api/ml/training', {
    preHandler: [requireAuth, requireOrganization],
  }, async (request: TenantRequest, reply) => {
    try {
      const query = ModelListQuerySchema.parse(request.query);
      const { page, limit, status } = query;

      let whereCondition = eq(dataTrainingLogs.organizationId, request.organization!.id);
      if (status) {
        whereCondition = and(whereCondition, eq(dataTrainingLogs.status, status));
      }

      const trainingJobs = await db
        .select()
        .from(dataTrainingLogs)
        .where(whereCondition)
        .orderBy(desc(dataTrainingLogs.createdAt))
        .limit(limit)
        .offset((page - 1) * limit);

      return {
        trainingJobs: trainingJobs.map(job => ({
          jobId: job.jobId,
          status: job.status,
          progress: job.progress,
          result: job.result,
          error: job.error,
          createdAt: job.createdAt,
          updatedAt: job.updatedAt,
          metadata: job.metadata,
        })),
        pagination: {
          page,
          limit,
          total: trainingJobs.length,
        },
      };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid query parameters',
          details: error.errors,
        });
      }

      fastify.log.error('Failed to list training jobs:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to list training jobs',
      });
    }
  });

  // Cancel training job
  fastify.post('/api/ml/training/:jobId/cancel', {
    preHandler: [
      requireAuth,
      requireOrganization,
      auditLog('ml.training.cancelled', 'training', (req) => (req.params as any)?.jobId),
    ],
  }, async (request: TenantRequest, reply) => {
    try {
      const { jobId } = request.params as { jobId: string };

      // Get training log
      const trainingLog = await db.query.dataTrainingLogs.findFirst({
        where: and(
          eq(dataTrainingLogs.jobId, jobId),
          eq(dataTrainingLogs.organizationId, request.organization!.id)
        ),
      });

      if (!trainingLog) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Training job not found',
        });
      }

      if (trainingLog.status === 'completed' || trainingLog.status === 'failed') {
        return reply.status(400).send({
          error: 'Bad Request',
          message: 'Job cannot be cancelled',
        });
      }

      // Cancel job on worker
      const workerId = trainingLog.metadata?.workerId;
      if (workerId) {
        try {
          await workerClient.cancelJob(workerId, jobId);
        } catch (error) {
          fastify.log.warn('Failed to cancel job on worker', { jobId, workerId, error });
        }
      }

      // Update database
      await db
        .update(dataTrainingLogs)
        .set({
          status: 'cancelled',
          updatedAt: new Date(),
        })
        .where(eq(dataTrainingLogs.jobId, jobId));

      return {
        success: true,
        message: 'Training job cancelled successfully',
      };
    } catch (error) {
      fastify.log.error('Failed to cancel training job:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to cancel training job',
      });
    }
  });

  // Start prediction
  fastify.post('/api/ml/prediction/start', {
    preHandler: [
      requireAuth,
      requireOrganization,
      auditLog('ml.prediction.started', 'prediction', (req) => (req.body as any)?.modelId),
    ],
  }, async (request: TenantRequest, reply) => {
    try {
      const predictionData = PredictionRequestSchema.parse(request.body);

      // Get available prediction workers
      const workers = workerClient.getWorkersByCapability('prediction');
      if (workers.length === 0) {
        return reply.status(503).send({
          error: 'Service Unavailable',
          message: 'No prediction workers available',
        });
      }

      // Select worker
      const selectedWorker = workers[0];

      // Create prediction job
      const jobId = nanoid();

      // Submit job to worker
      const jobResponse = await workerClient.submitPredictionJob(selectedWorker.workerId, {
        organizationId: request.organization!.id,
        modelId: predictionData.modelId,
        inputData: predictionData.inputData,
        callbackUrl: `${fastify.config.BACKEND_URL}/api/internal/workers/job-update`,
      });

      return {
        success: true,
        jobId,
        status: jobResponse.status,
        workerId: selectedWorker.workerId,
        message: 'Prediction job started successfully',
      };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid prediction request data',
          details: error.errors,
        });
      }

      fastify.log.error('Failed to start prediction:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to start prediction job',
      });
    }
  });

  // Get prediction job status
  fastify.get('/api/ml/prediction/:jobId/status', {
    preHandler: [requireAuth, requireOrganization],
  }, async (request: TenantRequest, reply) => {
    try {
      const { jobId } = request.params as { jobId: string };

      // For now, we'll get status directly from worker
      // In a real implementation, you'd store prediction jobs in the database too
      const workers = workerClient.getAllWorkers();
      for (const worker of workers) {
        try {
          const status = await workerClient.getJobStatus(worker.workerId, jobId);
          return {
            jobId,
            status: status.status,
            progress: status.progress,
            result: status.result,
            error: status.error,
            startedAt: status.startedAt,
            completedAt: status.completedAt,
          };
        } catch (error) {
          // Continue to next worker
          continue;
        }
      }

      return reply.status(404).send({
        error: 'Not Found',
        message: 'Prediction job not found',
      });
    } catch (error) {
      fastify.log.error('Failed to get prediction status:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get prediction status',
      });
    }
  });

  // Get available models
  fastify.get('/api/ml/models', {
    preHandler: [requireAuth, requireOrganization],
  }, async (request: TenantRequest, reply) => {
    try {
      // Get completed training jobs that produced models
      const models = await db
        .select()
        .from(dataTrainingLogs)
        .where(and(
          eq(dataTrainingLogs.organizationId, request.organization!.id),
          eq(dataTrainingLogs.status, 'completed'),
          eq(dataTrainingLogs.jobType, 'model_training')
        ))
        .orderBy(desc(dataTrainingLogs.createdAt));

      return {
        models: models.map(job => ({
          modelId: job.result?.model_id || job.jobId,
          name: job.metadata?.name || 'Unnamed Model',
          description: job.metadata?.description,
          modelType: job.metadata?.modelType,
          accuracy: job.result?.accuracy,
          createdAt: job.createdAt,
          metadata: job.metadata,
        })),
      };
    } catch (error) {
      fastify.log.error('Failed to get models:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get models',
      });
    }
  });

  // Get worker status
  fastify.get('/api/ml/workers', {
    preHandler: [requireAuth, requireOrganization],
  }, async (request: TenantRequest, reply) => {
    try {
      const workers = workerClient.getAllWorkers();
      
      // Get health status for each worker
      const workersWithHealth = await Promise.all(
        workers.map(async (worker) => {
          try {
            const health = await workerClient.checkWorkerHealth(worker.workerId);
            return {
              ...worker,
              health,
            };
          } catch (error) {
            return {
              ...worker,
              health: { status: 'unhealthy', error: error instanceof Error ? error.message : 'Unknown error' },
            };
          }
        })
      );

      return { workers: workersWithHealth };
    } catch (error) {
      fastify.log.error('Failed to get worker status:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get worker status',
      });
    }
  });

  // Add audit logging to all routes
  fastify.addHook('onSend', auditLogResponse);
}

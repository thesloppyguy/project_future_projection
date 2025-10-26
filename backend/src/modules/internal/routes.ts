import { FastifyInstance } from 'fastify';
import { requireAuth, AuthenticatedRequest } from '../../middleware/auth.middleware';
import { auditLog, auditLogResponse } from '../../middleware/audit-log.middleware';
import { db } from '../../db';
import { dataTrainingLogs } from '../../db/schema';
import { eq, and, desc } from 'drizzle-orm';
import { z } from 'zod';
import { nanoid } from 'nanoid';

// Validation schemas
const WorkerRegistrationSchema = z.object({
  workerId: z.string(),
  workerType: z.string(),
  capabilities: z.array(z.string()),
  maxConcurrentJobs: z.number(),
  status: z.enum(['active', 'inactive', 'maintenance']),
});

const JobUpdateSchema = z.object({
  jobId: z.string(),
  status: z.enum(['pending', 'running', 'completed', 'failed', 'cancelled', 'expired']),
  result: z.record(z.any()).optional(),
  error: z.string().optional(),
  workerId: z.string(),
  timestamp: z.number(),
});

const WorkerHealthSchema = z.object({
  workerId: z.string(),
  status: z.string(),
  activeJobs: z.object({
    training: z.number(),
    prediction: z.number(),
  }),
  memoryUsage: z.number(),
  cpuUsage: z.number(),
  timestamp: z.number(),
});

const InternalMessageSchema = z.object({
  messageType: z.string(),
  payload: z.record(z.any()),
  timestamp: z.string(),
  workerId: z.string(),
});

// In-memory worker registry (in production, use Redis or database)
const workerRegistry = new Map<string, {
  workerId: string;
  workerType: string;
  capabilities: string[];
  maxConcurrentJobs: number;
  status: string;
  lastSeen: Date;
  health: any;
}>();

// Job tracking
const jobRegistry = new Map<string, {
  jobId: string;
  workerId: string;
  jobType: 'training' | 'prediction';
  organizationId: string;
  status: string;
  createdAt: Date;
  updatedAt: Date;
  result?: any;
  error?: string;
}>();

export async function internalRoutes(fastify: FastifyInstance) {
  // Worker registration endpoint
  fastify.post('/api/internal/workers/register', async (request, reply) => {
    try {
      const workerData = WorkerRegistrationSchema.parse(request.body);

      workerRegistry.set(workerData.workerId, {
        ...workerData,
        lastSeen: new Date(),
        health: null,
      });

      fastify.log.info('Worker registered', { workerId: workerData.workerId });

      return {
        success: true,
        message: 'Worker registered successfully',
        workerId: workerData.workerId,
      };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid worker registration data',
          details: error.errors,
        });
      }

      fastify.log.error('Worker registration failed:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to register worker',
      });
    }
  });

  // Worker health check endpoint
  fastify.post('/api/internal/workers/health', async (request, reply) => {
    try {
      const healthData = WorkerHealthSchema.parse(request.body);

      const worker = workerRegistry.get(healthData.workerId);
      if (!worker) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Worker not found',
        });
      }

      // Update worker health
      worker.lastSeen = new Date();
      worker.health = healthData;

      fastify.log.debug('Worker health updated', { 
        workerId: healthData.workerId,
        status: healthData.status,
        activeJobs: healthData.activeJobs,
      });

      return {
        success: true,
        message: 'Health check received',
      };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid health data',
          details: error.errors,
        });
      }

      fastify.log.error('Worker health check failed:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to process health check',
      });
    }
  });

  // Job update endpoint
  fastify.post('/api/internal/workers/job-update', async (request, reply) => {
    try {
      const jobUpdate = JobUpdateSchema.parse(request.body);

      // Update job registry
      const existingJob = jobRegistry.get(jobUpdate.jobId);
      if (existingJob) {
        existingJob.status = jobUpdate.status;
        existingJob.updatedAt = new Date();
        existingJob.result = jobUpdate.result;
        existingJob.error = jobUpdate.error;
      }

      // Log training job updates to database
      if (existingJob?.jobType === 'training') {
        await db.insert(dataTrainingLogs).values({
          organizationId: existingJob.organizationId,
          jobId: jobUpdate.jobId,
          jobType: 'model_training',
          status: jobUpdate.status,
          progress: jobUpdate.status === 'completed' ? 100 : 0,
          result: jobUpdate.result,
          error: jobUpdate.error,
          metadata: {
            workerId: jobUpdate.workerId,
            timestamp: jobUpdate.timestamp,
          },
        }).onConflictDoUpdate({
          target: dataTrainingLogs.jobId,
          set: {
            status: jobUpdate.status,
            progress: jobUpdate.status === 'completed' ? 100 : 0,
            result: jobUpdate.result,
            error: jobUpdate.error,
            updatedAt: new Date(),
          },
        });
      }

      fastify.log.info('Job update received', {
        jobId: jobUpdate.jobId,
        status: jobUpdate.status,
        workerId: jobUpdate.workerId,
      });

      return {
        success: true,
        message: 'Job update processed',
      };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid job update data',
          details: error.errors,
        });
      }

      fastify.log.error('Job update failed:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to process job update',
      });
    }
  });

  // Send message to worker
  fastify.post('/api/internal/workers/:workerId/message', {
    preHandler: [requireAuth],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const { workerId } = request.params as { workerId: string };
      const messageData = InternalMessageSchema.parse(request.body);

      const worker = workerRegistry.get(workerId);
      if (!worker) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Worker not found',
        });
      }

      // In a real implementation, you would send this message to the worker
      // For now, we'll just log it
      fastify.log.info('Message sent to worker', {
        workerId,
        messageType: messageData.messageType,
        from: request.user?.id,
      });

      return {
        success: true,
        message: 'Message sent to worker',
      };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return reply.status(400).send({
          error: 'Validation Error',
          message: 'Invalid message data',
          details: error.errors,
        });
      }

      fastify.log.error('Failed to send message to worker:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to send message',
      });
    }
  });

  // List registered workers
  fastify.get('/api/internal/workers', {
    preHandler: [requireAuth],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const workers = Array.from(workerRegistry.values()).map(worker => ({
        workerId: worker.workerId,
        workerType: worker.workerType,
        capabilities: worker.capabilities,
        maxConcurrentJobs: worker.maxConcurrentJobs,
        status: worker.status,
        lastSeen: worker.lastSeen,
        health: worker.health,
      }));

      return { workers };
    } catch (error) {
      fastify.log.error('Failed to list workers:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to list workers',
      });
    }
  });

  // Get worker details
  fastify.get('/api/internal/workers/:workerId', {
    preHandler: [requireAuth],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const { workerId } = request.params as { workerId: string };

      const worker = workerRegistry.get(workerId);
      if (!worker) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Worker not found',
        });
      }

      return {
        worker: {
          workerId: worker.workerId,
          workerType: worker.workerType,
          capabilities: worker.capabilities,
          maxConcurrentJobs: worker.maxConcurrentJobs,
          status: worker.status,
          lastSeen: worker.lastSeen,
          health: worker.health,
        },
      };
    } catch (error) {
      fastify.log.error('Failed to get worker details:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get worker details',
      });
    }
  });

  // Submit training job to worker
  fastify.post('/api/internal/workers/:workerId/training', {
    preHandler: [requireAuth],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const { workerId } = request.params as { workerId: string };
      const trainingData = request.body as {
        organizationId: string;
        modelType: string;
        trainingData: any;
        hyperparameters?: any;
      };

      const worker = workerRegistry.get(workerId);
      if (!worker) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Worker not found',
        });
      }

      if (!worker.capabilities.includes('training')) {
        return reply.status(400).send({
          error: 'Bad Request',
          message: 'Worker does not support training',
        });
      }

      const jobId = nanoid();

      // Register job
      jobRegistry.set(jobId, {
        jobId,
        workerId,
        jobType: 'training',
        organizationId: trainingData.organizationId,
        status: 'pending',
        createdAt: new Date(),
        updatedAt: new Date(),
      });

      // In a real implementation, you would send this to the worker via HTTP
      // For now, we'll simulate it
      fastify.log.info('Training job submitted to worker', {
        jobId,
        workerId,
        organizationId: trainingData.organizationId,
        modelType: trainingData.modelType,
      });

      return {
        success: true,
        jobId,
        message: 'Training job submitted to worker',
      };
    } catch (error) {
      fastify.log.error('Failed to submit training job:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to submit training job',
      });
    }
  });

  // Submit prediction job to worker
  fastify.post('/api/internal/workers/:workerId/prediction', {
    preHandler: [requireAuth],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const { workerId } = request.params as { workerId: string };
      const predictionData = request.body as {
        organizationId: string;
        modelId: string;
        inputData: any;
      };

      const worker = workerRegistry.get(workerId);
      if (!worker) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Worker not found',
        });
      }

      if (!worker.capabilities.includes('prediction')) {
        return reply.status(400).send({
          error: 'Bad Request',
          message: 'Worker does not support prediction',
        });
      }

      const jobId = nanoid();

      // Register job
      jobRegistry.set(jobId, {
        jobId,
        workerId,
        jobType: 'prediction',
        organizationId: predictionData.organizationId,
        status: 'pending',
        createdAt: new Date(),
        updatedAt: new Date(),
      });

      // In a real implementation, you would send this to the worker via HTTP
      fastify.log.info('Prediction job submitted to worker', {
        jobId,
        workerId,
        organizationId: predictionData.organizationId,
        modelId: predictionData.modelId,
      });

      return {
        success: true,
        jobId,
        message: 'Prediction job submitted to worker',
      };
    } catch (error) {
      fastify.log.error('Failed to submit prediction job:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to submit prediction job',
      });
    }
  });

  // Get job status
  fastify.get('/api/internal/jobs/:jobId', {
    preHandler: [requireAuth],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const { jobId } = request.params as { jobId: string };

      const job = jobRegistry.get(jobId);
      if (!job) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Job not found',
        });
      }

      return { job };
    } catch (error) {
      fastify.log.error('Failed to get job status:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to get job status',
      });
    }
  });

  // List all jobs
  fastify.get('/api/internal/jobs', {
    preHandler: [requireAuth],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const jobs = Array.from(jobRegistry.values()).map(job => ({
        jobId: job.jobId,
        workerId: job.workerId,
        jobType: job.jobType,
        organizationId: job.organizationId,
        status: job.status,
        createdAt: job.createdAt,
        updatedAt: job.updatedAt,
        result: job.result,
        error: job.error,
      }));

      return { jobs };
    } catch (error) {
      fastify.log.error('Failed to list jobs:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to list jobs',
      });
    }
  });

  // Cancel job
  fastify.post('/api/internal/jobs/:jobId/cancel', {
    preHandler: [requireAuth],
  }, async (request: AuthenticatedRequest, reply) => {
    try {
      const { jobId } = request.params as { jobId: string };

      const job = jobRegistry.get(jobId);
      if (!job) {
        return reply.status(404).send({
          error: 'Not Found',
          message: 'Job not found',
        });
      }

      if (job.status === 'completed' || job.status === 'failed') {
        return reply.status(400).send({
          error: 'Bad Request',
          message: 'Job cannot be cancelled',
        });
      }

      // Update job status
      job.status = 'cancelled';
      job.updatedAt = new Date();

      // Send cancel message to worker
      const worker = workerRegistry.get(job.workerId);
      if (worker) {
        fastify.log.info('Sending cancel message to worker', {
          jobId,
          workerId: job.workerId,
        });
      }

      return {
        success: true,
        message: 'Job cancellation requested',
      };
    } catch (error) {
      fastify.log.error('Failed to cancel job:', error);
      return reply.status(500).send({
        error: 'Internal Server Error',
        message: 'Failed to cancel job',
      });
    }
  });

  // Add audit logging to all routes
  fastify.addHook('onSend', auditLogResponse);
}

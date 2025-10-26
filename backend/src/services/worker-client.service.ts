import { FastifyInstance } from 'fastify';
import { nanoid } from 'nanoid';

export interface WorkerConfig {
  workerId: string;
  workerUrl: string;
  capabilities: string[];
  maxConcurrentJobs: number;
  status: 'active' | 'inactive' | 'maintenance';
}

export interface TrainingJobRequest {
  organizationId: string;
  modelType: string;
  trainingData: any;
  hyperparameters?: any;
  callbackUrl?: string;
}

export interface PredictionJobRequest {
  organizationId: string;
  modelId: string;
  inputData: any;
  callbackUrl?: string;
}

export interface JobResponse {
  jobId: string;
  status: string;
  progress?: number;
  result?: any;
  error?: string;
  startedAt?: string;
  completedAt?: string;
}

export class WorkerClientService {
  private workers: Map<string, WorkerConfig> = new Map();
  private fastify: FastifyInstance;

  constructor(fastify: FastifyInstance) {
    this.fastify = fastify;
  }

  /**
   * Register a worker
   */
  registerWorker(config: WorkerConfig): void {
    this.workers.set(config.workerId, config);
    this.fastify.log.info('Worker registered', { workerId: config.workerId });
  }

  /**
   * Get available workers by capability
   */
  getWorkersByCapability(capability: string): WorkerConfig[] {
    return Array.from(this.workers.values()).filter(
      worker => worker.capabilities.includes(capability) && worker.status === 'active'
    );
  }

  /**
   * Get worker by ID
   */
  getWorker(workerId: string): WorkerConfig | undefined {
    return this.workers.get(workerId);
  }

  /**
   * Submit training job to worker
   */
  async submitTrainingJob(
    workerId: string, 
    request: TrainingJobRequest
  ): Promise<JobResponse> {
    const worker = this.workers.get(workerId);
    if (!worker) {
      throw new Error(`Worker ${workerId} not found`);
    }

    if (!worker.capabilities.includes('training')) {
      throw new Error(`Worker ${workerId} does not support training`);
    }

    const jobId = nanoid();
    
    try {
      const response = await fetch(`${worker.workerUrl}/api/training/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_id: jobId,
          organization_id: request.organizationId,
          model_type: request.modelType,
          training_data: request.trainingData,
          hyperparameters: request.hyperparameters,
          callback_url: request.callbackUrl,
        }),
      });

      if (!response.ok) {
        throw new Error(`Worker responded with status ${response.status}`);
      }

      const result = await response.json();
      
      this.fastify.log.info('Training job submitted to worker', {
        jobId,
        workerId,
        organizationId: request.organizationId,
      });

      return {
        jobId,
        status: result.status,
        progress: result.progress,
        startedAt: result.started_at,
      };
    } catch (error) {
      this.fastify.log.error('Failed to submit training job', {
        jobId,
        workerId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Submit prediction job to worker
   */
  async submitPredictionJob(
    workerId: string, 
    request: PredictionJobRequest
  ): Promise<JobResponse> {
    const worker = this.workers.get(workerId);
    if (!worker) {
      throw new Error(`Worker ${workerId} not found`);
    }

    if (!worker.capabilities.includes('prediction')) {
      throw new Error(`Worker ${workerId} does not support prediction`);
    }

    const jobId = nanoid();
    
    try {
      const response = await fetch(`${worker.workerUrl}/api/prediction/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_id: jobId,
          organization_id: request.organizationId,
          model_id: request.modelId,
          input_data: request.inputData,
          callback_url: request.callbackUrl,
        }),
      });

      if (!response.ok) {
        throw new Error(`Worker responded with status ${response.status}`);
      }

      const result = await response.json();
      
      this.fastify.log.info('Prediction job submitted to worker', {
        jobId,
        workerId,
        organizationId: request.organizationId,
      });

      return {
        jobId,
        status: result.status,
        progress: result.progress,
        startedAt: result.started_at,
      };
    } catch (error) {
      this.fastify.log.error('Failed to submit prediction job', {
        jobId,
        workerId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Get job status from worker
   */
  async getJobStatus(workerId: string, jobId: string): Promise<JobResponse> {
    const worker = this.workers.get(workerId);
    if (!worker) {
      throw new Error(`Worker ${workerId} not found`);
    }

    try {
      const response = await fetch(`${worker.workerUrl}/api/jobs/${jobId}/status`);
      
      if (!response.ok) {
        throw new Error(`Worker responded with status ${response.status}`);
      }

      const result = await response.json();
      
      return {
        jobId: result.job_id,
        status: result.status,
        progress: result.progress,
        result: result.result,
        error: result.error,
        startedAt: result.started_at,
        completedAt: result.completed_at,
      };
    } catch (error) {
      this.fastify.log.error('Failed to get job status', {
        jobId,
        workerId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Cancel job on worker
   */
  async cancelJob(workerId: string, jobId: string): Promise<void> {
    const worker = this.workers.get(workerId);
    if (!worker) {
      throw new Error(`Worker ${workerId} not found`);
    }

    try {
      const response = await fetch(`${worker.workerUrl}/api/jobs/${jobId}/cancel`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Worker responded with status ${response.status}`);
      }

      this.fastify.log.info('Job cancellation requested', {
        jobId,
        workerId,
      });
    } catch (error) {
      this.fastify.log.error('Failed to cancel job', {
        jobId,
        workerId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Send internal message to worker
   */
  async sendMessage(
    workerId: string, 
    messageType: string, 
    payload: any
  ): Promise<void> {
    const worker = this.workers.get(workerId);
    if (!worker) {
      throw new Error(`Worker ${workerId} not found`);
    }

    try {
      const response = await fetch(`${worker.workerUrl}/api/internal/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message_type: messageType,
          payload,
          timestamp: new Date().toISOString(),
          worker_id: workerId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Worker responded with status ${response.status}`);
      }

      this.fastify.log.info('Message sent to worker', {
        workerId,
        messageType,
      });
    } catch (error) {
      this.fastify.log.error('Failed to send message to worker', {
        workerId,
        messageType,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Check worker health
   */
  async checkWorkerHealth(workerId: string): Promise<any> {
    const worker = this.workers.get(workerId);
    if (!worker) {
      throw new Error(`Worker ${workerId} not found`);
    }

    try {
      const response = await fetch(`${worker.workerUrl}/health`);
      
      if (!response.ok) {
        throw new Error(`Worker responded with status ${response.status}`);
      }

      const health = await response.json();
      
      this.fastify.log.debug('Worker health checked', {
        workerId,
        status: health.status,
      });

      return health;
    } catch (error) {
      this.fastify.log.error('Failed to check worker health', {
        workerId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Get all workers
   */
  getAllWorkers(): WorkerConfig[] {
    return Array.from(this.workers.values());
  }

  /**
   * Remove worker
   */
  removeWorker(workerId: string): void {
    this.workers.delete(workerId);
    this.fastify.log.info('Worker removed', { workerId });
  }

  /**
   * Update worker status
   */
  updateWorkerStatus(workerId: string, status: 'active' | 'inactive' | 'maintenance'): void {
    const worker = this.workers.get(workerId);
    if (worker) {
      worker.status = status;
      this.fastify.log.info('Worker status updated', { workerId, status });
    }
  }
}

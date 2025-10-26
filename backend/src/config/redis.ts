import Redis from 'ioredis';
import { config } from './env';

export const redis = new Redis(config.REDIS_URL, {
  retryDelayOnFailover: 100,
  enableReadyCheck: false,
  maxRetriesPerRequest: null,
});

redis.on('connect', () => {
  console.log('✅ Redis connected');
});

redis.on('error', (error) => {
  console.error('❌ Redis error:', error);
});

export default redis;

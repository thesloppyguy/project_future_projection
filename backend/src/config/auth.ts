import { betterAuth } from 'better-auth';
import { drizzleAdapter } from 'better-auth/adapters/drizzle';
import { db } from '../db';
import { config } from './env';

export const auth = betterAuth({
  database: drizzleAdapter(db, {
    provider: 'pg',
  }),
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: true,
  },
  socialProviders: {
    // Add social providers later if needed
  },
  session: {
    expiresIn: 60 * 60 * 24 * 7, // 7 days
    updateAge: 60 * 60 * 24, // 1 day
  },
  user: {
    additionalFields: {
      role: {
        type: 'string',
        defaultValue: 'team_user',
      },
      status: {
        type: 'string',
        defaultValue: 'pending',
      },
    },
  },
  plugins: [
    // Add plugins later if needed
  ],
  secret: config.BETTER_AUTH_SECRET,
  baseURL: config.BETTER_AUTH_URL,
  trustedOrigins: [config.BETTER_AUTH_URL],
});

export type Session = typeof auth.$Infer.Session;
export type User = typeof auth.$Infer.User;

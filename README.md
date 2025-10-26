# Multi-tenant Auth Platform

A comprehensive multi-tenant SaaS platform with role-based access control, built with Next.js 15, Fastify, Better Auth, and PostgreSQL.

## 🚀 Features

### Authentication & Authorization
- **Better Auth** integration with email/password authentication
- **Invite-only registration** with secure token-based invitations
- **Role-based access control** with 4 distinct roles:
  - `maintainer` - Platform super admin (not tied to any org)
  - `org_admin` - Full organization control
  - `team_admin` - Team management (can be in multiple teams)
  - `team_user` - Basic access (can be in multiple teams)
- **Password reset flow** with email verification
- **User impersonation** for maintainers (debugging support)

### Multi-tenancy
- **Organization isolation** with separate data and settings
- **Team management** within organizations
- **Member invitations** with role assignment
- **Custom role groups** with granular permissions

### Platform Management
- **Maintainer dashboard** for platform-wide administration
- **Organization management** (create, view, edit organizations)
- **User management** with role assignment
- **File uploads** with org/user association
- **Webhook system** with retry logic and delivery tracking

### Monitoring & Logging
- **Audit logging** for all user actions
- **Data sync logs** for synchronization events
- **Training logs** for ML/AI job tracking
- **Notification logs** for email/in-app notifications
- **System monitoring** with queue status and health checks

### Advanced Features
- **Time Series Analytics** with ClickHouse for high-performance data ingestion and querying
- **Daily Data Ingestion** service for automated time series data processing
- **Real-time Analytics** with materialized views and aggregations
- **Python Worker Service** for ML model training and predictions
- **Bidirectional Communication** between backend and worker services
- **Scheduled tasks** with BullMQ integration
- **Queue management** for background jobs
- **Rate limiting** (global, per-org, per-user)
- **Role preview mode** for org admins
- **Email integration** with Resend
- **Webhook delivery** with exponential backoff

## 🏗️ Architecture

### Backend (Fastify)
- **Framework**: Fastify with TypeScript
- **Database**: PostgreSQL with Drizzle ORM
- **Authentication**: Better Auth with Drizzle adapter
- **Queue**: BullMQ with Redis
- **Email**: Resend integration
- **File Storage**: Local filesystem (S3 ready)

### Frontend (Next.js 15)
- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui
- **State Management**: TanStack Query
- **Forms**: React Hook Form with Zod validation
- **Icons**: Lucide React

### Python Worker Service
- **Framework**: FastAPI with Python 3.11
- **ML Libraries**: scikit-learn, pandas, numpy
- **Communication**: HTTP-based bidirectional communication
- **Monitoring**: Prometheus metrics and health checks
- **Job Processing**: Async model training and predictions

### Infrastructure
- **Containerization**: Docker Compose
- **Database**: PostgreSQL 16
- **Time Series Database**: ClickHouse 24.8
- **Cache/Queue**: Redis 7
- **File Storage**: Docker volumes

## 📋 Prerequisites

- Node.js 22+
- Docker and Docker Compose
- Git

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd cdro
```

### 2. Environment Setup

Create environment files:

**Backend** (`backend/.env`):
```env
NODE_ENV=development
PORT=3000
HOST=0.0.0.0

# Database
DATABASE_URL=postgres://sashflow:password@localhost:5432/sashflow_db
DBMATE_DATABASE_URL=postgres://sashflow:password@localhost:5432/sashflow_db

# Redis
REDIS_URL=redis://localhost:6379

# ClickHouse
CLICKHOUSE_URL=http://localhost:8123
CLICKHOUSE_USERNAME=default
CLICKHOUSE_PASSWORD=clickhouse_password
CLICKHOUSE_DATABASE=timeseries_db

# Auth
BETTER_AUTH_SECRET=your-super-secret-better-auth-key-change-this-in-production-32-chars-min
BETTER_AUTH_URL=http://localhost:3000
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production

# Email
RESEND_API_KEY=your-resend-api-key-here

# CORS
CORS_ORIGIN=http://localhost:3000,http://localhost:3001

# File uploads
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=10485760

# Logging
LOG_LEVEL=info
```

**Frontend** (`frontend/.env.local`):
```env
NEXT_PUBLIC_API_URL=http://localhost:3000
BETTER_AUTH_SECRET=your-super-secret-better-auth-key-change-this-in-production-32-chars-min
BETTER_AUTH_URL=http://localhost:3000
```

### 3. Start with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 4. Access the Application

- **Frontend**: http://localhost:3001
- **Backend API**: http://localhost:3000
- **Database**: localhost:5432
- **ClickHouse**: http://localhost:8123
- **Redis**: localhost:6379

### 5. Initial Setup

The application will automatically:
1. Run database migrations
2. Seed the database with a maintainer user
3. Create a default organization

**Default Maintainer Credentials**:
- Email: `maintainer@platform.com`
- Password: `maintainer123!`

⚠️ **Important**: Change the password after first login!

## 🛠️ Development

### Backend Development

```bash
cd backend

# Install dependencies
pnpm install

# Run database migrations
pnpm db:generate
pnpm db:migrate

# Seed database
pnpm db:seed

# Start development server
pnpm start
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Database Management

```bash
# Generate migrations
pnpm db:generate

# Run migrations
pnpm db:migrate

# Open Drizzle Studio
pnpm db:studio

# Seed database
pnpm db:seed
```

## 📚 API Documentation

### Authentication Endpoints

- `POST /api/auth/sign-in` - Sign in with email/password
- `POST /api/auth/sign-up` - Register with invitation token
- `POST /api/auth/forgot-password` - Request password reset
- `POST /api/auth/reset-password` - Reset password with token
- `GET /api/auth/me` - Get current user
- `POST /api/auth/logout` - Sign out

### Maintainer Endpoints

- `GET /api/maintainer/overview` - Platform overview
- `GET /api/maintainer/organizations` - List organizations
- `POST /api/maintainer/organizations` - Create organization
- `GET /api/maintainer/users` - List users
- `POST /api/maintainer/users/maintainer` - Create maintainer
- `GET /api/maintainer/logs/audit` - Get audit logs
- `GET /api/maintainer/system/status` - System status
- `POST /api/maintainer/impersonate/:userId` - Impersonate user

### Time Series & Analytics Endpoints

- `POST /api/timeseries/metrics` - Insert single metric
- `POST /api/timeseries/metrics/batch` - Insert multiple metrics
- `POST /api/timeseries/events` - Insert event
- `POST /api/timeseries/logs` - Insert log entry
- `POST /api/timeseries/performance` - Insert performance metric
- `POST /api/timeseries/business` - Insert business metric
- `POST /api/timeseries/activity` - Insert user activity
- `GET /api/timeseries/metrics` - Query metrics
- `GET /api/timeseries/metrics/summary` - Get daily metrics summary
- `GET /api/timeseries/performance/summary` - Get performance summary
- `GET /api/timeseries/activity/summary` - Get user activity summary
- `GET /api/timeseries/metrics/top` - Get top metrics by value

### Analytics Endpoints

- `GET /api/analytics/dashboard` - Get dashboard metrics
- `GET /api/analytics/timeseries` - Get time series data for charts
- `GET /api/analytics/performance` - Get performance metrics
- `GET /api/analytics/engagement` - Get user engagement metrics
- `GET /api/analytics/business` - Get business metrics
- `GET /api/analytics/system` - Get system health metrics
- `POST /api/analytics/ingestion/trigger` - Trigger daily data ingestion
- `GET /api/analytics/ingestion/status` - Get ingestion status
- `POST /api/analytics/ingestion/schedule-all` - Schedule ingestion for all orgs

### ClickHouse Management Endpoints (Maintainer Only)

- `GET /api/maintainer/clickhouse/status` - Get ClickHouse status and version
- `GET /api/maintainer/clickhouse/tables` - Get table statistics
- `POST /api/maintainer/clickhouse/optimize` - Optimize all tables
- `POST /api/maintainer/clickhouse/cleanup` - Cleanup old data

## 🔐 Security Features

### Authentication Security
- **Better Auth** with secure session management
- **CSRF protection** built-in
- **Password hashing** with bcrypt
- **Email verification** required
- **Invite-only registration**

### Authorization Security
- **Role-based access control** with granular permissions
- **Organization isolation** with tenant context
- **Permission middleware** for route protection
- **Audit logging** for all sensitive actions

### Infrastructure Security
- **Rate limiting** on all endpoints
- **SQL injection prevention** with Drizzle ORM
- **File upload validation** (type, size)
- **Webhook signature verification**
- **HTTPS enforcement** in production

## 📊 Monitoring & Observability

### Audit Logging
All user actions are logged with:
- User ID and organization context
- Action type and resource
- IP address and user agent
- Timestamp and metadata

### System Monitoring
- **Queue status** (pending, active, completed, failed)
- **Webhook delivery** tracking
- **Scheduled task** monitoring
- **Database connection** health
- **Redis connection** health

### Logs Available
- **Audit logs** - User actions and system events
- **Data sync logs** - Data synchronization events
- **Training logs** - ML/AI training job history
- **Notification logs** - Email and notification delivery

## 🚀 Deployment

### Production Environment Variables

Update the following for production:

```env
# Security
BETTER_AUTH_SECRET=<32-char-random-string>
JWT_SECRET=<32-char-random-string>

# Database
DATABASE_URL=<production-postgres-url>

# Redis
REDIS_URL=<production-redis-url>

# Email
RESEND_API_KEY=<production-resend-key>

# CORS
CORS_ORIGIN=<production-frontend-url>

# File Storage (optional)
AWS_ACCESS_KEY_ID=<s3-access-key>
AWS_SECRET_ACCESS_KEY=<s3-secret-key>
AWS_S3_BUCKET=<s3-bucket-name>
AWS_S3_REGION=<s3-region>
```

### Docker Production Build

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Start production services
docker-compose -f docker-compose.prod.yml up -d
```

### Health Checks

- **Backend**: `GET /api/v1/health`
- **Frontend**: `GET /api/health`
- **Database**: PostgreSQL health check
- **Redis**: Redis ping check

## 🧪 Testing

### Backend Tests

```bash
cd backend

# Unit tests
pnpm test

# Integration tests
pnpm test:e2e

# Coverage
pnpm test:coverage
```

### Frontend Tests

```bash
cd frontend

# Unit tests
npm test

# E2E tests
npm run test:e2e
```

## 📝 Database Schema

### Core Tables

- `users` - User accounts with roles and status
- `organizations` - Multi-tenant organizations
- `teams` - Teams within organizations
- `organization_members` - User-organization relationships
- `team_members` - User-team relationships
- `invitations` - Pending user invitations
- `audit_logs` - System audit trail
- `webhooks` - Webhook configurations
- `file_uploads` - File metadata
- `scheduled_tasks` - Task definitions
- `notification_logs` - Notification history

### Relationships

- Users can belong to multiple organizations
- Users can belong to multiple teams
- Organizations contain multiple teams
- Teams contain multiple users
- All actions are audited
- Webhooks can be configured per organization

## 🔗 Webhook System

The platform includes a comprehensive webhook system for real-time event notifications.

### Webhook Endpoints

#### Public Endpoints (No Authentication)
- `POST /webhook/:webhookId` - Receive webhook requests from external services
- `POST /webhook/test/:webhookId` - Public webhook testing endpoint

#### Management Endpoints (Authentication Required)
- `POST /api/webhooks` - Create webhook configuration
- `GET /api/webhooks` - List organization webhooks
- `GET /api/webhooks/:id` - Get webhook details
- `PUT /api/webhooks/:id` - Update webhook configuration
- `DELETE /api/webhooks/:id` - Delete webhook
- `POST /api/webhooks/:id/test` - Test webhook delivery
- `GET /api/webhooks/:id/events` - Get webhook delivery events
- `POST /api/webhooks/:id/events/:eventId/retry` - Retry failed webhook
- `GET /api/webhooks/:id/stats` - Get webhook statistics

### Webhook Features

- **Security**: HMAC-SHA256 signature validation
- **Reliability**: Configurable retry logic with exponential backoff
- **Event Routing**: Automatic routing based on event types
- **Monitoring**: Comprehensive delivery statistics and event logs
- **Testing**: Built-in webhook testing capabilities
- **Multi-tenant**: Organization-scoped webhook management

### Event Types

- **User Events**: `user.created`, `user.updated`, `user.deleted`, `user.activated`
- **Order Events**: `order.created`, `order.updated`, `order.cancelled`, `order.completed`
- **Payment Events**: `payment.created`, `payment.completed`, `payment.failed`, `payment.refunded`
- **System Events**: `system.maintenance`, `system.error`, `system.alert`

### Testing Webhooks

```bash
# Test webhook with default payload
node backend/scripts/test-webhook.js https://your-app.com/webhook/webhook_123

# Test with custom payload
node backend/scripts/test-webhook.js https://your-app.com/webhook/webhook_123 backend/scripts/sample-webhook-payload.json

# Set custom secret
WEBHOOK_SECRET=your-secret node backend/scripts/test-webhook.js https://your-app.com/webhook/webhook_123
```

For detailed webhook documentation, see [backend/docs/WEBHOOKS.md](backend/docs/WEBHOOKS.md).

## 🤖 ML Worker Service

The platform includes a Python worker service for machine learning model training and predictions.

### ML Endpoints

#### Model Training
- `POST /api/ml/training/start` - Start model training job
- `GET /api/ml/training/:jobId/status` - Get training job status
- `GET /api/ml/training` - List training jobs
- `POST /api/ml/training/:jobId/cancel` - Cancel training job

#### Model Predictions
- `POST /api/ml/prediction/start` - Start prediction job
- `GET /api/ml/prediction/:jobId/status` - Get prediction job status

#### Model Management
- `GET /api/ml/models` - List available models
- `GET /api/ml/workers` - Get worker status and health

### Worker Communication

#### Internal API Routes
- `POST /api/internal/workers/register` - Worker registration
- `POST /api/internal/workers/health` - Worker health updates
- `POST /api/internal/workers/job-update` - Job status updates
- `POST /api/internal/workers/:workerId/message` - Send message to worker
- `GET /api/internal/workers` - List registered workers
- `POST /api/internal/workers/:workerId/training` - Submit training job
- `POST /api/internal/workers/:workerId/prediction` - Submit prediction job

### Worker Features

- **Model Types**: Classification, regression, clustering, neural networks
- **Async Processing**: Background job execution with status tracking
- **Health Monitoring**: Comprehensive health checks and metrics
- **Bidirectional Communication**: HTTP-based communication with backend
- **Job Management**: Start, monitor, and cancel ML jobs
- **Model Caching**: In-memory model caching for faster predictions

### Testing the Worker

```bash
# Test worker health
curl http://localhost:8000/health

# Start training job
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"job_id": "test", "organization_id": "test", "model_type": "classification", "training_data": {}}'

# Check job status
curl http://localhost:8000/api/jobs/test/status
```

For detailed worker documentation, see [worker/README.md](worker/README.md).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the ISC License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the API endpoints

## 🔄 Roadmap

- [ ] S3 file storage integration
- [ ] Advanced webhook filtering
- [ ] Custom email templates
- [ ] Advanced analytics dashboard
- [ ] Mobile app support
- [ ] SSO integration (SAML, OAuth)
- [ ] Advanced role customization
- [ ] API rate limiting per user
- [ ] Real-time notifications
- [ ] Advanced audit log filtering

---

Built with ❤️ using Next.js, Fastify, and Better Auth.

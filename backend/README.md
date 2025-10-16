# Sashflow Backend

A production-ready Fastify server with comprehensive features including GraphQL, REST API, database integration, and monitoring.

## Features

- 🚀 **Fastify Server** - High-performance Node.js web framework
- 🔒 **Security** - Helmet, CORS, rate limiting, and security headers
- 📊 **GraphQL** - Full GraphQL support with Mercurius
- 🗄️ **Database** - PostgreSQL with Drizzle ORM and connection pooling
- 🔧 **Dependency Injection** - Awilix container for clean architecture
- 📝 **API Documentation** - Swagger/OpenAPI documentation
- 🏥 **Health Checks** - Comprehensive health monitoring
- 📈 **Monitoring** - Request logging, performance monitoring, and under-pressure detection
- 🐳 **Docker** - Production-ready Docker configuration
- 🧪 **Testing** - Unit and E2E testing setup
- 📦 **TypeScript** - Full TypeScript support with strict configuration

## Quick Start

### Prerequisites

- Node.js 22+
- PostgreSQL 16+
- pnpm (recommended) or npm

### Installation

1. Clone the repository and navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
pnpm install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Set up the database:
```bash
# Generate and run migrations
pnpm run db:generate
pnpm run db:migrate

# Or push schema directly (for development)
pnpm run db:push

# Seed the database (optional)
pnpm run db:seed
```

5. Start the development server:
```bash
pnpm start
```

The server will be available at:
- API: http://localhost:3000
- API Documentation: http://localhost:3000/api-docs
- GraphQL Playground: http://localhost:3000/graphql

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NODE_ENV` | Environment mode | `development` |
| `PORT` | Server port | `3000` |
| `HOST` | Server host | `0.0.0.0` |
| `DBMATE_DATABASE_URL` | PostgreSQL connection string | Required |
| `JWT_SECRET` | JWT signing secret | Required |
| `API_KEY` | API key for authentication | Required |
| `CORS_ORIGIN` | CORS allowed origins | `http://localhost:3000` |
| `LOG_LEVEL` | Logging level | `info` |
| `RATE_LIMIT_MAX` | Rate limit max requests | `100` |
| `RATE_LIMIT_TIME_WINDOW` | Rate limit time window (ms) | `60000` |

## API Endpoints

### REST API

- `GET /api/v1/users` - Get all users
- `GET /api/v1/users/:id` - Get user by ID
- `POST /api/v1/users` - Create user
- `PUT /api/v1/users/:id` - Update user
- `DELETE /api/v1/users/:id` - Delete user
- `GET /api/v1/health` - Health check
- `GET /api/v1/ready` - Readiness probe
- `GET /api/v1/live` - Liveness probe

### GraphQL

- GraphQL endpoint: `/graphql`
- GraphQL Playground: `/graphql` (in development)

## Scripts

| Script | Description |
|--------|-------------|
| `pnpm start` | Start development server with hot reload |
| `pnpm start:prod` | Start production server |
| `pnpm build` | Build for production |
| `pnpm test` | Run unit tests |
| `pnpm test:e2e` | Run E2E tests |
| `pnpm lint` | Run ESLint |
| `pnpm format` | Format code with Prettier |
| `pnpm db:generate` | Generate Drizzle migrations |
| `pnpm db:migrate` | Run database migrations |
| `pnpm db:push` | Push schema to database (dev) |
| `pnpm db:studio` | Open Drizzle Studio |
| `pnpm db:seed` | Seed database |

## Docker

### Development

```bash
docker-compose up -d
```

### Production

```bash
docker build -t sashflow-backend .
docker run -p 3000:3000 --env-file .env sashflow-backend
```

## Architecture

The application follows a clean architecture pattern:

```
src/
├── config/          # Configuration files
├── controllers/     # Request handlers
├── services/        # Business logic
├── repositories/    # Data access layer
├── routes/          # Route definitions
├── plugins/         # Fastify plugins
├── graphql/         # GraphQL schema and resolvers
└── types/           # TypeScript type definitions
```

## Database

The application uses PostgreSQL with Drizzle ORM and the following features:

- Type-safe database queries
- Connection pooling
- Migrations with Drizzle Kit
- Seeds for development data
- UUID primary keys
- Timestamps with timezone support
- Schema validation with Zod

## Monitoring

The application includes comprehensive monitoring:

- Request/response logging
- Performance metrics
- Health checks
- Under-pressure detection
- Memory usage monitoring
- Database connection monitoring

## Security

Security features include:

- Helmet for security headers
- CORS configuration
- Rate limiting
- Input validation
- SQL injection prevention
- XSS protection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

ISC

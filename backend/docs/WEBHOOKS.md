# Webhook System Documentation

## Overview

The webhook system allows external services to send real-time notifications to your application. It provides secure, reliable webhook delivery with signature validation, retry mechanisms, and comprehensive monitoring.

## Features

- **Secure Delivery**: HMAC-SHA256 signature validation
- **Retry Logic**: Configurable retry attempts for failed deliveries
- **Event Routing**: Automatic routing based on event types
- **Monitoring**: Comprehensive statistics and event logging
- **Testing**: Built-in webhook testing capabilities
- **Multi-tenant**: Organization-scoped webhook management

## Webhook Endpoints

### Public Endpoints (No Authentication Required)

#### Receive Webhook
```
POST /webhook/:webhookId
```

Receives incoming webhook requests from external services.

**Headers:**
- `Content-Type: application/json`
- `X-Webhook-Signature` (optional): HMAC-SHA256 signature
- `X-Webhook-Event`: Event name
- `X-Webhook-Type`: Event type

**Response:**
```json
{
  "success": true,
  "message": "Webhook processed successfully"
}
```

#### Test Webhook (Public)
```
POST /webhook/test/:webhookId
```

Public endpoint for testing webhook delivery.

**Request Body:**
```json
{
  "type": "user",
  "name": "user.created",
  "data": {
    "userId": "123",
    "email": "test@example.com"
  }
}
```

### Authenticated Endpoints

#### Create Webhook
```
POST /api/webhooks
```

**Request Body:**
```json
{
  "name": "User Events Webhook",
  "url": "https://example.com/webhook",
  "events": ["user.created", "user.updated"],
  "secret": "your-webhook-secret",
  "isActive": true,
  "retryCount": 3,
  "timeout": 30
}
```

#### List Webhooks
```
GET /api/webhooks
```

**Response:**
```json
{
  "webhooks": [
    {
      "id": "webhook_123",
      "name": "User Events Webhook",
      "url": "https://example.com/webhook",
      "events": ["user.created", "user.updated"],
      "isActive": true,
      "retryCount": "3",
      "timeout": "30",
      "createdAt": "2024-01-15T10:30:00.000Z"
    }
  ]
}
```

#### Get Webhook
```
GET /api/webhooks/:webhookId
```

#### Update Webhook
```
PUT /api/webhooks/:webhookId
```

#### Delete Webhook
```
DELETE /api/webhooks/:webhookId
```

#### Test Webhook
```
POST /api/webhooks/:webhookId/test
```

**Request Body:**
```json
{
  "eventType": "user",
  "eventName": "user.created",
  "payload": {
    "userId": "123",
    "email": "test@example.com"
  }
}
```

#### Get Webhook Events
```
GET /api/webhooks/:webhookId/events?page=1&limit=50
```

#### Retry Failed Webhook
```
POST /api/webhooks/:webhookId/events/:eventId/retry
```

#### Get Webhook Statistics
```
GET /api/webhooks/:webhookId/stats?startDate=2024-01-01&endDate=2024-01-31
```

**Response:**
```json
{
  "stats": {
    "totalEvents": 150,
    "successfulDeliveries": 145,
    "failedDeliveries": 5,
    "successRate": 96.7,
    "averageResponseTime": 250,
    "lastDelivery": "2024-01-15T10:30:00.000Z",
    "eventsByStatus": {
      "delivered": 145,
      "failed": 5
    },
    "eventsByDay": [
      {
        "date": "2024-01-15",
        "count": 10,
        "success": 9,
        "failed": 1
      }
    ]
  }
}
```

## Event Types

### User Events
- `user.created` - New user registered
- `user.updated` - User profile updated
- `user.deleted` - User account deleted
- `user.activated` - User account activated

### Order Events
- `order.created` - New order placed
- `order.updated` - Order status updated
- `order.cancelled` - Order cancelled
- `order.completed` - Order completed

### Payment Events
- `payment.created` - Payment initiated
- `payment.completed` - Payment successful
- `payment.failed` - Payment failed
- `payment.refunded` - Payment refunded

### System Events
- `system.maintenance` - System maintenance started/ended
- `system.error` - System error occurred
- `system.alert` - System alert triggered

## Signature Validation

Webhooks use HMAC-SHA256 for signature validation:

```javascript
const crypto = require('crypto');

function generateSignature(payload, secret) {
  const body = typeof payload === 'string' ? payload : JSON.stringify(payload);
  return crypto
    .createHmac('sha256', secret)
    .update(body)
    .digest('hex');
}

// Validate signature
const signature = request.headers['x-webhook-signature'];
const expectedSignature = generateSignature(request.body, webhookSecret);
const isValid = crypto.timingSafeEqual(
  Buffer.from(signature, 'hex'),
  Buffer.from(expectedSignature, 'hex')
);
```

## Testing

### Using the Test Script

```bash
# Test with default payload
node scripts/test-webhook.js https://your-app.com/webhook/webhook_123

# Test with custom payload
node scripts/test-webhook.js https://your-app.com/webhook/webhook_123 scripts/sample-webhook-payload.json

# Set custom secret
WEBHOOK_SECRET=your-secret node scripts/test-webhook.js https://your-app.com/webhook/webhook_123
```

### Using cURL

```bash
# Test webhook delivery
curl -X POST https://your-app.com/webhook/webhook_123 \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Event: user.created" \
  -H "X-Webhook-Type: user" \
  -H "X-Webhook-Signature: sha256=your-signature" \
  -d '{
    "type": "user",
    "name": "user.created",
    "data": {
      "userId": "123",
      "email": "test@example.com"
    }
  }'
```

### Using the Frontend

1. Navigate to the Webhooks page in your dashboard
2. Click the "Test" button on any webhook
3. Configure the test event type, name, and payload
4. Click "Test Webhook" to send the test

## Error Handling

### HTTP Status Codes

- `200` - Webhook processed successfully
- `400` - Bad request (invalid payload)
- `401` - Unauthorized (invalid signature)
- `404` - Webhook not found or inactive
- `500` - Internal server error

### Retry Logic

Failed webhooks are automatically retried based on the configured retry count:

1. **Immediate Retry**: First retry happens immediately
2. **Exponential Backoff**: Subsequent retries use exponential backoff
3. **Max Retries**: Retries stop after reaching the configured limit
4. **Manual Retry**: Failed events can be manually retried via the API

### Error Logging

All webhook events are logged with:
- Event payload
- Response status and body
- Error messages
- Retry attempts
- Delivery timestamps

## Security Considerations

1. **Signature Validation**: Always validate webhook signatures
2. **HTTPS Only**: Use HTTPS endpoints for webhook URLs
3. **Secret Management**: Store webhook secrets securely
4. **Rate Limiting**: Implement rate limiting on webhook endpoints
5. **IP Whitelisting**: Consider IP whitelisting for trusted sources
6. **Payload Validation**: Validate and sanitize webhook payloads

## Monitoring and Analytics

### Webhook Statistics

- Total events processed
- Success/failure rates
- Average response times
- Events by status
- Daily event trends

### Event Logs

- Complete event history
- Response details
- Error messages
- Retry attempts
- Delivery timestamps

### Alerts

- Failed webhook deliveries
- High error rates
- Slow response times
- Webhook endpoint downtime

## Best Practices

1. **Idempotency**: Make webhook handlers idempotent
2. **Async Processing**: Process webhooks asynchronously when possible
3. **Timeout Handling**: Set appropriate timeouts for webhook delivery
4. **Error Recovery**: Implement proper error recovery mechanisms
5. **Monitoring**: Monitor webhook delivery success rates
6. **Documentation**: Document webhook payloads and expected responses
7. **Versioning**: Version your webhook payloads for backward compatibility

## Troubleshooting

### Common Issues

1. **Signature Validation Fails**
   - Check secret configuration
   - Verify payload format
   - Ensure proper header format

2. **Webhook Not Receiving Events**
   - Verify webhook is active
   - Check URL accessibility
   - Confirm event types are configured

3. **High Failure Rates**
   - Check endpoint availability
   - Verify payload format
   - Review timeout settings

4. **Slow Response Times**
   - Optimize webhook handler
   - Check network connectivity
   - Review server resources

### Debug Mode

Enable debug logging by setting the log level to `debug` in your environment configuration.

## Integration Examples

### Node.js Webhook Handler

```javascript
const express = require('express');
const crypto = require('crypto');

const app = express();
app.use(express.json());

app.post('/webhook', (req, res) => {
  const signature = req.headers['x-webhook-signature'];
  const payload = JSON.stringify(req.body);
  
  // Validate signature
  const expectedSignature = crypto
    .createHmac('sha256', process.env.WEBHOOK_SECRET)
    .update(payload)
    .digest('hex');
  
  if (signature !== expectedSignature) {
    return res.status(401).send('Invalid signature');
  }
  
  // Process webhook
  console.log('Received webhook:', req.body);
  
  res.status(200).send('OK');
});

app.listen(3000);
```

### Python Webhook Handler

```python
from flask import Flask, request, jsonify
import hmac
import hashlib
import json

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    signature = request.headers.get('X-Webhook-Signature')
    payload = request.get_data()
    
    # Validate signature
    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(signature, expected_signature):
        return jsonify({'error': 'Invalid signature'}), 401
    
    # Process webhook
    data = request.get_json()
    print(f'Received webhook: {data}')
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(port=3000)
```

## API Rate Limits

Webhook endpoints are subject to rate limiting:

- **Public Endpoints**: 100 requests per minute per IP
- **Authenticated Endpoints**: 1000 requests per minute per user
- **Webhook Delivery**: 10 requests per second per webhook

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

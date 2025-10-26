#!/usr/bin/env node

/**
 * Webhook Testing Script
 * 
 * This script demonstrates how to test webhook endpoints
 * Usage: node scripts/test-webhook.js <webhook-url> [payload-file]
 */

const https = require('https');
const http = require('http');
const fs = require('fs');
const crypto = require('crypto');

// Configuration
const WEBHOOK_URL = process.argv[2];
const PAYLOAD_FILE = process.argv[3];
const SECRET = process.env.WEBHOOK_SECRET || 'test-secret';

if (!WEBHOOK_URL) {
  console.error('Usage: node test-webhook.js <webhook-url> [payload-file]');
  process.exit(1);
}

// Default test payload
const defaultPayload = {
  type: 'user',
  name: 'user.created',
  timestamp: new Date().toISOString(),
  test: true,
  data: {
    userId: 'test-user-123',
    email: 'test@example.com',
    name: 'Test User',
    organizationId: 'test-org-456',
  },
};

// Load payload from file or use default
let payload;
if (PAYLOAD_FILE && fs.existsSync(PAYLOAD_FILE)) {
  try {
    payload = JSON.parse(fs.readFileSync(PAYLOAD_FILE, 'utf8'));
    console.log(`📄 Loaded payload from ${PAYLOAD_FILE}`);
  } catch (error) {
    console.error(`❌ Error reading payload file: ${error.message}`);
    process.exit(1);
  }
} else {
  payload = defaultPayload;
  console.log('📦 Using default test payload');
}

// Generate webhook signature
function generateSignature(payload, secret) {
  const body = JSON.stringify(payload);
  return crypto
    .createHmac('sha256', secret)
    .update(body)
    .digest('hex');
}

// Make webhook request
function makeWebhookRequest(url, payload, secret) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload);
    const signature = generateSignature(payload, secret);
    
    const urlObj = new URL(url);
    const isHttps = urlObj.protocol === 'https:';
    const client = isHttps ? https : http;
    
    const options = {
      hostname: urlObj.hostname,
      port: urlObj.port || (isHttps ? 443 : 80),
      path: urlObj.pathname + urlObj.search,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
        'User-Agent': 'WebhookTestScript/1.0',
        'X-Webhook-Event': payload.name || 'test.event',
        'X-Webhook-Type': payload.type || 'test',
        'X-Webhook-Signature': signature,
        'X-Webhook-Timestamp': new Date().toISOString(),
      },
    };

    console.log(`🚀 Sending webhook to: ${url}`);
    console.log(`📋 Headers:`, options.headers);
    console.log(`📦 Payload:`, JSON.stringify(payload, null, 2));

    const req = client.request(options, (res) => {
      let responseBody = '';
      
      res.on('data', (chunk) => {
        responseBody += chunk;
      });
      
      res.on('end', () => {
        const result = {
          statusCode: res.statusCode,
          headers: res.headers,
          body: responseBody,
        };
        
        console.log(`\n📊 Response:`);
        console.log(`   Status: ${res.statusCode}`);
        console.log(`   Headers:`, res.headers);
        console.log(`   Body:`, responseBody);
        
        if (res.statusCode >= 200 && res.statusCode < 300) {
          console.log(`✅ Webhook delivered successfully!`);
          resolve(result);
        } else {
          console.log(`❌ Webhook delivery failed!`);
          reject(new Error(`HTTP ${res.statusCode}: ${responseBody}`));
        }
      });
    });

    req.on('error', (error) => {
      console.error(`❌ Request error:`, error.message);
      reject(error);
    });

    req.on('timeout', () => {
      console.error(`❌ Request timeout`);
      req.destroy();
      reject(new Error('Request timeout'));
    });

    req.setTimeout(30000); // 30 second timeout
    req.write(body);
    req.end();
  });
}

// Test webhook signature validation
function testSignatureValidation() {
  console.log('\n🔐 Testing signature validation...');
  
  const testPayload = { test: 'signature validation' };
  const signature = generateSignature(testPayload, SECRET);
  
  console.log(`   Payload: ${JSON.stringify(testPayload)}`);
  console.log(`   Secret: ${SECRET}`);
  console.log(`   Signature: ${signature}`);
  
  // Verify signature
  const expectedSignature = crypto
    .createHmac('sha256', SECRET)
    .update(JSON.stringify(testPayload))
    .digest('hex');
  
  const isValid = signature === expectedSignature;
  console.log(`   Valid: ${isValid ? '✅' : '❌'}`);
  
  return isValid;
}

// Main execution
async function main() {
  console.log('🧪 Webhook Testing Script');
  console.log('========================\n');
  
  try {
    // Test signature validation
    testSignatureValidation();
    
    // Make webhook request
    console.log('\n📡 Making webhook request...');
    const result = await makeWebhookRequest(WEBHOOK_URL, payload, SECRET);
    
    console.log('\n🎉 Test completed successfully!');
    process.exit(0);
    
  } catch (error) {
    console.error(`\n💥 Test failed: ${error.message}`);
    process.exit(1);
  }
}

// Handle process signals
process.on('SIGINT', () => {
  console.log('\n⏹️  Test interrupted by user');
  process.exit(1);
});

process.on('uncaughtException', (error) => {
  console.error(`\n💥 Uncaught exception: ${error.message}`);
  process.exit(1);
});

// Run the test
main();

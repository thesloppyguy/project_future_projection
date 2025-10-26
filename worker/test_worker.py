#!/usr/bin/env python3
"""
Test script for Python Worker Service

This script tests the worker endpoints and functionality.
"""

import asyncio
import json
import time
import requests
from typing import Dict, Any

# Configuration
WORKER_URL = "http://localhost:8000"
BACKEND_URL = "http://localhost:3000"

def test_health_check():
    """Test worker health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{WORKER_URL}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data['status']}")
            print(f"   Worker ID: {health_data['worker_id']}")
            print(f"   Active jobs: {health_data['active_jobs']}")
            print(f"   Memory usage: {health_data['memory_usage']:.1f}%")
            print(f"   CPU usage: {health_data['cpu_usage']:.1f}%")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_metrics():
    """Test metrics endpoint"""
    print("\n📊 Testing metrics endpoint...")
    try:
        response = requests.get(f"{WORKER_URL}/metrics", timeout=10)
        if response.status_code == 200:
            print("✅ Metrics endpoint accessible")
            return True
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metrics endpoint error: {e}")
        return False

def test_training_job():
    """Test model training job"""
    print("\n🤖 Testing training job...")
    
    training_data = {
        "job_id": f"test_training_{int(time.time())}",
        "organization_id": "test_org_123",
        "model_type": "classification",
        "training_data": {
            "dataSource": "test_data.csv",
            "features": ["feature1", "feature2", "feature3"],
            "target": "label",
            "testSize": 0.2
        },
        "hyperparameters": {
            "max_depth": 10,
            "n_estimators": 100,
            "learning_rate": 0.1
        }
    }
    
    try:
        # Start training job
        response = requests.post(
            f"{WORKER_URL}/api/training/start",
            json=training_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result["job_id"]
            print(f"✅ Training job started: {job_id}")
            
            # Monitor job status
            for i in range(10):  # Check for up to 10 iterations
                time.sleep(2)
                
                status_response = requests.get(
                    f"{WORKER_URL}/api/jobs/{job_id}/status",
                    timeout=10
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"   Status: {status_data['status']}, Progress: {status_data.get('progress', 0):.1%}")
                    
                    if status_data['status'] in ['completed', 'failed']:
                        if status_data['status'] == 'completed':
                            print(f"✅ Training completed successfully!")
                            print(f"   Model ID: {status_data['result']['model_id']}")
                            print(f"   Accuracy: {status_data['result']['accuracy']:.3f}")
                        else:
                            print(f"❌ Training failed: {status_data.get('error', 'Unknown error')}")
                        break
                else:
                    print(f"❌ Failed to get job status: {status_response.status_code}")
                    break
            else:
                print("⏰ Training job still running after 20 seconds")
            
            return True
        else:
            print(f"❌ Failed to start training job: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Training job error: {e}")
        return False

def test_prediction_job():
    """Test model prediction job"""
    print("\n🔮 Testing prediction job...")
    
    prediction_data = {
        "job_id": f"test_prediction_{int(time.time())}",
        "organization_id": "test_org_123",
        "model_id": "model_test_123",
        "input_data": {
            "features": [1.0, 2.0, 3.0],
            "batch_size": 1
        }
    }
    
    try:
        # Start prediction job
        response = requests.post(
            f"{WORKER_URL}/api/prediction/start",
            json=prediction_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result["job_id"]
            print(f"✅ Prediction job started: {job_id}")
            
            # Monitor job status
            for i in range(5):  # Check for up to 5 iterations
                time.sleep(1)
                
                status_response = requests.get(
                    f"{WORKER_URL}/api/jobs/{job_id}/status",
                    timeout=10
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"   Status: {status_data['status']}, Progress: {status_data.get('progress', 0):.1%}")
                    
                    if status_data['status'] in ['completed', 'failed']:
                        if status_data['status'] == 'completed':
                            print(f"✅ Prediction completed successfully!")
                            print(f"   Predictions: {status_data['result']['predictions']}")
                            print(f"   Confidence: {status_data['result']['confidence']:.3f}")
                        else:
                            print(f"❌ Prediction failed: {status_data.get('error', 'Unknown error')}")
                        break
                else:
                    print(f"❌ Failed to get job status: {status_response.status_code}")
                    break
            else:
                print("⏰ Prediction job still running after 5 seconds")
            
            return True
        else:
            print(f"❌ Failed to start prediction job: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction job error: {e}")
        return False

def test_job_listing():
    """Test job listing endpoint"""
    print("\n📋 Testing job listing...")
    try:
        response = requests.get(f"{WORKER_URL}/api/jobs", timeout=10)
        if response.status_code == 200:
            jobs = response.json()
            print(f"✅ Job listing successful")
            print(f"   Training jobs: {len(jobs['training'])}")
            print(f"   Prediction jobs: {len(jobs['prediction'])}")
            print(f"   Total jobs: {jobs['total']}")
            return True
        else:
            print(f"❌ Job listing failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Job listing error: {e}")
        return False

def test_internal_message():
    """Test internal message endpoint"""
    print("\n💬 Testing internal message...")
    
    message_data = {
        "message_type": "test_message",
        "payload": {
            "test": True,
            "timestamp": time.time()
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "worker_id": "test_client"
    }
    
    try:
        response = requests.post(
            f"{WORKER_URL}/api/internal/message",
            json=message_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Internal message sent successfully")
            print(f"   Response: {result['message']}")
            return True
        else:
            print(f"❌ Internal message failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Internal message error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Python Worker Service Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Metrics", test_metrics),
        ("Job Listing", test_job_listing),
        ("Internal Message", test_internal_message),
        ("Training Job", test_training_job),
        ("Prediction Job", test_prediction_job),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())

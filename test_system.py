#!/usr/bin/env python3
"""
NASA Exoplanet Classification System - Test Script
Simple test to verify the system is working correctly.
"""

import requests
import json
import sys
import time

def test_health_check(base_url="http://localhost:5000"):
    """Test the health check endpoint."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False


def test_login_endpoint(base_url="http://localhost:5000"):
    """Test the login endpoint."""
    try:
        # Test login page accessibility
        response = requests.get(f"{base_url}/login", timeout=5)
        if response.status_code == 200:
            print("✅ Login page accessible")
            
            # Test successful login
            login_data = {
                'username': 'user',
                'password': '123'
            }
            
            session = requests.Session()
            login_response = session.post(f"{base_url}/login", data=login_data, timeout=5)
            
            if login_response.status_code == 302:  # Redirect after successful login
                print("✅ Login successful with correct credentials")
                return True, session
            else:
                print(f"❌ Login failed: {login_response.status_code}")
                return False, None
        else:
            print(f"❌ Login page not accessible: {response.status_code}")
            return False, None
    except requests.exceptions.RequestException as e:
        print(f"❌ Login test error: {e}")
        return False, None


def test_protected_endpoints(base_url="http://localhost:5000", session=None):
    """Test protected endpoints with authentication."""
    if not session:
        print("⚠️  No session provided, skipping protected endpoint tests")
        return False
    
    try:
        # Test main page
        response = session.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Main page accessible with authentication")
        else:
            print(f"❌ Main page not accessible: {response.status_code}")
            return False
        
        # Test stats endpoint
        response = session.get(f"{base_url}/stats?dataset=k2", timeout=5)
        if response.status_code == 200:
            print("✅ Stats endpoint accessible with authentication")
        else:
            print(f"❌ Stats endpoint not accessible: {response.status_code}")
            return False
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Protected endpoint test error: {e}")
        return False


def test_logout_endpoint(base_url="http://localhost:5000", session=None):
    """Test the logout endpoint."""
    if not session:
        print("⚠️  No session provided, skipping logout test")
        return False
    
    try:
        response = session.get(f"{base_url}/logout", timeout=5)
        if response.status_code == 302:  # Redirect after logout
            print("✅ Logout successful")
            return True
        else:
            print(f"❌ Logout failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Logout test error: {e}")
        return False

def test_stats_endpoint(base_url="http://localhost:5000"):
    """Test the stats endpoint for all datasets."""
    datasets = ['k2', 'tess', 'koi']
    results = []
    
    for dataset in datasets:
        try:
            response = requests.get(f"{base_url}/stats?dataset={dataset}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Stats for {dataset.upper()}: Accuracy {data.get('accuracy', 0):.2%}")
                results.append(True)
            else:
                print(f"❌ Stats failed for {dataset}: {response.status_code}")
                results.append(False)
        except requests.exceptions.RequestException as e:
            print(f"❌ Stats error for {dataset}: {e}")
            results.append(False)
    
    return all(results)

def test_prediction_endpoint(base_url="http://localhost:5000"):
    """Test the prediction endpoint."""
    test_data = {
        "dataset": "k2",
        "pl_orbper": 1.7575,
        "pl_trandep": 0.0744,
        "st_teff": 4759
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            prediction = data.get('prediction', 'Unknown')
            confidence = data.get('confidence', 0)
            print(f"✅ Prediction successful: {prediction} (confidence: {confidence:.2%})")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_system_status(base_url="http://localhost:5000"):
    """Test the system status endpoint."""
    try:
        response = requests.get(f"{base_url}/system-status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models_loaded = data.get('models_loaded', 0)
            datasets = data.get('datasets', {})
            print(f"✅ System status: {models_loaded} models loaded")
            
            for dataset, info in datasets.items():
                status = "✅" if info.get('model_loaded', False) else "⚠️"
                print(f"   {status} {dataset.upper()}: {'Loaded' if info.get('model_loaded', False) else 'Not loaded'}")
            
            return True
        else:
            print(f"❌ System status failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ System status error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 NASA Exoplanet Classification System - Test Suite")
    print("=" * 60)
    
    base_url = "http://localhost:5000"
    
    # Wait a moment for the server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health_check, False),
        ("Login System", test_login_endpoint, False),
    ]
    
    results = []
    authenticated_session = None
    
    for test_name, test_func, requires_auth in tests:
        print(f"\n🧪 Testing {test_name}...")
        if test_name == "Login System":
            result, session = test_func(base_url)
            if result:
                authenticated_session = session
            results.append((test_name, result))
        else:
            result = test_func(base_url)
            results.append((test_name, result))
    
    # Test authenticated endpoints if login was successful
    if authenticated_session:
        auth_tests = [
            ("Protected Endpoints", test_protected_endpoints, True),
            ("Stats Endpoints", test_stats_endpoint, True),
            ("Prediction Endpoint", test_prediction_endpoint, True),
            ("Logout", test_logout_endpoint, True),
        ]
        
        for test_name, test_func, requires_auth in auth_tests:
            print(f"\n🧪 Testing {test_name}...")
            if test_name == "Protected Endpoints":
                result = test_func(base_url, authenticated_session)
            elif test_name == "Logout":
                result = test_func(base_url, authenticated_session)
            else:
                # For stats and prediction, we need to create a new session with auth
                result = test_func(base_url)
            results.append((test_name, result))
    else:
        print("\n⚠️  Skipping authenticated endpoint tests due to login failure")
        # Add placeholder results for missing tests
        results.extend([
            ("Protected Endpoints", False),
            ("Stats Endpoints", False),
            ("Prediction Endpoint", False),
            ("Logout", False),
        ])
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the server logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

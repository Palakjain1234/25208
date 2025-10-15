"""
test_api.py
Comprehensive testing for the FastAPI endpoints
"""

import requests
import json
import time
import pandas as pd
from typing import Dict, List

BASE_URL = "http://localhost:8000"

class APITester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        
    def print_step(self, message: str):
        print(f"\n{'='*50}")
        print(f"🔹 {message}")
        print(f"{'='*50}")
        
    def print_success(self, message: str):
        print(f"✅ {message}")
        
    def print_error(self, message: str):
        print(f"❌ {message}")
        
    def print_info(self, message: str):
        print(f"ℹ️  {message}")
        
    def test_health(self) -> bool:
        """Test health endpoint"""
        self.print_step("Testing Health Endpoint")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Health check passed - Status: {data['status']}, Pipeline: {data['pipeline']}")
                return True
            else:
                self.print_error(f"Health check failed with status {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"Health check failed: {e}")
            return False
    
    def test_root(self) -> bool:
        """Test root endpoint"""
        self.print_step("Testing Root Endpoint")
        try:
            response = self.session.get(self.base_url)
            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Root endpoint: {data['message']}")
                return True
            else:
                self.print_error(f"Root endpoint failed with status {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"Root endpoint failed: {e}")
            return False
    
    def test_get_plans(self) -> bool:
        """Test getting available plans"""
        self.print_step("Testing Get Plans Endpoint")
        try:
            response = self.session.get(f"{self.base_url}/plans?limit=5")
            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Found {data['total_plans']} total plans, showing {len(data['plans'])}")
                for plan in data['plans']:
                    print(f"   📋 {plan['plan_id']}: {plan['customer_name']} -> {plan['destination']} ({plan['planned_qty_t']} tons)")
                return True
            else:
                self.print_error(f"Get plans failed with status {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"Get plans failed: {e}")
            return False
    
    def test_plan_details(self) -> bool:
        """Test getting plan details"""
        self.print_step("Testing Plan Details Endpoint")
        try:
            # First get some plan IDs
            plans_response = self.session.get(f"{self.base_url}/plans?limit=3")
            if plans_response.status_code != 200:
                self.print_error("Could not get plans for details test")
                return False
                
            plans = plans_response.json()['plans']
            if not plans:
                self.print_error("No plans available for testing")
                return False
                
            for plan in plans[:2]:  # Test first 2 plans
                plan_id = plan['plan_id']
                response = self.session.get(f"{self.base_url}/plan/{plan_id}")
                if response.status_code == 200:
                    data = response.json()
                    self.print_success(f"Plan {plan_id}: {data['customer_name']} -> {data['destination']}")
                    print(f"      Quantity: {data['planned_qty_t']} tons, Priority: {data['priority_score']}")
                else:
                    self.print_error(f"Failed to get details for plan {plan_id}")
                    return False
            return True
        except Exception as e:
            self.print_error(f"Plan details test failed: {e}")
            return False
    
    def test_optimization_small(self) -> bool:
        """Test optimization with a small subset"""
        self.print_step("Testing Optimization with Small Dataset")
        try:
            # Get a few plan IDs for testing
            plans_response = self.session.get(f"{self.base_url}/plans?limit=10")
            if plans_response.status_code != 200:
                self.print_error("Could not get plans for optimization test")
                return False
                
            plans = plans_response.json()['plans']
            test_plan_ids = [plan['plan_id'] for plan in plans[:5]]  # Use first 5 plans
            
            self.print_info(f"Testing with {len(test_plan_ids)} plans: {test_plan_ids}")
            
            # Start optimization
            data = {
                "use_sample": False,
                "plan_ids": test_plan_ids
            }
            
            response = self.session.post(f"{self.base_url}/optimize", json=data)
            if response.status_code != 200:
                self.print_error(f"Optimization start failed with status {response.status_code}")
                return False
                
            result = response.json()
            job_id = result["job_id"]
            self.print_success(f"Optimization job started: {job_id}")
            
            # Poll for completion
            return self._poll_job_completion(job_id, timeout=60)  # 60 second timeout for small test
            
        except Exception as e:
            self.print_error(f"Small optimization test failed: {e}")
            return False
    
    def test_optimization_full(self) -> bool:
        """Test optimization with full dataset"""
        self.print_step("Testing Optimization with Full Dataset")
        try:
            # Start optimization with all plans
            data = {
                "use_sample": True,
                "plan_ids": []  # Empty = use all
            }
            
            response = self.session.post(f"{self.base_url}/optimize", json=data)
            if response.status_code != 200:
                self.print_error(f"Full optimization start failed with status {response.status_code}")
                return False
                
            result = response.json()
            job_id = result["job_id"]
            self.print_success(f"Full optimization job started: {job_id}")
            self.print_info("This may take a few minutes for the full dataset...")
            
            # Poll for completion with longer timeout
            return self._poll_job_completion(job_id, timeout=300)  # 5 minute timeout for full test
            
        except Exception as e:
            self.print_error(f"Full optimization test failed: {e}")
            return False
    
    def _poll_job_completion(self, job_id: str, timeout: int = 120) -> bool:
        """Poll job status until completion or timeout"""
        start_time = time.time()
        last_progress = -1
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.base_url}/job/{job_id}")
                if response.status_code != 200:
                    self.print_error(f"Failed to get job status: {response.status_code}")
                    return False
                    
                status = response.json()
                current_progress = status["progress"] * 100
                
                # Only print progress if it changed
                if current_progress != last_progress:
                    print(f"   📊 Progress: {current_progress:.1f}% - {status['message']}")
                    last_progress = current_progress
                
                if status["status"] == "completed":
                    self.print_success("Optimization completed successfully!")
                    if status.get("results"):
                        summary = status["results"]
                        print(f"   📈 Summary:")
                        print(f"      Total Orders: {summary['total_orders']}")
                        print(f"      Rail Orders: {summary['rail_orders']} ({summary['rail_orders_percentage']}%)")
                        print(f"      Rail Tonnage: {summary['rail_tonnage']} tons ({summary['rail_tonnage_percentage']}%)")
                        print(f"      Total Cost: ₹{summary['total_cost']:,.2f}")
                    
                    # Get detailed results
                    self._test_results_endpoint(job_id)
                    return True
                    
                elif status["status"] == "failed":
                    self.print_error(f"Optimization failed: {status['message']}")
                    return False
                    
                time.sleep(2)  # Wait 2 seconds between checks
                
            except Exception as e:
                self.print_error(f"Error polling job status: {e}")
                return False
        
        self.print_error(f"Optimization timed out after {timeout} seconds")
        return False
    
    def _test_results_endpoint(self, job_id: str):
        """Test the results endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/results/{job_id}?limit=10")
            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Results retrieved: {len(data['results'])} sample records")
                
                # Print sample results
                print(f"\n   🚂 Sample Optimization Results:")
                rail_count = 0
                for result in data['results']:
                    mode_icon = "🚆" if result['optimized_mode'] == 'Rail' else "🚛"
                    print(f"      {mode_icon} {result['plan_id']}: {result['optimized_mode']} - {result['q_rail_tons']} tons - ₹{result['optimized_total_cost']:,.2f}")
                    if result['optimized_mode'] == 'Rail':
                        rail_count += 1
                
                print(f"\n   📊 In this sample: {rail_count} rail, {len(data['results']) - rail_count} road")
                
                # Test pagination
                response_page2 = self.session.get(f"{self.base_url}/results/{job_id}?limit=5&offset=5")
                if response_page2.status_code == 200:
                    page2_data = response_page2.json()
                    self.print_info(f"Pagination works: page 2 has {len(page2_data['results'])} records")
                    
            else:
                self.print_error(f"Failed to get results: {response.status_code}")
        except Exception as e:
            self.print_error(f"Results endpoint test failed: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        self.print_step("STARTING COMPREHENSIVE API TESTS")
        
        tests = [
            ("Health Check", self.test_health),
            ("Root Endpoint", self.test_root),
            ("Get Plans", self.test_get_plans),
            ("Plan Details", self.test_plan_details),
            ("Small Optimization", self.test_optimization_small),
            ("Full Optimization", self.test_optimization_full),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                self.print_error(f"Test {test_name} crashed: {e}")
                results.append((test_name, False))
            
            time.sleep(1)  # Brief pause between tests
        
        # Print summary
        self.print_step("TEST SUMMARY")
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        print(f"📊 Results: {passed}/{total} tests passed")
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status} {test_name}")
        
        if passed == total:
            self.print_success("🎉 All tests passed! API is ready for integration.")
        else:
            self.print_error("⚠️  Some tests failed. Check the logs above.")

def main():
    """Main test function"""
    tester = APITester()
    
    # Check if server is running
    try:
        tester.session.get(BASE_URL, timeout=5)
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running!")
        print("💡 Please start the server first with:")
        print("   python api.py")
        print("   or")
        print("   uvicorn api:app --host 0.0.0.0 --port 8000 --reload")
        return
    
    # Run tests
    tester.run_all_tests()

if __name__ == "__main__":
    main()
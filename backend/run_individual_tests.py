"""
Individual API Endpoint Tester with Per-Test Timeout (max 30s)
"""
import sys
import time
import asyncio
import traceback

def log(msg):
    print(msg, flush=True)

async def test_endpoint(name, func, timeout=30):
    log(f"\n--- Testing: {name} (timeout {timeout}s) ---")
    start = time.time()
    try:
        if asyncio.iscoroutinefunction(func):
            res = await asyncio.wait_for(func(), timeout=timeout)
        else:
            # Run sync function in thread pool or direct
            res = func()
        elapsed = time.time() - start
        log(f"  ✓ PASSED: {name} in {elapsed:.2f}s")
        return {"name": name, "status": "PASSED", "duration": elapsed, "error": None, "result": res}
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        log(f"  ✗ FAILED: {name} TIMED OUT after {elapsed:.2f}s")
        return {"name": name, "status": "FAILED", "duration": elapsed, "error": f"Timed out after {timeout}s", "result": None}
    except Exception as e:
        elapsed = time.time() - start
        err = f"{type(e).__name__}: {str(e)}"
        log(f"  ✗ FAILED: {name} error: {err}")
        traceback.print_exc(file=sys.stdout)
        return {"name": name, "status": "FAILED", "duration": elapsed, "error": err, "result": None}

async def main():
    log("==================================================")
    log("  RIDEGUARD AI - INDIVIDUAL ENDPOINT TEST RUNNER  ")
    log("==================================================")

    import server

    # Pre-test: ensure state
    log("\n[0] Initializing Server State (ensure_state)...")
    t0 = time.time()
    server.ensure_state()
    log(f"  ✓ State initialized in {time.time()-t0:.2f}s. Model: {server.STATE.get('best_model_name')}")

    results = []

    # 1. GET /api/health
    results.append(await test_endpoint("GET /api/health", lambda: server.get_health(), timeout=5))

    # 2. GET /api/dashboard
    results.append(await test_endpoint("GET /api/dashboard", lambda: server.get_dashboard(), timeout=5))

    # 3. GET /api/overview
    results.append(await test_endpoint("GET /api/overview", lambda: server.get_overview(), timeout=5))

    # 4. GET /api/analytics
    results.append(await test_endpoint("GET /api/analytics", lambda: server.get_analytics(), timeout=5))

    # 5. GET /api/models
    results.append(await test_endpoint("GET /api/models", lambda: server.get_model_benchmarks(), timeout=5))

    # 6. POST /api/predict
    payload = {
        "Vehicle_Type": "Prime Sedan",
        "Pickup_Location": "Airport Terminal 1",
        "Drop_Location": "Downtown Arts District",
        "Ride_Distance": 18.5,
        "Booking_Value": 450.0,
        "Hour": 19,
        "Driver_Arrival_Time": 12,
        "Waiting_Time": 6,
        "Payment_Method": "UPI / Digital Wallet",
        "Customer_Rating": 4.7
    }
    results.append(await test_endpoint("POST /api/predict", lambda: server.predict_single(payload), timeout=10))

    # 7. GET /api/recommendations
    results.append(await test_endpoint("GET /api/recommendations", lambda: server.get_recommendations(), timeout=5))

    # 8. GET /api/explainability
    results.append(await test_endpoint("GET /api/explainability", lambda: server.get_global_explainability(samples=20), timeout=15))

    # 9. GET /api/dataset/info
    results.append(await test_endpoint("GET /api/dataset/info", lambda: server.get_dataset_info(), timeout=5))

    # 10. GET /api/predict/schema
    results.append(await test_endpoint("GET /api/predict/schema", lambda: server.get_prediction_schema(), timeout=5))

    # Summary report
    log("\n" + "="*50)
    log("  TEST SUMMARY REPORT")
    log("="*50)
    passed = [r for r in results if r["status"] == "PASSED"]
    failed = [r for r in results if r["status"] == "FAILED"]
    
    log(f"Total Tests: {len(results)}")
    log(f"Passed: {len(passed)}")
    log(f"Failed: {len(failed)}")
    for r in results:
        status_sym = "✓" if r["status"] == "PASSED" else "✗"
        log(f" {status_sym} {r['name']:<25} : {r['status']:<7} ({r['duration']:.2f}s) {r['error'] or ''}")

if __name__ == "__main__":
    asyncio.run(main())

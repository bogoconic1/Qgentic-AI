"""Unit tests for dynamic resource pool allocation (CPU cores and MIG instances)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import time


class TestDynamicResourceAllocation(unittest.TestCase):
    """Test dynamic CPU and MIG resource pool allocation with 5 processes and 3 workers."""

    def test_five_processes_three_workers_dynamic_allocation(self):
        """
        Test scenario: 5 processes, 3 workers (CPU/MIG pairs)
        - Process 0, 1, 2 start immediately with CPU 0, 1, 2 and MIG 0, 1, 2
        - Process 2 finishes first (fastest - 0.1s)
        - Process 3 should get CPU 2 + MIG 2 (from completed Process 2)
        - Process 0 finishes second (0.3s)
        - Process 4 should get CPU 0 + MIG 0 (from completed Process 0)
        """
        # Setup: 3 CPU core ranges and 3 MIG instances
        cpu_pool = Queue()
        cpu_pool.put([0, 1, 2])      # CPU 0
        cpu_pool.put([3, 4, 5])      # CPU 1
        cpu_pool.put([6, 7, 8])      # CPU 2

        mig_pool = Queue()
        mig_pool.put("MIG-0")
        mig_pool.put("MIG-1")
        mig_pool.put("MIG-2")

        # Track assignments
        process_assignments = {}
        start_time = time.time()

        # Process durations: 2 > 0 > 1 (Process 2 fastest)
        durations = {
            0: 0.3,  # Medium speed
            1: 0.5,  # Slowest
            2: 0.1,  # Fastest
            3: 0.2,  # Waits for Process 2
            4: 0.2,  # Waits for Process 0
        }

        def _run_developer_baseline(slug, iteration_suffix, model_name, now_recommendations,
                                     later_recommendations, key, cpu_core_pool=None, mig_pool=None):
            """Mock implementation that just does time.sleep() and tracks resource allocation."""
            # Extract process ID from iteration_suffix (e.g., "1_1" -> 0)
            process_id = int(iteration_suffix.split('_')[1]) - 1

            # Acquire resources
            cpu_core_range = cpu_core_pool.get() if cpu_core_pool else None
            mig_instance = mig_pool.get() if mig_pool else None

            # Record assignment
            process_assignments[process_id] = {
                "cpu": cpu_core_range,
                "mig": mig_instance,
                "start_time": time.time() - start_time
            }

            try:
                # Simulate work with specific duration
                time.sleep(durations[process_id])
                return key, 0.85, f"code_{process_id}.py", []
            finally:
                # Return resources to pools
                if cpu_core_pool and cpu_core_range is not None:
                    cpu_core_pool.put(cpu_core_range)
                if mig_pool and mig_instance is not None:
                    mig_pool.put(mig_instance)

        # Create tasks like in orchestrator
        tasks = []
        for i in range(5):
            task = {
                "slug": "test-comp",
                "iteration_suffix": f"1_{i+1}",
                "model_name": f"model-{i}",
                "now_recommendations": {},
                "later_recommendations": {},
                "key": f"key-{i}",
                "cpu_core_pool": cpu_pool,
                "mig_pool": mig_pool
            }
            tasks.append(task)

        # Execute with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_run_developer_baseline, **task) for task in tasks]
            results = [f.result() for f in futures]

        # Verify assignments
        print("\n=== Process Assignments ===")
        for pid in sorted(process_assignments.keys()):
            assignment = process_assignments[pid]
            print(f"Process {pid}: CPU {assignment['cpu']} + {assignment['mig']} "
                  f"(started at {assignment['start_time']:.2f}s)")

        # Assertions
        # Process 3 should get resources from Process 2 (fastest to complete)
        self.assertEqual(process_assignments[3]["cpu"], process_assignments[2]["cpu"],
                        "Process 3 should get CPU from completed Process 2")
        self.assertEqual(process_assignments[3]["mig"], process_assignments[2]["mig"],
                        "Process 3 should get MIG from completed Process 2")

        # Process 4 should get resources from Process 0 (second to complete)
        self.assertEqual(process_assignments[4]["cpu"], process_assignments[0]["cpu"],
                        "Process 4 should get CPU from completed Process 0")
        self.assertEqual(process_assignments[4]["mig"], process_assignments[0]["mig"],
                        "Process 4 should get MIG from completed Process 0")

        # Verify all resources returned to pool
        self.assertEqual(cpu_pool.qsize(), 3, "All 3 CPU ranges should be back in pool")
        self.assertEqual(mig_pool.qsize(), 3, "All 3 MIG instances should be back in pool")


if __name__ == "__main__":
    unittest.main(verbosity=2)

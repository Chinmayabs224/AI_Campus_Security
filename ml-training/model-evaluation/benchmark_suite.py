#!/usr/bin/env python3
"""
Comprehensive Benchmarking Suite for Security YOLO Models
Standardized benchmarks for evaluating campus security models.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import cv2
import time
from ultralytics import YOLO
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityModelBenchmark:
    """Comprehensive benchmarking suite for security YOLO models."""
    
    def __init__(self, model_path: str, config_path: str = "../config/dataset_config.yaml"):
        """Initialize the benchmark suite."""
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Load model
        self.model = YOLO(str(self.model_path))
        
        # Set up paths
        self.yolo_data_dir = Path(self.config['dataset']['yolo_data_dir'])
        self.benchmark_dir = Path("../benchmark_results")
        self.benchmark_dir.mkdir(exist_ok=True)
        
        # Benchmark configuration
        self.benchmark_config = {
            'accuracy_tests': {
                'confidence_thresholds': [0.1, 0.25, 0.5, 0.75, 0.9],
                'iou_thresholds': [0.3, 0.5, 0.7, 0.9],
                'test_iterations': 3
            },
            'performance_tests': {
                'input_sizes': [320, 416, 640, 832],
                'batch_sizes': [1, 2, 4, 8, 16],
                'warmup_iterations': 10,
                'test_iterations': 50
            },
            'stress_tests': {
                'duration_minutes': 10,
                'concurrent_threads': [1, 2, 4, 8],
                'memory_limit_gb': 8
            },
            'robustness_tests': {
                'noise_levels': [0, 10, 25, 50, 100],
                'brightness_factors': [0.3, 0.5, 1.0, 1.5, 2.0],
                'blur_kernels': [0, 3, 5, 7, 9]
            }
        }
        
        logger.info(f"SecurityModelBenchmark initialized for model: {self.model_path}")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_full_benchmark_suite(self) -> Dict:
        """Run the complete benchmark suite."""
        logger.info("Starting comprehensive benchmark suite...")
        
        benchmark_results = {
            'benchmark_info': {
                'model_path': str(self.model_path),
                'benchmark_timestamp': datetime.now().isoformat(),
                'system_info': self._get_system_info(),
                'model_info': self._get_model_info()
            },
            'accuracy_benchmarks': {},
            'performance_benchmarks': {},
            'stress_benchmarks': {},
            'robustness_benchmarks': {},
            'security_benchmarks': {},
            'benchmark_summary': {}
        }
        
        try:
            # 1. Accuracy Benchmarks
            logger.info("Running accuracy benchmarks...")
            benchmark_results['accuracy_benchmarks'] = self._run_accuracy_benchmarks()
            
            # 2. Performance Benchmarks
            logger.info("Running performance benchmarks...")
            benchmark_results['performance_benchmarks'] = self._run_performance_benchmarks()
            
            # 3. Stress Benchmarks
            logger.info("Running stress benchmarks...")
            benchmark_results['stress_benchmarks'] = self._run_stress_benchmarks()
            
            # 4. Robustness Benchmarks
            logger.info("Running robustness benchmarks...")
            benchmark_results['robustness_benchmarks'] = self._run_robustness_benchmarks()
            
            # 5. Security-Specific Benchmarks
            logger.info("Running security-specific benchmarks...")
            benchmark_results['security_benchmarks'] = self._run_security_benchmarks()
            
            # 6. Generate Summary
            benchmark_results['benchmark_summary'] = self._generate_benchmark_summary(benchmark_results)
            
            # Save results
            self._save_benchmark_results(benchmark_results)
            
            logger.info("Benchmark suite completed successfully!")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Benchmark suite failed: {e}")
            benchmark_results['benchmark_summary'] = {'status': 'failed', 'error': str(e)}
            return benchmark_results
    
    def _get_system_info(self) -> Dict:
        """Get system information for benchmarking context."""
        try:
            system_info = {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'platform': sys.platform,
                'python_version': sys.version,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available()
            }
            
            if torch.cuda.is_available():
                system_info['cuda_device_count'] = torch.cuda.device_count()
                system_info['cuda_device_name'] = torch.cuda.get_device_name(0)
                system_info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            return system_info
        except Exception as e:
            return {'error': str(e)}
    
    def _get_model_info(self) -> Dict:
        """Get model information."""
        try:
            model_info = {
                'model_size_mb': self.model_path.stat().st_size / (1024 * 1024),
                'parameters': sum(p.numel() for p in self.model.model.parameters()),
                'model_type': self._determine_model_type(),
                'input_size': 640  # Default YOLO input size
            }
            return model_info
        except Exception as e:
            return {'error': str(e)}
    
    def _determine_model_type(self) -> str:
        """Determine YOLO model type."""
        model_name = self.model_path.name.lower()
        for size in ['n', 's', 'm', 'l', 'x']:
            if f'yolov8{size}' in model_name:
                return f'yolov8{size}'
        return 'unknown'
    
    def _run_accuracy_benchmarks(self) -> Dict:
        """Run comprehensive accuracy benchmarks."""
        accuracy_benchmarks = {}
        
        # Standard accuracy metrics
        accuracy_benchmarks['standard_metrics'] = self._benchmark_standard_accuracy()
        
        # Confidence threshold analysis
        accuracy_benchmarks['confidence_analysis'] = self._benchmark_confidence_thresholds()
        
        # IoU threshold analysis
        accuracy_benchmarks['iou_analysis'] = self._benchmark_iou_thresholds()
        
        # Per-class accuracy
        accuracy_benchmarks['per_class_accuracy'] = self._benchmark_per_class_accuracy()
        
        return accuracy_benchmarks
    
    def _benchmark_standard_accuracy(self) -> Dict:
        """Benchmark standard accuracy metrics."""
        try:
            results = self.model.val(
                data=str(self.yolo_data_dir / "dataset.yaml"),
                split='test',
                verbose=False
            )
            
            return {
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / 
                           (float(results.box.mp) + float(results.box.mr)) 
                           if (float(results.box.mp) + float(results.box.mr)) > 0 else 0
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _benchmark_confidence_thresholds(self) -> Dict:
        """Benchmark performance at different confidence thresholds."""
        confidence_results = {}
        
        for conf_threshold in self.benchmark_config['accuracy_tests']['confidence_thresholds']:
            try:
                results = self.model.val(
                    data=str(self.yolo_data_dir / "dataset.yaml"),
                    split='test',
                    conf=conf_threshold,
                    verbose=False
                )
                
                confidence_results[str(conf_threshold)] = {
                    'mAP50': float(results.box.map50),
                    'precision': float(results.box.mp),
                    'recall': float(results.box.mr),
                    'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / 
                               (float(results.box.mp) + float(results.box.mr)) 
                               if (float(results.box.mp) + float(results.box.mr)) > 0 else 0
                }
            except Exception as e:
                confidence_results[str(conf_threshold)] = {'error': str(e)}
        
        return confidence_results
    
    def _benchmark_iou_thresholds(self) -> Dict:
        """Benchmark performance at different IoU thresholds."""
        iou_results = {}
        
        for iou_threshold in self.benchmark_config['accuracy_tests']['iou_thresholds']:
            try:
                results = self.model.val(
                    data=str(self.yolo_data_dir / "dataset.yaml"),
                    split='test',
                    iou=iou_threshold,
                    verbose=False
                )
                
                iou_results[str(iou_threshold)] = {
                    'mAP50': float(results.box.map50),
                    'precision': float(results.box.mp),
                    'recall': float(results.box.mr)
                }
            except Exception as e:
                iou_results[str(iou_threshold)] = {'error': str(e)}
        
        return iou_results
    
    def _benchmark_per_class_accuracy(self) -> Dict:
        """Benchmark per-class accuracy metrics."""
        try:
            results = self.model.val(
                data=str(self.yolo_data_dir / "dataset.yaml"),
                split='test',
                verbose=False
            )
            
            per_class_results = {}
            class_names = list(self.config['classes'].keys())
            
            if hasattr(results.box, 'ap_class_index'):
                for i, class_idx in enumerate(results.box.ap_class_index):
                    if class_idx < len(class_names):
                        class_name = class_names[class_idx]
                        per_class_results[class_name] = {
                            'ap50': float(results.box.ap50[i]),
                            'ap50_95': float(results.box.ap[i]),
                            'precision': float(results.box.p[i]) if hasattr(results.box, 'p') else 0,
                            'recall': float(results.box.r[i]) if hasattr(results.box, 'r') else 0
                        }
            
            return per_class_results
        except Exception as e:
            return {'error': str(e)} 
   
    def _run_performance_benchmarks(self) -> Dict:
        """Run comprehensive performance benchmarks."""
        performance_benchmarks = {}
        
        # Inference speed benchmarks
        performance_benchmarks['inference_speed'] = self._benchmark_inference_speed()
        
        # Batch processing benchmarks
        performance_benchmarks['batch_processing'] = self._benchmark_batch_processing()
        
        # Memory usage benchmarks
        performance_benchmarks['memory_usage'] = self._benchmark_memory_usage()
        
        # Throughput benchmarks
        performance_benchmarks['throughput'] = self._benchmark_throughput()
        
        return performance_benchmarks
    
    def _benchmark_inference_speed(self) -> Dict:
        """Benchmark inference speed across different input sizes."""
        speed_results = {}
        
        for input_size in self.benchmark_config['performance_tests']['input_sizes']:
            try:
                # Create test input
                test_input = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
                
                # Warm up
                for _ in range(self.benchmark_config['performance_tests']['warmup_iterations']):
                    _ = self.model.predict(test_input, verbose=False)
                
                # Benchmark inference time
                times = []
                for _ in range(self.benchmark_config['performance_tests']['test_iterations']):
                    start_time = time.perf_counter()
                    _ = self.model.predict(test_input, verbose=False)
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                
                speed_results[f'input_{input_size}'] = {
                    'mean_time_ms': np.mean(times),
                    'std_time_ms': np.std(times),
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times),
                    'p95_time_ms': np.percentile(times, 95),
                    'p99_time_ms': np.percentile(times, 99),
                    'fps': 1000 / np.mean(times),
                    'input_size': input_size
                }
                
            except Exception as e:
                speed_results[f'input_{input_size}'] = {'error': str(e)}
        
        return speed_results
    
    def _benchmark_batch_processing(self) -> Dict:
        """Benchmark batch processing performance."""
        batch_results = {}
        
        for batch_size in self.benchmark_config['performance_tests']['batch_sizes']:
            try:
                # Create batch input
                batch_input = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) 
                              for _ in range(batch_size)]
                
                # Warm up
                for _ in range(5):
                    _ = self.model.predict(batch_input, verbose=False)
                
                # Benchmark batch processing
                times = []
                for _ in range(10):
                    start_time = time.perf_counter()
                    _ = self.model.predict(batch_input, verbose=False)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                per_image_time = avg_time / batch_size
                
                batch_results[f'batch_{batch_size}'] = {
                    'total_time_s': avg_time,
                    'per_image_time_s': per_image_time,
                    'per_image_time_ms': per_image_time * 1000,
                    'throughput_fps': batch_size / avg_time,
                    'batch_size': batch_size,
                    'efficiency_gain': (1.0 / per_image_time) / (1000 / batch_results.get('batch_1', {}).get('per_image_time_ms', 1000)) if 'batch_1' in batch_results else 1.0
                }
                
            except Exception as e:
                batch_results[f'batch_{batch_size}'] = {'error': str(e)}
        
        return batch_results
    
    def _benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage during inference."""
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            
            # Get baseline memory
            gc.collect()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_usage = {
                'baseline_memory_mb': baseline_memory,
                'peak_memory_mb': baseline_memory,
                'memory_growth_mb': 0,
                'memory_stable': True
            }
            
            # Test memory usage with different scenarios
            test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            memory_samples = []
            
            # Single inference memory usage
            for i in range(20):
                _ = self.model.predict(test_input, verbose=False)
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                if i % 5 == 0:
                    gc.collect()
            
            peak_memory = max(memory_samples)
            final_memory = memory_samples[-1]
            memory_growth = final_memory - baseline_memory
            
            # Check memory stability (growth should be minimal)
            memory_stable = memory_growth < 100  # Less than 100MB growth
            
            memory_usage.update({
                'peak_memory_mb': peak_memory,
                'final_memory_mb': final_memory,
                'memory_growth_mb': memory_growth,
                'memory_stable': memory_stable,
                'memory_samples': memory_samples[-10:]  # Last 10 samples
            })
            
            # Batch memory usage
            try:
                large_batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(8)]
                gc.collect()
                pre_batch_memory = process.memory_info().rss / 1024 / 1024
                
                _ = self.model.predict(large_batch, verbose=False)
                
                post_batch_memory = process.memory_info().rss / 1024 / 1024
                batch_memory_usage = post_batch_memory - pre_batch_memory
                
                memory_usage['batch_memory_usage_mb'] = batch_memory_usage
                
            except Exception as e:
                memory_usage['batch_memory_error'] = str(e)
            
            return memory_usage
            
        except ImportError:
            return {'error': 'psutil not available for memory benchmarking'}
        except Exception as e:
            return {'error': str(e)}
    
    def _benchmark_throughput(self) -> Dict:
        """Benchmark model throughput under different conditions."""
        throughput_results = {}
        
        # Sequential throughput
        try:
            test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            start_time = time.perf_counter()
            for _ in range(100):
                _ = self.model.predict(test_input, verbose=False)
            end_time = time.perf_counter()
            
            sequential_throughput = 100 / (end_time - start_time)
            
            throughput_results['sequential_throughput_fps'] = sequential_throughput
            
        except Exception as e:
            throughput_results['sequential_error'] = str(e)
        
        # Concurrent throughput (if supported)
        try:
            def single_inference():
                test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                return self.model.predict(test_input, verbose=False)
            
            # Test with different thread counts
            for num_threads in [1, 2, 4]:
                try:
                    with ThreadPoolExecutor(max_workers=num_threads) as executor:
                        start_time = time.perf_counter()
                        
                        futures = [executor.submit(single_inference) for _ in range(20)]
                        for future in futures:
                            future.result()
                        
                        end_time = time.perf_counter()
                        concurrent_throughput = 20 / (end_time - start_time)
                        
                        throughput_results[f'concurrent_{num_threads}_threads_fps'] = concurrent_throughput
                        
                except Exception as e:
                    throughput_results[f'concurrent_{num_threads}_threads_error'] = str(e)
                    
        except Exception as e:
            throughput_results['concurrent_error'] = str(e)
        
        return throughput_results
    
    def _run_stress_benchmarks(self) -> Dict:
        """Run stress testing benchmarks."""
        stress_benchmarks = {}
        
        # Duration stress test
        stress_benchmarks['duration_stress'] = self._benchmark_duration_stress()
        
        # Memory stress test
        stress_benchmarks['memory_stress'] = self._benchmark_memory_stress()
        
        # Concurrent stress test
        stress_benchmarks['concurrent_stress'] = self._benchmark_concurrent_stress()
        
        return stress_benchmarks
    
    def _benchmark_duration_stress(self) -> Dict:
        """Benchmark model performance over extended duration."""
        try:
            duration_minutes = self.benchmark_config['stress_tests']['duration_minutes']
            test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            
            inference_times = []
            inference_count = 0
            
            while time.time() < end_time:
                inference_start = time.perf_counter()
                _ = self.model.predict(test_input, verbose=False)
                inference_end = time.perf_counter()
                
                inference_times.append((inference_end - inference_start) * 1000)
                inference_count += 1
                
                # Sample every 10th inference to avoid memory issues
                if len(inference_times) > 1000:
                    inference_times = inference_times[::10]
            
            actual_duration = time.time() - start_time
            
            return {
                'duration_minutes': actual_duration / 60,
                'total_inferences': inference_count,
                'avg_inference_time_ms': np.mean(inference_times),
                'inference_time_std_ms': np.std(inference_times),
                'throughput_fps': inference_count / actual_duration,
                'performance_degradation': (np.mean(inference_times[-100:]) - np.mean(inference_times[:100])) / np.mean(inference_times[:100]) if len(inference_times) > 200 else 0,
                'stable_performance': abs((np.mean(inference_times[-100:]) - np.mean(inference_times[:100])) / np.mean(inference_times[:100])) < 0.1 if len(inference_times) > 200 else True
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _benchmark_memory_stress(self) -> Dict:
        """Benchmark memory usage under stress conditions."""
        try:
            import psutil
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Gradually increase batch size to stress memory
            memory_results = {}
            
            for batch_size in [1, 2, 4, 8, 16, 32]:
                try:
                    batch_input = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) 
                                  for _ in range(batch_size)]
                    
                    # Run multiple iterations
                    for _ in range(5):
                        _ = self.model.predict(batch_input, verbose=False)
                    
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    
                    memory_results[f'batch_{batch_size}'] = {
                        'memory_mb': current_memory,
                        'memory_increase_mb': memory_increase,
                        'memory_per_batch_item_mb': memory_increase / batch_size if batch_size > 0 else 0
                    }
                    
                    # Stop if memory usage becomes excessive
                    if memory_increase > self.benchmark_config['stress_tests']['memory_limit_gb'] * 1024:
                        memory_results[f'batch_{batch_size}']['memory_limit_reached'] = True
                        break
                        
                except Exception as e:
                    memory_results[f'batch_{batch_size}'] = {'error': str(e)}
                    break
            
            return {
                'initial_memory_mb': initial_memory,
                'memory_stress_results': memory_results,
                'max_stable_batch_size': max([int(k.split('_')[1]) for k in memory_results.keys() 
                                            if k.startswith('batch_') and 'error' not in memory_results[k]])
            }
            
        except ImportError:
            return {'error': 'psutil not available for memory stress testing'}
        except Exception as e:
            return {'error': str(e)}
    
    def _benchmark_concurrent_stress(self) -> Dict:
        """Benchmark concurrent inference stress."""
        concurrent_results = {}
        
        for num_threads in self.benchmark_config['stress_tests']['concurrent_threads']:
            try:
                def concurrent_inference_worker():
                    test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    times = []
                    
                    for _ in range(10):  # 10 inferences per thread
                        start_time = time.perf_counter()
                        _ = self.model.predict(test_input, verbose=False)
                        end_time = time.perf_counter()
                        times.append((end_time - start_time) * 1000)
                    
                    return times
                
                # Run concurrent workers
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    start_time = time.perf_counter()
                    
                    futures = [executor.submit(concurrent_inference_worker) for _ in range(num_threads)]
                    all_times = []
                    
                    for future in futures:
                        thread_times = future.result()
                        all_times.extend(thread_times)
                    
                    end_time = time.perf_counter()
                    total_duration = end_time - start_time
                
                concurrent_results[f'{num_threads}_threads'] = {
                    'total_inferences': len(all_times),
                    'total_duration_s': total_duration,
                    'avg_inference_time_ms': np.mean(all_times),
                    'throughput_fps': len(all_times) / total_duration,
                    'thread_efficiency': (len(all_times) / total_duration) / num_threads,
                    'inference_time_std_ms': np.std(all_times)
                }
                
            except Exception as e:
                concurrent_results[f'{num_threads}_threads'] = {'error': str(e)}
        
        return concurrent_results    

    def _run_robustness_benchmarks(self) -> Dict:
        """Run robustness benchmarks."""
        robustness_benchmarks = {}
        
        # Noise robustness
        robustness_benchmarks['noise_robustness'] = self._benchmark_noise_robustness()
        
        # Lighting robustness
        robustness_benchmarks['lighting_robustness'] = self._benchmark_lighting_robustness()
        
        # Blur robustness
        robustness_benchmarks['blur_robustness'] = self._benchmark_blur_robustness()
        
        return robustness_benchmarks
    
    def _benchmark_noise_robustness(self) -> Dict:
        """Benchmark robustness to different noise levels."""
        try:
            base_input = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
            
            # Get baseline performance
            baseline_results = self.model.predict(base_input, verbose=False)
            baseline_detections = len(baseline_results[0].boxes) if baseline_results[0].boxes is not None else 0
            
            noise_results = {}
            
            for noise_level in self.benchmark_config['robustness_tests']['noise_levels']:
                try:
                    if noise_level == 0:
                        # No noise case
                        noisy_input = base_input
                    else:
                        # Add Gaussian noise
                        noisy_input = base_input.astype(np.float32)
                        noise = np.random.normal(0, noise_level, noisy_input.shape)
                        noisy_input = np.clip(noisy_input + noise, 0, 255).astype(np.uint8)
                    
                    # Test multiple times for stability
                    detection_counts = []
                    confidence_scores = []
                    
                    for _ in range(5):
                        results = self.model.predict(noisy_input, verbose=False)
                        detections = len(results[0].boxes) if results[0].boxes is not None else 0
                        detection_counts.append(detections)
                        
                        if results[0].boxes is not None and len(results[0].boxes) > 0:
                            confidences = results[0].boxes.conf.cpu().numpy()
                            confidence_scores.extend(confidences)
                    
                    avg_detections = np.mean(detection_counts)
                    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
                    
                    noise_results[f'noise_{noise_level}'] = {
                        'avg_detections': avg_detections,
                        'detection_ratio': avg_detections / max(baseline_detections, 1),
                        'avg_confidence': avg_confidence,
                        'detection_stability': np.std(detection_counts),
                        'noise_level': noise_level
                    }
                    
                except Exception as e:
                    noise_results[f'noise_{noise_level}'] = {'error': str(e)}
            
            # Calculate overall robustness score
            detection_ratios = [result['detection_ratio'] for result in noise_results.values() 
                              if isinstance(result, dict) and 'detection_ratio' in result]
            robustness_score = np.mean(detection_ratios) if detection_ratios else 0
            
            return {
                'baseline_detections': baseline_detections,
                'noise_results': noise_results,
                'robustness_score': robustness_score,
                'robust_to_noise': robustness_score > 0.7
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _benchmark_lighting_robustness(self) -> Dict:
        """Benchmark robustness to lighting variations."""
        try:
            base_input = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
            
            lighting_results = {}
            
            for brightness_factor in self.benchmark_config['robustness_tests']['brightness_factors']:
                try:
                    # Adjust brightness
                    bright_input = np.clip(base_input.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
                    
                    # Test detection performance
                    results = self.model.predict(bright_input, verbose=False)
                    detections = len(results[0].boxes) if results[0].boxes is not None else 0
                    
                    # Calculate image statistics
                    mean_brightness = np.mean(bright_input)
                    brightness_std = np.std(bright_input)
                    
                    lighting_results[f'brightness_{brightness_factor}'] = {
                        'detections': detections,
                        'mean_brightness': mean_brightness,
                        'brightness_std': brightness_std,
                        'brightness_factor': brightness_factor
                    }
                    
                except Exception as e:
                    lighting_results[f'brightness_{brightness_factor}'] = {'error': str(e)}
            
            # Assess lighting robustness
            detection_counts = [result['detections'] for result in lighting_results.values() 
                              if isinstance(result, dict) and 'detections' in result]
            lighting_robust = len([d for d in detection_counts if d > 0]) > len(detection_counts) * 0.6
            
            return {
                'lighting_results': lighting_results,
                'lighting_robust': lighting_robust,
                'detection_variance': np.var(detection_counts) if detection_counts else 0
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _benchmark_blur_robustness(self) -> Dict:
        """Benchmark robustness to blur effects."""
        try:
            base_input = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
            
            blur_results = {}
            
            for kernel_size in self.benchmark_config['robustness_tests']['blur_kernels']:
                try:
                    if kernel_size == 0:
                        # No blur case
                        blurred_input = base_input
                    else:
                        # Apply Gaussian blur
                        blurred_input = cv2.GaussianBlur(base_input, (kernel_size, kernel_size), 0)
                    
                    # Test detection performance
                    results = self.model.predict(blurred_input, verbose=False)
                    detections = len(results[0].boxes) if results[0].boxes is not None else 0
                    
                    blur_results[f'blur_{kernel_size}'] = {
                        'detections': detections,
                        'kernel_size': kernel_size
                    }
                    
                except Exception as e:
                    blur_results[f'blur_{kernel_size}'] = {'error': str(e)}
            
            # Assess blur robustness
            detection_counts = [result['detections'] for result in blur_results.values() 
                              if isinstance(result, dict) and 'detections' in result]
            blur_robust = len([d for d in detection_counts if d > 0]) > len(detection_counts) * 0.5
            
            return {
                'blur_results': blur_results,
                'blur_robust': blur_robust
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _run_security_benchmarks(self) -> Dict:
        """Run security-specific benchmarks."""
        security_benchmarks = {}
        
        # False positive rate benchmark
        security_benchmarks['false_positive_rate'] = self._benchmark_false_positive_rate()
        
        # Critical event detection benchmark
        security_benchmarks['critical_event_detection'] = self._benchmark_critical_event_detection()
        
        # Response time benchmark
        security_benchmarks['response_time'] = self._benchmark_response_time()
        
        return security_benchmarks
    
    def _benchmark_false_positive_rate(self) -> Dict:
        """Benchmark false positive rates for security applications."""
        try:
            # Test with different confidence thresholds
            fp_results = {}
            
            for conf_threshold in [0.1, 0.25, 0.5, 0.75]:
                results = self.model.val(
                    data=str(self.yolo_data_dir / "dataset.yaml"),
                    split='test',
                    conf=conf_threshold,
                    verbose=False
                )
                
                precision = float(results.box.mp)
                false_positive_rate = 1 - precision
                
                fp_results[str(conf_threshold)] = {
                    'precision': precision,
                    'false_positive_rate': false_positive_rate,
                    'acceptable_fp_rate': false_positive_rate <= 0.3  # 30% threshold
                }
            
            # Find optimal threshold for security (balance FP rate and detection)
            optimal_threshold = 0.5
            best_score = 0
            
            for threshold_str, metrics in fp_results.items():
                # Security score emphasizes low false positives
                security_score = metrics['precision'] * 0.8 + (1 - metrics['false_positive_rate']) * 0.2
                if security_score > best_score:
                    best_score = security_score
                    optimal_threshold = float(threshold_str)
            
            return {
                'threshold_results': fp_results,
                'optimal_threshold': optimal_threshold,
                'meets_security_standards': fp_results[str(optimal_threshold)]['acceptable_fp_rate']
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _benchmark_critical_event_detection(self) -> Dict:
        """Benchmark detection of critical security events."""
        try:
            # Focus on critical security classes
            critical_classes = ['violence', 'emergency', 'theft']
            
            results = self.model.val(
                data=str(self.yolo_data_dir / "dataset.yaml"),
                split='test',
                conf=0.25,
                verbose=False
            )
            
            critical_performance = {}
            class_names = list(self.config['classes'].keys())
            
            if hasattr(results.box, 'ap_class_index'):
                for class_name in critical_classes:
                    if class_name in class_names:
                        class_idx = class_names.index(class_name)
                        if class_idx in results.box.ap_class_index:
                            idx = list(results.box.ap_class_index).index(class_idx)
                            critical_performance[class_name] = {
                                'ap50': float(results.box.ap50[idx]),
                                'precision': float(results.box.p[idx]) if hasattr(results.box, 'p') else 0,
                                'recall': float(results.box.r[idx]) if hasattr(results.box, 'r') else 0
                            }
            
            # Calculate critical detection rate
            critical_recalls = [perf['recall'] for perf in critical_performance.values()]
            avg_critical_recall = np.mean(critical_recalls) if critical_recalls else 0
            
            return {
                'critical_performance': critical_performance,
                'avg_critical_recall': avg_critical_recall,
                'meets_critical_threshold': avg_critical_recall >= 0.8,  # 80% recall for critical events
                'critical_classes_detected': len(critical_performance)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _benchmark_response_time(self) -> Dict:
        """Benchmark end-to-end response time for security alerts."""
        try:
            # Simulate security alert pipeline
            test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            response_times = []
            
            for _ in range(20):
                # Measure complete pipeline time
                start_time = time.perf_counter()
                
                # 1. Inference
                results = self.model.predict(test_input, verbose=False)
                
                # 2. Post-processing (simulated)
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    # Simulate confidence filtering and NMS
                    time.sleep(0.001)  # 1ms processing time
                    
                    # Simulate alert generation
                    time.sleep(0.002)  # 2ms alert processing
                
                end_time = time.perf_counter()
                response_times.append((end_time - start_time) * 1000)  # Convert to ms
            
            avg_response_time = np.mean(response_times)
            
            return {
                'avg_response_time_ms': avg_response_time,
                'p95_response_time_ms': np.percentile(response_times, 95),
                'p99_response_time_ms': np.percentile(response_times, 99),
                'meets_realtime_requirement': avg_response_time <= 100,  # 100ms requirement
                'response_time_stability': np.std(response_times)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_benchmark_summary(self, benchmark_results: Dict) -> Dict:
        """Generate comprehensive benchmark summary."""
        summary = {
            'overall_score': 0,
            'category_scores': {},
            'performance_grade': 'Unknown',
            'key_metrics': {},
            'recommendations': [],
            'deployment_readiness': {}
        }
        
        try:
            # Calculate category scores
            categories = {
                'accuracy': self._score_accuracy_benchmarks(benchmark_results.get('accuracy_benchmarks', {})),
                'performance': self._score_performance_benchmarks(benchmark_results.get('performance_benchmarks', {})),
                'robustness': self._score_robustness_benchmarks(benchmark_results.get('robustness_benchmarks', {})),
                'security': self._score_security_benchmarks(benchmark_results.get('security_benchmarks', {}))
            }
            
            summary['category_scores'] = categories
            
            # Calculate overall score (weighted)
            weights = {'accuracy': 0.3, 'performance': 0.25, 'robustness': 0.2, 'security': 0.25}
            overall_score = sum(categories[cat] * weights[cat] for cat in categories)
            summary['overall_score'] = overall_score
            
            # Determine performance grade
            if overall_score >= 0.9:
                summary['performance_grade'] = 'Excellent'
            elif overall_score >= 0.8:
                summary['performance_grade'] = 'Good'
            elif overall_score >= 0.7:
                summary['performance_grade'] = 'Acceptable'
            elif overall_score >= 0.6:
                summary['performance_grade'] = 'Needs Improvement'
            else:
                summary['performance_grade'] = 'Poor'
            
            # Extract key metrics
            summary['key_metrics'] = self._extract_key_metrics(benchmark_results)
            
            # Generate recommendations
            summary['recommendations'] = self._generate_benchmark_recommendations(benchmark_results, categories)
            
            # Assess deployment readiness
            summary['deployment_readiness'] = self._assess_deployment_readiness(benchmark_results)
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def _score_accuracy_benchmarks(self, accuracy_benchmarks: Dict) -> float:
        """Score accuracy benchmark results."""
        try:
            standard_metrics = accuracy_benchmarks.get('standard_metrics', {})
            mAP50 = standard_metrics.get('mAP50', 0)
            precision = standard_metrics.get('precision', 0)
            recall = standard_metrics.get('recall', 0)
            
            # Weighted accuracy score
            accuracy_score = 0.4 * mAP50 + 0.3 * precision + 0.3 * recall
            return min(accuracy_score, 1.0)
        except:
            return 0.0
    
    def _score_performance_benchmarks(self, performance_benchmarks: Dict) -> float:
        """Score performance benchmark results."""
        try:
            inference_speed = performance_benchmarks.get('inference_speed', {})
            
            # Get best FPS across input sizes
            fps_values = []
            for size_result in inference_speed.values():
                if isinstance(size_result, dict) and 'fps' in size_result:
                    fps_values.append(size_result['fps'])
            
            best_fps = max(fps_values) if fps_values else 0
            
            # Score based on real-time capability
            if best_fps >= 30:
                performance_score = 1.0
            elif best_fps >= 15:
                performance_score = 0.8
            elif best_fps >= 10:
                performance_score = 0.6
            elif best_fps >= 5:
                performance_score = 0.4
            else:
                performance_score = 0.2
            
            return performance_score
        except:
            return 0.0
    
    def _score_robustness_benchmarks(self, robustness_benchmarks: Dict) -> float:
        """Score robustness benchmark results."""
        try:
            noise_robust = robustness_benchmarks.get('noise_robustness', {}).get('robust_to_noise', False)
            lighting_robust = robustness_benchmarks.get('lighting_robustness', {}).get('lighting_robust', False)
            blur_robust = robustness_benchmarks.get('blur_robustness', {}).get('blur_robust', False)
            
            robustness_score = (int(noise_robust) + int(lighting_robust) + int(blur_robust)) / 3
            return robustness_score
        except:
            return 0.0
    
    def _score_security_benchmarks(self, security_benchmarks: Dict) -> float:
        """Score security benchmark results."""
        try:
            fp_rate = security_benchmarks.get('false_positive_rate', {})
            critical_detection = security_benchmarks.get('critical_event_detection', {})
            response_time = security_benchmarks.get('response_time', {})
            
            fp_score = 1.0 if fp_rate.get('meets_security_standards', False) else 0.5
            critical_score = 1.0 if critical_detection.get('meets_critical_threshold', False) else 0.5
            response_score = 1.0 if response_time.get('meets_realtime_requirement', False) else 0.5
            
            security_score = (fp_score + critical_score + response_score) / 3
            return security_score
        except:
            return 0.0
    
    def _extract_key_metrics(self, benchmark_results: Dict) -> Dict:
        """Extract key metrics from benchmark results."""
        key_metrics = {}
        
        try:
            # Accuracy metrics
            accuracy = benchmark_results.get('accuracy_benchmarks', {}).get('standard_metrics', {})
            key_metrics['mAP50'] = accuracy.get('mAP50', 0)
            key_metrics['precision'] = accuracy.get('precision', 0)
            key_metrics['recall'] = accuracy.get('recall', 0)
            
            # Performance metrics
            performance = benchmark_results.get('performance_benchmarks', {})
            inference_speed = performance.get('inference_speed', {})
            
            fps_values = []
            for size_result in inference_speed.values():
                if isinstance(size_result, dict) and 'fps' in size_result:
                    fps_values.append(size_result['fps'])
            
            key_metrics['best_fps'] = max(fps_values) if fps_values else 0
            key_metrics['real_time_capable'] = key_metrics['best_fps'] >= 15
            
            # Model info
            model_info = benchmark_results.get('benchmark_info', {}).get('model_info', {})
            key_metrics['model_size_mb'] = model_info.get('model_size_mb', 0)
            key_metrics['parameters'] = model_info.get('parameters', 0)
            
        except Exception as e:
            key_metrics['extraction_error'] = str(e)
        
        return key_metrics
    
    def _generate_benchmark_recommendations(self, benchmark_results: Dict, category_scores: Dict) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        try:
            # Accuracy recommendations
            if category_scores.get('accuracy', 0) < 0.7:
                recommendations.append("Consider retraining with more data or hyperparameter optimization to improve accuracy")
            
            # Performance recommendations
            if category_scores.get('performance', 0) < 0.7:
                recommendations.append("Consider using a smaller model size (YOLOv8n) or optimizing inference for better performance")
            
            # Robustness recommendations
            if category_scores.get('robustness', 0) < 0.7:
                recommendations.append("Improve robustness through data augmentation and diverse training scenarios")
            
            # Security recommendations
            if category_scores.get('security', 0) < 0.7:
                recommendations.append("Optimize confidence thresholds and implement additional security validation layers")
            
            # Specific recommendations based on detailed results
            key_metrics = self._extract_key_metrics(benchmark_results)
            
            if key_metrics.get('best_fps', 0) < 15:
                recommendations.append("Model does not meet real-time requirements - consider model optimization or hardware upgrade")
            
            if key_metrics.get('model_size_mb', 0) > 50:
                recommendations.append("Model size may be too large for edge deployment - consider model compression")
            
        except Exception as e:
            recommendations.append(f"Could not generate all recommendations: {str(e)}")
        
        return recommendations
    
    def _assess_deployment_readiness(self, benchmark_results: Dict) -> Dict:
        """Assess readiness for different deployment scenarios."""
        readiness = {
            'edge_deployment': False,
            'server_deployment': False,
            'mobile_deployment': False,
            'production_ready': False
        }
        
        try:
            key_metrics = self._extract_key_metrics(benchmark_results)
            
            # Edge deployment readiness
            edge_ready = (
                key_metrics.get('best_fps', 0) >= 15 and
                key_metrics.get('model_size_mb', 0) <= 50 and
                key_metrics.get('mAP50', 0) >= 0.6
            )
            readiness['edge_deployment'] = edge_ready
            
            # Server deployment readiness
            server_ready = (
                key_metrics.get('mAP50', 0) >= 0.7 and
                key_metrics.get('precision', 0) >= 0.7
            )
            readiness['server_deployment'] = server_ready
            
            # Mobile deployment readiness
            mobile_ready = (
                key_metrics.get('model_size_mb', 0) <= 25 and
                key_metrics.get('best_fps', 0) >= 10
            )
            readiness['mobile_deployment'] = mobile_ready
            
            # Overall production readiness
            production_ready = (
                key_metrics.get('mAP50', 0) >= 0.7 and
                key_metrics.get('best_fps', 0) >= 15 and
                key_metrics.get('real_time_capable', False)
            )
            readiness['production_ready'] = production_ready
            
        except Exception as e:
            readiness['assessment_error'] = str(e)
        
        return readiness
    
    def _save_benchmark_results(self, results: Dict) -> None:
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = self.benchmark_dir / f"benchmark_{timestamp}"
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        results_file = results_dir / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate benchmark report
        self._generate_benchmark_report(results, results_dir)
        
        logger.info(f"Benchmark results saved to: {results_dir}")
    
    def _generate_benchmark_report(self, results: Dict, output_dir: Path) -> None:
        """Generate human-readable benchmark report."""
        report_file = output_dir / 'benchmark_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Security Model Benchmark Report\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            summary = results.get('benchmark_summary', {})
            f.write(f"**Overall Score**: {summary.get('overall_score', 0):.3f}\n")
            f.write(f"**Performance Grade**: {summary.get('performance_grade', 'Unknown')}\n\n")
            
            # Key Metrics
            f.write("## Key Metrics\n\n")
            key_metrics = summary.get('key_metrics', {})
            f.write(f"- **mAP@0.5**: {key_metrics.get('mAP50', 0):.3f}\n")
            f.write(f"- **Best FPS**: {key_metrics.get('best_fps', 0):.1f}\n")
            f.write(f"- **Model Size**: {key_metrics.get('model_size_mb', 0):.1f} MB\n")
            f.write(f"- **Real-time Capable**: {'Yes' if key_metrics.get('real_time_capable', False) else 'No'}\n\n")
            
            # Category Scores
            f.write("## Category Scores\n\n")
            category_scores = summary.get('category_scores', {})
            for category, score in category_scores.items():
                f.write(f"- **{category.title()}**: {score:.3f}\n")
            f.write("\n")
            
            # Deployment Readiness
            f.write("## Deployment Readiness\n\n")
            readiness = summary.get('deployment_readiness', {})
            for deployment_type, ready in readiness.items():
                if isinstance(ready, bool):
                    status = "✅ Ready" if ready else "❌ Not Ready"
                    f.write(f"- **{deployment_type.replace('_', ' ').title()}**: {status}\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            recommendations = summary.get('recommendations', [])
            for rec in recommendations:
                f.write(f"- {rec}\n")

def main():
    """Main benchmark function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive security model benchmarks')
    parser.add_argument('model_path', help='Path to trained YOLO model')
    parser.add_argument('--config', type=str, default='../config/dataset_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        # Initialize benchmark suite
        benchmark = SecurityModelBenchmark(args.model_path, args.config)
        
        # Run full benchmark suite
        results = benchmark.run_full_benchmark_suite()
        
        # Print summary
        summary = results.get('benchmark_summary', {})
        logger.info("="*60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*60)
        logger.info(f"Overall Score: {summary.get('overall_score', 0):.3f}")
        logger.info(f"Performance Grade: {summary.get('performance_grade', 'Unknown')}")
        
        key_metrics = summary.get('key_metrics', {})
        logger.info(f"mAP@0.5: {key_metrics.get('mAP50', 0):.3f}")
        logger.info(f"Best FPS: {key_metrics.get('best_fps', 0):.1f}")
        logger.info(f"Real-time Capable: {'Yes' if key_metrics.get('real_time_capable', False) else 'No'}")
        
        return 0 if summary.get('performance_grade') in ['Excellent', 'Good', 'Acceptable'] else 1
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
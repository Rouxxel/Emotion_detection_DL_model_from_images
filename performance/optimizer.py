#!/usr/bin/env python3
"""
Performance Optimization Module

This module provides performance optimization utilities for the emotion detection system,
including model optimization, inference acceleration, and resource monitoring.
"""

import logging
import os
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from functools import wraps
import threading
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2


class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {}
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start continuous performance monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return collected metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        summary = self._calculate_summary()
        self.logger.info("Performance monitoring stopped")
        return summary
    
    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            timestamp = time.time()
            
            # CPU and Memory metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU metrics (if available)
            gpu_metrics = self._get_gpu_metrics()
            
            metrics = {
                'timestamp': timestamp,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                **gpu_metrics
            }
            
            # Store metrics
            for key, value in metrics.items():
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)
            
            time.sleep(interval)
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU metrics if available."""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if not gpus:
                return {}
            
            # Get GPU memory info
            gpu_details = tf.config.experimental.get_memory_info('GPU:0')
            return {
                'gpu_memory_current_mb': gpu_details['current'] / (1024**2),
                'gpu_memory_peak_mb': gpu_details['peak'] / (1024**2)
            }
        except Exception:
            return {}
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics from collected metrics."""
        if not self.metrics:
            return {}
        
        summary = {}
        for key, values in self.metrics.items():
            if key == 'timestamp':
                continue
            
            if values:
                summary[f'{key}_avg'] = np.mean(values)
                summary[f'{key}_max'] = np.max(values)
                summary[f'{key}_min'] = np.min(values)
        
        return summary


class ModelOptimizer:
    """Optimize TensorFlow models for better performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize_model(self, model_path: str, optimization_type: str = "tflite") -> str:
        """
        Optimize a trained model for inference.
        
        Args:
            model_path: Path to the trained model
            optimization_type: Type of optimization ('tflite', 'tensorrt', 'onnx')
            
        Returns:
            Path to optimized model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.logger.info(f"Optimizing model with {optimization_type}")
        
        if optimization_type == "tflite":
            return self._optimize_tflite(model_path)
        elif optimization_type == "tensorrt":
            return self._optimize_tensorrt(model_path)
        elif optimization_type == "onnx":
            return self._optimize_onnx(model_path)
        else:
            raise ValueError(f"Unsupported optimization type: {optimization_type}")
    
    def _optimize_tflite(self, model_path: Path) -> str:
        """Optimize model using TensorFlow Lite."""
        # Load the model
        model = tf.keras.models.load_model(model_path)
        
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Enable dynamic range quantization
        converter.representative_dataset = self._representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save optimized model
        output_path = model_path.with_suffix('.tflite')
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Calculate compression ratio
        original_size = model_path.stat().st_size
        optimized_size = output_path.stat().st_size
        compression_ratio = original_size / optimized_size
        
        self.logger.info(f"TFLite optimization complete:")
        self.logger.info(f"  Original size: {original_size / (1024**2):.2f} MB")
        self.logger.info(f"  Optimized size: {optimized_size / (1024**2):.2f} MB")
        self.logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
        
        return str(output_path)
    
    def _representative_dataset_gen(self):
        """Generate representative dataset for quantization."""
        # Generate dummy data for quantization
        for _ in range(100):
            yield [np.random.random((1, 48, 48, 3)).astype(np.float32)]
    
    def _optimize_tensorrt(self, model_path: Path) -> str:
        """Optimize model using TensorRT (requires TensorRT installation)."""
        try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
            
            # Convert to TensorRT
            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=str(model_path),
                precision_mode=trt.TrtPrecisionMode.FP16
            )
            
            converter.convert()
            
            # Save optimized model
            output_path = model_path.parent / f"{model_path.stem}_tensorrt"
            converter.save(str(output_path))
            
            self.logger.info(f"TensorRT optimization complete: {output_path}")
            return str(output_path)
            
        except ImportError:
            self.logger.warning("TensorRT not available, skipping optimization")
            return str(model_path)
    
    def _optimize_onnx(self, model_path: Path) -> str:
        """Convert model to ONNX format."""
        try:
            import tf2onnx
            
            # Load model
            model = tf.keras.models.load_model(model_path)
            
            # Convert to ONNX
            output_path = model_path.with_suffix('.onnx')
            
            # Get model signature
            spec = (tf.TensorSpec((None, 48, 48, 3), tf.float32, name="input"),)
            
            # Convert
            onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
            
            # Save ONNX model
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            self.logger.info(f"ONNX conversion complete: {output_path}")
            return str(output_path)
            
        except ImportError:
            self.logger.warning("tf2onnx not available, skipping ONNX conversion")
            return str(model_path)


class InferenceOptimizer:
    """Optimize inference performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_tensorflow_optimizations()
    
    def _setup_tensorflow_optimizations(self) -> None:
        """Configure TensorFlow for optimal performance."""
        # Enable mixed precision
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            self.logger.info("Mixed precision enabled")
        except Exception as e:
            self.logger.warning(f"Could not enable mixed precision: {e}")
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info("GPU memory growth enabled")
            except RuntimeError as e:
                self.logger.warning(f"Could not configure GPU: {e}")
        
        # Enable XLA compilation
        tf.config.optimizer.set_jit(True)
        self.logger.info("XLA JIT compilation enabled")
    
    @contextmanager
    def optimized_inference(self):
        """Context manager for optimized inference."""
        # Disable eager execution for better performance
        tf.config.run_functions_eagerly(False)
        
        # Clear any existing graphs
        tf.keras.backend.clear_session()
        
        try:
            yield
        finally:
            # Cleanup
            gc.collect()
    
    def create_inference_function(self, model: Model) -> callable:
        """Create optimized inference function."""
        @tf.function(experimental_relax_shapes=True)
        def inference_fn(inputs):
            return model(inputs, training=False)
        
        return inference_fn
    
    def batch_inference(self, model: Model, images: List[np.ndarray], 
                       batch_size: int = 32) -> List[np.ndarray]:
        """Perform batched inference for better throughput."""
        results = []
        inference_fn = self.create_inference_function(model)
        
        with self.optimized_inference():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batch_array = np.array(batch)
                
                # Perform inference
                predictions = inference_fn(batch_array)
                results.extend(predictions.numpy())
        
        return results


class ImageProcessor:
    """Optimized image processing utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def preprocess_batch(self, images: List[np.ndarray], 
                        target_size: Tuple[int, int] = (48, 48),
                        normalize: bool = True) -> np.ndarray:
        """Optimized batch image preprocessing."""
        processed_images = []
        
        for img in images:
            # Resize using OpenCV (faster than PIL)
            if img.shape[:2] != target_size:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize if requested
            if normalize:
                img = img.astype(np.float32) / 255.0
            
            processed_images.append(img)
        
        return np.array(processed_images, dtype=np.float32)
    
    def preprocess_stream(self, frame: np.ndarray, 
                         target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
        """Optimized single frame preprocessing for real-time streams."""
        # Resize
        if frame.shape[:2] != target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize and add batch dimension
        frame = frame.astype(np.float32) / 255.0
        return np.expand_dims(frame, axis=0)


def performance_profiler(func):
    """Decorator to profile function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger = logging.getLogger(func.__module__)
            logger.info(f"Performance Profile - {func.__name__}:")
            logger.info(f"  Execution time: {execution_time:.3f}s")
            logger.info(f"  Memory delta: {memory_delta:+.2f}MB")
            
            return result
            
        except Exception as e:
            logger = logging.getLogger(func.__module__)
            logger.error(f"Function {func.__name__} failed: {str(e)}")
            raise
    
    return wrapper


class ResourceManager:
    """Manage system resources efficiently."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize_system_settings(self) -> None:
        """Optimize system settings for ML workloads."""
        # Set thread counts
        num_cores = psutil.cpu_count(logical=False)
        
        # Configure TensorFlow threading
        tf.config.threading.set_inter_op_parallelism_threads(num_cores)
        tf.config.threading.set_intra_op_parallelism_threads(num_cores)
        
        # Configure OpenCV threading
        cv2.setNumThreads(num_cores)
        
        # Set environment variables for optimal performance
        os.environ['OMP_NUM_THREADS'] = str(num_cores)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
        
        self.logger.info(f"System optimized for {num_cores} cores")
    
    def cleanup_resources(self) -> None:
        """Clean up system resources."""
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU memory if available
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.reset_memory_stats('GPU:0')
        except Exception:
            pass
        
        self.logger.info("Resources cleaned up")
    
    def get_optimal_batch_size(self, model: Model, input_shape: Tuple[int, ...]) -> int:
        """Determine optimal batch size based on available memory."""
        try:
            # Get available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            # Estimate model memory usage
            model_params = model.count_params()
            model_memory_gb = (model_params * 4) / (1024**3)  # 4 bytes per float32
            
            # Calculate input memory per sample
            input_size = np.prod(input_shape) * 4 / (1024**3)  # 4 bytes per float32
            
            # Reserve 2GB for system and other processes
            usable_memory = max(0.5, available_gb - 2.0 - model_memory_gb)
            
            # Calculate optimal batch size
            optimal_batch_size = int(usable_memory / (input_size * 2))  # Factor of 2 for safety
            
            # Clamp to reasonable range
            optimal_batch_size = max(1, min(optimal_batch_size, 128))
            
            self.logger.info(f"Optimal batch size: {optimal_batch_size}")
            return optimal_batch_size
            
        except Exception as e:
            self.logger.warning(f"Could not determine optimal batch size: {e}")
            return 32  # Default fallback


def main():
    """Example usage of performance optimization tools."""
    # Initialize components
    monitor = PerformanceMonitor()
    optimizer = ModelOptimizer()
    inference_opt = InferenceOptimizer()
    resource_mgr = ResourceManager()
    
    # Optimize system settings
    resource_mgr.optimize_system_settings()
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Example: optimize a model (if it exists)
        model_path = "trained_dl_models/emotion_detection_from_image_transfer_learning.h5"
        if os.path.exists(model_path):
            optimized_path = optimizer.optimize_model(model_path, "tflite")
            print(f"Model optimized: {optimized_path}")
        
        # Simulate some work
        time.sleep(5)
        
    finally:
        # Stop monitoring and get results
        metrics = monitor.stop_monitoring()
        print("Performance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")
        
        # Cleanup
        resource_mgr.cleanup_resources()


if __name__ == "__main__":
    main()
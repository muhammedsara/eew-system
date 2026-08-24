"""
1D CNN Earthquake Detection Model

Architecture from Paper Section III.E:
- Input: 150 timesteps × 6 channels (accel_xyz + gyro_xyz)
- 3 convolutional layers (32-64-128 filters)
- GlobalAveragePooling + Dense(2) with softmax
- Deployed size: 188 KiB INT8 TFLite

The model is designed to be:
1. Lightweight for edge deployment (mobile + Raspberry Pi)
2. Accurate enough for reliable earthquake detection
3. Extensible for future dataset integrations (MyShake, INSTANCE, etc.)

Author: Muhammed Şara
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import os

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, Sequential
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Model training disabled.")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class EarthquakeDetectorCNN:
    """
    1D CNN for Earthquake Detection
    
    This model distinguishes earthquake P-waves from everyday vibrations
    (walking, vehicle motion, dropped phones, etc.)
    
    Architecture Details:
    - Conv1D layers with increasing filter depth (32 → 64 → 128)
    - BatchNormalization for training stability
    - MaxPooling for dimensionality reduction
    - GlobalAveragePooling for fixed-size output
    - Dense output with softmax for probability distribution
    
    Attributes:
        model: Keras Sequential model
        config: Model configuration from config.py
        input_shape: (timesteps, channels) tuple
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize the earthquake detector model.
        
        Args:
            config_override: Optional dict to override default config values
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for model operations")
        
        self.cfg = config.model
        if config_override:
            for key, value in config_override.items():
                if hasattr(self.cfg, key):
                    setattr(self.cfg, key, value)
        
        self.input_shape = (self.cfg.input_timesteps, self.cfg.input_channels)
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """
        Build the 1D CNN architecture.
        
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential([
            # Input specification
            layers.InputLayer(input_shape=self.input_shape),
            
            # Conv Block 1: 32 filters
            layers.Conv1D(
                filters=self.cfg.conv_filters[0],
                kernel_size=self.cfg.kernel_size,
                padding='same',
                activation='relu',
                name='conv1'
            ),
            layers.BatchNormalization(name='bn1'),
            layers.MaxPooling1D(pool_size=self.cfg.pool_size, name='pool1'),
            layers.Dropout(self.cfg.dropout_rate, name='drop1'),
            
            # Conv Block 2: 64 filters
            layers.Conv1D(
                filters=self.cfg.conv_filters[1],
                kernel_size=self.cfg.kernel_size,
                padding='same',
                activation='relu',
                name='conv2'
            ),
            layers.BatchNormalization(name='bn2'),
            layers.MaxPooling1D(pool_size=self.cfg.pool_size, name='pool2'),
            layers.Dropout(self.cfg.dropout_rate, name='drop2'),
            
            # Conv Block 3: 128 filters
            layers.Conv1D(
                filters=self.cfg.conv_filters[2],
                kernel_size=self.cfg.kernel_size,
                padding='same',
                activation='relu',
                name='conv3'
            ),
            layers.BatchNormalization(name='bn3'),
            layers.MaxPooling1D(pool_size=self.cfg.pool_size, name='pool3'),
            layers.Dropout(self.cfg.dropout_rate, name='drop3'),
            
            # Global pooling and output
            layers.GlobalAveragePooling1D(name='gap'),
            layers.Dense(64, activation='relu', name='fc1'),
            layers.Dropout(self.cfg.dropout_rate, name='drop4'),
            layers.Dense(self.cfg.num_classes, activation='softmax', name='output')
        ], name='earthquake_detector_cnn')
        
        # Compile with Adam optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.cfg.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def summary(self) -> str:
        """Get model summary as string"""
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Train the model with early stopping and learning rate reduction.
        
        Args:
            X_train: Training data (n_samples, timesteps, channels)
            y_train: Training labels (n_samples, num_classes) one-hot encoded
            X_val: Validation data
            y_val: Validation labels
            callbacks: Optional additional callbacks
            
        Returns:
            Training history dictionary
        """
        # Default callbacks
        default_callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.cfg.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.cfg.model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        if callbacks:
            default_callbacks.extend(callbacks)
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.cfg.batch_size,
            epochs=self.cfg.epochs,
            validation_data=(X_val, y_val),
            callbacks=default_callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on input data.
        
        Args:
            X: Input data (n_samples, timesteps, channels) or (timesteps, channels)
            
        Returns:
            Tuple of (predicted_classes, confidence_scores)
        """
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
        
        probabilities = self.model.predict(X, verbose=0)
        classes = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        
        return classes, confidences
    
    def predict_single(self, X: np.ndarray) -> Tuple[bool, float]:
        """
        Make prediction for a single sample.
        
        Args:
            X: Single sample (timesteps, channels)
            
        Returns:
            Tuple of (is_earthquake, confidence)
        """
        classes, confidences = self.predict(X)
        is_earthquake = bool(classes[0] == 1)
        confidence = float(confidences[0])
        
        return is_earthquake, confidence
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Dictionary with loss, accuracy, precision, recall
        """
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = float(value)
        
        return metrics
    
    def save(self, path: Optional[str] = None):
        """Save model to H5 file"""
        path = path or self.cfg.model_save_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to: {path}")
    
    def load(self, path: Optional[str] = None):
        """Load model from H5 file"""
        path = path or self.cfg.model_save_path
        self.model = keras.models.load_model(path)
        print(f"Model loaded from: {path}")
    
    def convert_to_tflite(
        self,
        output_path: Optional[str] = None,
        quantize: bool = True
    ) -> str:
        """
        Convert model to TensorFlow Lite for edge deployment.
        
        Args:
            output_path: Output path for .tflite file
            quantize: Whether to apply dynamic range quantization
            
        Returns:
            Path to saved TFLite model
        """
        output_path = output_path or self.cfg.tflite_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            # Dynamic range quantization for smaller model
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Calculate size
        size_kb = len(tflite_model) / 1024
        print(f"TFLite model saved to: {output_path} ({size_kb:.1f} KB)")
        
        return output_path
    
    def get_model_size(self) -> Dict[str, float]:
        """Get model size information"""
        # Count parameters
        trainable = np.sum([
            np.prod(w.shape) for w in self.model.trainable_weights
        ])
        non_trainable = np.sum([
            np.prod(w.shape) for w in self.model.non_trainable_weights
        ])
        
        return {
            'trainable_params': int(trainable),
            'non_trainable_params': int(non_trainable),
            'total_params': int(trainable + non_trainable),
            'estimated_size_kb': float((trainable + non_trainable) * 4 / 1024)
        }


class TFLiteInference:
    """
    TFLite inference wrapper for edge deployment.
    
    This class is used for inference on mobile devices and Raspberry Pi,
    achieving <200ms inference time.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize TFLite interpreter.
        
        Args:
            model_path: Path to .tflite model file
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TFLite inference")
        
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape'][1:]
    
    def predict(self, X: np.ndarray) -> Tuple[int, float]:
        """
        Run inference on a single sample.
        
        Args:
            X: Input data (timesteps, channels)
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Prepare input
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
        
        X = X.astype(np.float32)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], X)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        predicted_class = int(np.argmax(output[0]))
        confidence = float(np.max(output[0]))
        
        return predicted_class, confidence
    
    def predict_earthquake(self, X: np.ndarray) -> Tuple[bool, float]:
        """
        Predict if input is an earthquake.
        
        Args:
            X: Input accelerometer data
            
        Returns:
            Tuple of (is_earthquake, confidence)
        """
        predicted_class, confidence = self.predict(X)
        return bool(predicted_class == 1), confidence


def create_model(config_override: Optional[Dict] = None) -> EarthquakeDetectorCNN:
    """
    Factory function to create a new model.
    
    Args:
        config_override: Optional config overrides
        
    Returns:
        Initialized EarthquakeDetectorCNN instance
    """
    return EarthquakeDetectorCNN(config_override)


def load_model(path: str) -> EarthquakeDetectorCNN:
    """
    Load a pre-trained model.
    
    Args:
        path: Path to model file (.h5)
        
    Returns:
        EarthquakeDetectorCNN with loaded weights
    """
    detector = EarthquakeDetectorCNN()
    detector.load(path)
    return detector

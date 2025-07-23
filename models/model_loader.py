import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import suppress_warnings

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, weights_dir="models/weights"):
        """Initialize model loader with paths to weight files"""
        self.weights_dir = Path(weights_dir)
        
        # Model paths (adjust these based on your setup)
        self.STGCN_WEIGHTS = self.weights_dir / "best_stgcn.weights.h5"
        self.TRANSFORMER_MODEL_PATH = self.weights_dir / "Transformer_12rel_4_bs16_sl32.keras"
        self.ANGLE_MODEL_PATH = self.weights_dir / "Transformer_12rel_4_angle3_branch_bs16_sl32.keras"
        self.SWIN3D_WEIGHTS = self.weights_dir / "best_swin3d_b_22k.pth"
        
        # Constants
        self.SEQ_LEN = 32
        self.NUM_CLASSES = 22
        self.ACTIONS = [
            "barbell biceps curl","lateral raise","push-up","bench press",
            "chest fly machine","deadlift","decline bench press","hammer curl",
            "hip thrust","incline bench press","lat pulldown","leg extension",
            "leg raises","plank","pull Up","romanian deadlift","russian twist",
            "shoulder press","squat","t bar row","tricep Pushdown","tricep dips"
        ]
        
        # Device for PyTorch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self.models = {}
        self._load_all_models()
    
    def _build_stgcn_model(self):
        """Build ST-GCN model architecture"""
        from tensorflow.keras import layers, Model, regularizers
        
        # ST-GCN Block
        class STGCNBlock(layers.Layer):
            def __init__(self, in_ch, out_ch, A_norm, stride=1, dropout=0.3, **kwargs):
                super().__init__(**kwargs)
                self.A_const = tf.constant(A_norm.astype(np.float32))
                self.B = self.add_weight(shape=A_norm.shape, initializer="zeros", trainable=True)
                self.conv1x1 = layers.Conv2D(out_ch, 1, use_bias=False,
                                            kernel_regularizer=regularizers.l2(1e-5))
                self.branches = [
                    layers.Conv2D(out_ch, (k,1), strides=(stride,1), padding="same",
                                  use_bias=False, kernel_regularizer=regularizers.l2(1e-5))
                    for k in (3,5,9)
                ]
                self.bn = layers.BatchNormalization()
                if in_ch == out_ch and stride == 1:
                    self.res = lambda x, training: x
                else:
                    self.res = tf.keras.Sequential([
                        layers.Conv2D(out_ch,1,strides=(stride,1),padding="same",
                                      use_bias=False, kernel_regularizer=regularizers.l2(1e-5)),
                        layers.BatchNormalization()
                    ])
                self.act = layers.Activation("relu")
                self.drop = layers.SpatialDropout2D(dropout)

            def call(self, x, training=False):
                A_mat = self.A_const + tf.nn.softmax(self.B, axis=1)
                x_sp = tf.einsum('ij,btjk->btik', A_mat, x)
                x_sp = self.conv1x1(x_sp)
                out = sum(branch(x_sp) for branch in self.branches) / len(self.branches)
                out = self.bn(out, training=training)
                r = self.res(x, training=training) if callable(self.res) else self.res(x)
                y = self.act(out + r)
                return self.drop(y, training=training)
        
        # Build adjacency matrix
        V = 31  # Number of joints in MediaPipe
        connections = [
            (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
            (11,12),(11,13),(13,15),(15,17),(17,19),(19,21),
            (12,14),(14,16),(16,18),(18,20),(20,22),
            (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(27,31),
            (24,26),(26,28),(28,30)
        ]
        
        A = np.zeros((V, V), dtype=np.float32)
        for u, v in connections:
            if u < V and v < V:
                A[u, v] = A[v, u] = 1.0
        
        # Normalize adjacency matrix
        D = A.sum(axis=1)
        D_inv = np.diag(1.0 / np.sqrt(D + 1e-6))
        A_norm = D_inv @ A @ D_inv
        
        # Build model
        C = 4  # x, y, z, visibility
        inp = layers.Input((self.SEQ_LEN, V, C))
        x = STGCNBlock(C,   64, A_norm, stride=1)(inp)
        x = STGCNBlock(64,  64, A_norm, stride=2)(x)
        x = STGCNBlock(64, 128, A_norm, stride=2)(x)
        x = STGCNBlock(128,256, A_norm, stride=2)(x)
        x = layers.GlobalAveragePooling2D()(x)
        out = layers.Dense(self.NUM_CLASSES, activation="softmax",
                          kernel_regularizer=regularizers.l2(1e-5))(x)
        
        return Model(inp, out)
    
    def _build_swin3d_model(self):
        """Build Swin3D model"""
        from torchvision.models.video import swin3d_b, Swin3D_B_Weights
        
        model = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
        model.head = nn.Linear(model.head.in_features, self.NUM_CLASSES)
        model = model.to(self.device)
        
        return model
    
    @tf.keras.utils.register_keras_serializable()
    class PositionalEncoding(tf.keras.layers.Layer):
        """Positional encoding for Transformer models"""
        def __init__(self, maxlen, dm, **kwargs):
            super().__init__(**kwargs)
            pos = np.arange(maxlen)[:, None]
            i = np.arange(dm)[None, :]
            angle = pos / np.power(10000, (2*(i//2))/dm)
            pe = np.zeros((maxlen, dm), dtype=np.float32)
            pe[:,0::2] = np.sin(angle[:,0::2])
            pe[:,1::2] = np.cos(angle[:,1::2])
            self.pe = tf.constant(pe[None,...])
        
        def call(self, x):
            return x + self.pe[:, :tf.shape(x)[1],:]
        
        def get_config(self):
            cfg = super().get_config()
            cfg.update({"maxlen": int(self.pe.shape[1]), "dm": int(self.pe.shape[2])})
            return cfg
    
    def _load_all_models(self):
        """Load all four models"""
        logger.info("Loading all models...")
        
        try:
            # 1. Load ST-GCN
            logger.info("Loading ST-GCN...")
            if self.STGCN_WEIGHTS.exists():
                model_stgcn = self._build_stgcn_model()
                try:
                    model_stgcn.load_weights(str(self.STGCN_WEIGHTS))
                    logger.info("ST-GCN weights loaded successfully.")
                except Exception as e:
                    logger.warning(f"ST-GCN weight load error: {e}. Using skip_mismatch.")
                    model_stgcn.load_weights(str(self.STGCN_WEIGHTS), skip_mismatch=True)
                self.models['stgcn'] = model_stgcn
            else:
                logger.warning(f"ST-GCN weights not found: {self.STGCN_WEIGHTS}")
                self.models['stgcn'] = None
            
            # 2. Load Transformer 12rel
            logger.info("Loading Transformer 12rel...")
            if self.TRANSFORMER_MODEL_PATH.exists():
                model_transformer = tf.keras.models.load_model(
                    str(self.TRANSFORMER_MODEL_PATH),
                    custom_objects={'PositionalEncoding': self.PositionalEncoding}
                )
                self.models['transformer_12rel'] = model_transformer
                logger.info("Transformer 12rel loaded successfully.")
            else:
                logger.warning(f"Transformer 12rel not found: {self.TRANSFORMER_MODEL_PATH}")
                self.models['transformer_12rel'] = None
            
            # 3. Load Transformer angle branch
            logger.info("Loading Transformer angle branch...")
            if self.ANGLE_MODEL_PATH.exists():
                model_angle = tf.keras.models.load_model(str(self.ANGLE_MODEL_PATH))
                self.models['transformer_angle'] = model_angle
                logger.info("Transformer angle branch loaded successfully.")
            else:
                logger.warning(f"Transformer angle not found: {self.ANGLE_MODEL_PATH}")
                self.models['transformer_angle'] = None
            
            # 4. Load Swin3D
            logger.info("Loading Swin3D...")
            if self.SWIN3D_WEIGHTS.exists():
                model_swin3d = self._build_swin3d_model()
                state = torch.load(str(self.SWIN3D_WEIGHTS), map_location=self.device)
                model_swin3d.load_state_dict(state)
                model_swin3d.eval()
                self.models['swin3d'] = model_swin3d
                logger.info("Swin3D loaded successfully.")
            else:
                logger.warning(f"Swin3D weights not found: {self.SWIN3D_WEIGHTS}")
                self.models['swin3d'] = None
            
            # Check if any models loaded
            loaded_models = [name for name, model in self.models.items() if model is not None]
            if not loaded_models:
                logger.warning("No models could be loaded. App will run in demo mode with mock predictions.")
                # Don't raise error, just warn - app can still work for testing UI
            else:
                logger.info(f"Successfully loaded models: {loaded_models}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def get_model(self, model_name):
        """Get a specific model by name"""
        return self.models.get(model_name)
    
    def get_available_models(self):
        """Get list of available (loaded) models"""
        return [name for name, model in self.models.items() if model is not None]
    
    def predict_stgcn(self, X):
        """Predict using ST-GCN model"""
        if self.models['stgcn'] is None:
            return None
        return self.models['stgcn'].predict(X, batch_size=32, verbose=0)
    
    def predict_transformer_12rel(self, X):
        """Predict using Transformer 12rel model"""
        if self.models['transformer_12rel'] is None:
            return None
        return self.models['transformer_12rel'].predict(X, batch_size=32, verbose=0)
    
    def predict_transformer_angle(self, X_rel, X_ang):
        """Predict using Transformer angle branch model"""
        if self.models['transformer_angle'] is None:
            return None
        return self.models['transformer_angle'].predict([X_rel, X_ang], batch_size=32, verbose=0)
    
    def predict_swin3d(self, X):
        """Predict using Swin3D model"""
        if self.models['swin3d'] is None:
            return None
        
        with torch.no_grad():
            probas = []
            for x in X:
                x_batch = x.unsqueeze(0).to(self.device)
                logits = self.models['swin3d'](x_batch)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                probas.append(proba)
        
        return np.vstack(probas)
    
    def cleanup(self):
        """Clean up model resources"""
        logger.info("Cleaning up model resources...")
        for name, model in self.models.items():
            if model is not None:
                if name == 'swin3d':
                    # PyTorch model cleanup
                    del model
                    torch.cuda.empty_cache()
                else:
                    # TensorFlow model cleanup
                    del model
        
        # Clear TensorFlow session
        tf.keras.backend.clear_session() 
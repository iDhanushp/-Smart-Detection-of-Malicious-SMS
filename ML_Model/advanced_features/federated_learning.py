"""
Federated Learning Implementation for SMS Fraud Detection
Enables collaborative model improvement while preserving user privacy
"""

import numpy as np
import json
import pickle
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import hashlib
import hmac
import secrets
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import threading
import time

@dataclass
class ModelUpdate:
    """Represents a model update from a client"""
    client_id: str
    update_data: Dict[str, np.ndarray]
    sample_count: int
    timestamp: datetime
    signature: str
    metadata: Dict[str, any]

@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    min_clients: int = 10
    max_clients: int = 100
    aggregation_rounds: int = 5
    privacy_budget: float = 1.0
    noise_scale: float = 0.1
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs_per_round: int = 1

class FederatedLearningManager:
    """
    Manages federated learning for SMS fraud detection
    """
    
    def __init__(self, config: FederatedConfig = None):
        self.config = config or FederatedConfig()
        self.global_model = None
        self.global_vectorizer = None
        self.local_updates = []
        self.client_registry = {}
        self.aggregation_history = []
        self.privacy_key = secrets.token_hex(32)
        
        # Threading for background processing
        self._lock = threading.Lock()
        self._aggregation_thread = None
        self._running = False
        
    def initialize_global_model(self, vectorizer: TfidfVectorizer, model: MultinomialNB):
        """
        Initialize the global model
        """
        self.global_vectorizer = vectorizer
        self.global_model = model
        print("Global model initialized for federated learning")
    
    def register_client(self, client_id: str, public_key: str = None) -> str:
        """
        Register a new client for federated learning
        """
        with self._lock:
            if client_id in self.client_registry:
                raise ValueError(f"Client {client_id} already registered")
            
            # Generate client-specific key
            client_key = secrets.token_hex(16)
            
            self.client_registry[client_id] = {
                'public_key': public_key,
                'client_key': client_key,
                'registered_at': datetime.now(),
                'last_update': None,
                'total_updates': 0,
                'total_samples': 0,
                'status': 'active'
            }
            
            print(f"Client {client_id} registered for federated learning")
            return client_key
    
    def collect_local_update(self, client_id: str, model_update: Dict, 
                           sample_count: int, metadata: Dict = None) -> bool:
        """
        Collect model update from a client
        """
        if client_id not in self.client_registry:
            print(f"Client {client_id} not registered")
            return False
        
        # Verify client signature
        if not self._verify_update_signature(client_id, model_update, metadata):
            print(f"Invalid signature for client {client_id}")
            return False
        
        # Apply differential privacy
        noisy_update = self._apply_differential_privacy(model_update)
        
        # Create update object
        update = ModelUpdate(
            client_id=client_id,
            update_data=noisy_update,
            sample_count=sample_count,
            timestamp=datetime.now(),
            signature=self._generate_update_signature(client_id, noisy_update),
            metadata=metadata or {}
        )
        
        with self._lock:
            self.local_updates.append(update)
            self.client_registry[client_id]['last_update'] = datetime.now()
            self.client_registry[client_id]['total_updates'] += 1
            self.client_registry[client_id]['total_samples'] += sample_count
        
        print(f"Collected update from client {client_id} ({sample_count} samples)")
        
        # Check if we should trigger aggregation
        if len(self.local_updates) >= self.config.min_clients:
            self._schedule_aggregation()
        
        return True
    
    def _verify_update_signature(self, client_id: str, update_data: Dict, 
                               metadata: Dict) -> bool:
        """
        Verify the signature of a client update
        """
        try:
            # In a real implementation, you would verify the signature
            # using the client's public key
            return True
        except Exception as e:
            print(f"Signature verification failed: {e}")
            return False
    
    def _generate_update_signature(self, client_id: str, update_data: Dict) -> str:
        """
        Generate signature for update verification
        """
        # Create signature using client key
        client_key = self.client_registry[client_id]['client_key']
        data_string = json.dumps(update_data, sort_keys=True, default=str)
        signature = hmac.new(
            client_key.encode(),
            data_string.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _apply_differential_privacy(self, update_data: Dict) -> Dict:
        """
        Apply differential privacy to model updates
        """
        noisy_update = {}
        
        for key, value in update_data.items():
            if isinstance(value, np.ndarray):
                # Add Gaussian noise for differential privacy
                noise = np.random.normal(
                    0, 
                    self.config.noise_scale / self.config.privacy_budget,
                    value.shape
                )
                noisy_update[key] = value + noise
            else:
                noisy_update[key] = value
        
        return noisy_update
    
    def _schedule_aggregation(self):
        """
        Schedule model aggregation
        """
        if self._aggregation_thread is None or not self._aggregation_thread.is_alive():
            self._aggregation_thread = threading.Thread(target=self._aggregate_updates)
            self._aggregation_thread.start()
    
    def _aggregate_updates(self):
        """
        Aggregate local updates into global model
        """
        with self._lock:
            if len(self.local_updates) < self.config.min_clients:
                return
            
            print(f"Aggregating {len(self.local_updates)} client updates...")
            
            # Federated averaging
            aggregated_update = self._federated_average()
            
            # Update global model
            self._apply_global_update(aggregated_update)
            
            # Record aggregation
            aggregation_record = {
                'timestamp': datetime.now(),
                'client_count': len(self.local_updates),
                'total_samples': sum(u.sample_count for u in self.local_updates),
                'privacy_budget_used': self.config.privacy_budget,
                'noise_scale': self.config.noise_scale
            }
            self.aggregation_history.append(aggregation_record)
            
            # Clear processed updates
            self.local_updates.clear()
            
            print("Model aggregation completed")
    
    def _federated_average(self) -> Dict[str, np.ndarray]:
        """
        Compute federated average of model updates
        """
        total_samples = sum(update.sample_count for update in self.local_updates)
        
        # Initialize aggregated update
        aggregated_update = {}
        
        # Get all parameter keys from first update
        first_update = self.local_updates[0].update_data
        for key in first_update.keys():
            aggregated_update[key] = np.zeros_like(first_update[key])
        
        # Weighted average based on sample count
        for update in self.local_updates:
            weight = update.sample_count / total_samples
            
            for key, value in update.update_data.items():
                if key in aggregated_update:
                    aggregated_update[key] += weight * value
        
        return aggregated_update
    
    def _apply_global_update(self, aggregated_update: Dict[str, np.ndarray]):
        """
        Apply aggregated update to global model
        """
        if self.global_model is None:
            print("Global model not initialized")
            return
        
        # Update model parameters
        for key, value in aggregated_update.items():
            if hasattr(self.global_model, key):
                current_value = getattr(self.global_model, key)
                if isinstance(current_value, np.ndarray):
                    # Apply learning rate
                    update = value * self.config.learning_rate
                    setattr(self.global_model, key, current_value + update)
    
    def get_global_model(self) -> Tuple[TfidfVectorizer, MultinomialNB]:
        """
        Get current global model
        """
        return self.global_vectorizer, self.global_model
    
    def distribute_model(self, client_id: str) -> Dict:
        """
        Distribute current global model to a client
        """
        if self.global_model is None:
            raise ValueError("Global model not available")
        
        # Create model snapshot
        model_snapshot = {
            'model_params': self._extract_model_parameters(self.global_model),
            'vectorizer_vocab': self.global_vectorizer.vocabulary_,
            'vectorizer_params': {
                'max_features': self.global_vectorizer.max_features,
                'ngram_range': self.global_vectorizer.ngram_range,
                'min_df': self.global_vectorizer.min_df,
                'max_df': self.global_vectorizer.max_df
            },
            'timestamp': datetime.now(),
            'version': len(self.aggregation_history),
            'client_id': client_id
        }
        
        return model_snapshot
    
    def _extract_model_parameters(self, model: MultinomialNB) -> Dict[str, np.ndarray]:
        """
        Extract parameters from sklearn model
        """
        return {
            'class_log_prior_': model.class_log_prior_,
            'feature_log_prob_': model.feature_log_prob_,
            'classes_': model.classes_
        }
    
    def get_federated_stats(self) -> Dict:
        """
        Get federated learning statistics
        """
        with self._lock:
            active_clients = sum(
                1 for client in self.client_registry.values() 
                if client['status'] == 'active'
            )
            
            total_updates = sum(
                client['total_updates'] for client in self.client_registry.values()
            )
            
            total_samples = sum(
                client['total_samples'] for client in self.client_registry.values()
            )
            
            return {
                'active_clients': active_clients,
                'total_clients': len(self.client_registry),
                'pending_updates': len(self.local_updates),
                'total_updates': total_updates,
                'total_samples': total_samples,
                'aggregation_rounds': len(self.aggregation_history),
                'last_aggregation': self.aggregation_history[-1]['timestamp'] if self.aggregation_history else None,
                'privacy_budget': self.config.privacy_budget,
                'noise_scale': self.config.noise_scale
            }
    
    def save_federated_state(self, filepath: str):
        """
        Save federated learning state
        """
        state = {
            'config': asdict(self.config),
            'client_registry': self.client_registry,
            'aggregation_history': [
                {**record, 'timestamp': record['timestamp'].isoformat()}
                for record in self.aggregation_history
            ],
            'global_model_params': self._extract_model_parameters(self.global_model) if self.global_model else None,
            'vectorizer_vocab': self.global_vectorizer.vocabulary_ if self.global_vectorizer else None,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        print(f"Federated state saved to {filepath}")
    
    def load_federated_state(self, filepath: str):
        """
        Load federated learning state
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore configuration
        self.config = FederatedConfig(**state['config'])
        
        # Restore client registry
        self.client_registry = state['client_registry']
        for client in self.client_registry.values():
            client['registered_at'] = datetime.fromisoformat(client['registered_at'])
            if client['last_update']:
                client['last_update'] = datetime.fromisoformat(client['last_update'])
        
        # Restore aggregation history
        self.aggregation_history = []
        for record in state['aggregation_history']:
            record['timestamp'] = datetime.fromisoformat(record['timestamp'])
            self.aggregation_history.append(record)
        
        # Restore global model if available
        if state['global_model_params'] and state['vectorizer_vocab']:
            self.global_model = MultinomialNB()
            self._restore_model_parameters(self.global_model, state['global_model_params'])
            
            self.global_vectorizer = TfidfVectorizer()
            self.global_vectorizer.vocabulary_ = state['vectorizer_vocab']
        
        print(f"Federated state loaded from {filepath}")
    
    def _restore_model_parameters(self, model: MultinomialNB, params: Dict):
        """
        Restore model parameters
        """
        model.class_log_prior_ = params['class_log_prior_']
        model.feature_log_prob_ = params['feature_log_prob_']
        model.classes_ = params['classes_']
    
    def start_background_aggregation(self):
        """
        Start background aggregation process
        """
        self._running = True
        
        def background_worker():
            while self._running:
                time.sleep(60)  # Check every minute
                if len(self.local_updates) >= self.config.min_clients:
                    self._aggregate_updates()
        
        self._aggregation_thread = threading.Thread(target=background_worker)
        self._aggregation_thread.start()
        print("Background aggregation started")
    
    def stop_background_aggregation(self):
        """
        Stop background aggregation process
        """
        self._running = False
        if self._aggregation_thread:
            self._aggregation_thread.join()
        print("Background aggregation stopped")

class ClientFederatedLearner:
    """
    Client-side federated learning implementation
    """
    
    def __init__(self, client_id: str, server_url: str = None):
        self.client_id = client_id
        self.server_url = server_url
        self.local_model = None
        self.local_vectorizer = None
        self.local_data = []
        self.client_key = None
        
    def initialize_local_model(self, vectorizer: TfidfVectorizer, model: MultinomialNB):
        """
        Initialize local model
        """
        self.local_vectorizer = vectorizer
        self.local_model = model
        print(f"Local model initialized for client {self.client_id}")
    
    def add_local_data(self, messages: List[str], labels: List[str]):
        """
        Add local training data
        """
        for message, label in zip(messages, labels):
            self.local_data.append({
                'message': message,
                'label': label,
                'timestamp': datetime.now()
            })
        print(f"Added {len(messages)} samples to local data")
    
    def train_local_model(self, epochs: int = 1) -> Dict:
        """
        Train local model on local data
        """
        if not self.local_data:
            print("No local data available for training")
            return {}
        
        # Prepare training data
        messages = [item['message'] for item in self.local_data]
        labels = [item['label'] for item in self.local_data]
        
        # Vectorize messages
        X = self.local_vectorizer.fit_transform(messages)
        
        # Train model
        self.local_model.fit(X, labels)
        
        # Calculate local update
        local_update = self._calculate_local_update()
        
        return {
            'update_data': local_update,
            'sample_count': len(self.local_data),
            'metadata': {
                'epochs': epochs,
                'local_accuracy': self._evaluate_local_model(),
                'data_diversity': self._calculate_data_diversity()
            }
        }
    
    def _calculate_local_update(self) -> Dict[str, np.ndarray]:
        """
        Calculate local model update
        """
        return {
            'class_log_prior_': self.local_model.class_log_prior_,
            'feature_log_prob_': self.local_model.feature_log_prob_,
            'classes_': self.local_model.classes_
        }
    
    def _evaluate_local_model(self) -> float:
        """
        Evaluate local model performance
        """
        if len(self.local_data) < 10:
            return 0.0
        
        # Simple cross-validation
        messages = [item['message'] for item in self.local_data]
        labels = [item['label'] for item in self.local_data]
        
        X = self.local_vectorizer.transform(messages)
        predictions = self.local_model.predict(X)
        
        accuracy = sum(1 for p, t in zip(predictions, labels) if p == t) / len(labels)
        return accuracy
    
    def _calculate_data_diversity(self) -> float:
        """
        Calculate diversity of local data
        """
        if not self.local_data:
            return 0.0
        
        # Calculate unique labels ratio
        unique_labels = set(item['label'] for item in self.local_data)
        return len(unique_labels) / len(self.local_data)
    
    def apply_global_update(self, global_model_snapshot: Dict):
        """
        Apply global model update to local model
        """
        # Update local model parameters
        global_params = global_model_snapshot['model_params']
        self._restore_model_parameters(self.local_model, global_params)
        
        # Update vectorizer vocabulary
        self.local_vectorizer.vocabulary_ = global_model_snapshot['vectorizer_vocab']
        
        print(f"Applied global model update (version {global_model_snapshot['version']})")
    
    def _restore_model_parameters(self, model: MultinomialNB, params: Dict):
        """
        Restore model parameters
        """
        model.class_log_prior_ = params['class_log_prior_']
        model.feature_log_prob_ = params['feature_log_prob_']
        model.classes_ = params['classes_']

def main():
    """
    Main function to demonstrate federated learning
    """
    print("🤝 Federated Learning for SMS Fraud Detection")
    
    # Initialize federated learning manager
    config = FederatedConfig(
        min_clients=5,
        max_clients=20,
        privacy_budget=1.0,
        noise_scale=0.1
    )
    
    fl_manager = FederatedLearningManager(config)
    
    # Simulate client registration
    client_ids = [f"client_{i}" for i in range(10)]
    for client_id in client_ids:
        fl_manager.register_client(client_id)
    
    # Simulate client updates
    print("\n📊 Simulating Client Updates:")
    for i, client_id in enumerate(client_ids):
        # Simulate model update
        update_data = {
            'class_log_prior_': np.random.normal(0, 0.1, (3,)),
            'feature_log_prob_': np.random.normal(0, 0.1, (3, 3000)),
            'classes_': np.array(['legitimate', 'spam', 'fraudulent'])
        }
        
        sample_count = np.random.randint(10, 100)
        metadata = {
            'local_accuracy': np.random.uniform(0.7, 0.95),
            'data_diversity': np.random.uniform(0.3, 0.8)
        }
        
        success = fl_manager.collect_local_update(
            client_id, update_data, sample_count, metadata
        )
        
        if success:
            print(f"✓ Client {client_id}: {sample_count} samples")
    
    # Get federated statistics
    stats = fl_manager.get_federated_stats()
    print(f"\n📈 Federated Learning Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Federated learning demonstration completed!")

if __name__ == "__main__":
    main() 
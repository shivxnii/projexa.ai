import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings
warnings.filterwarnings('ignore')

class CloudWorkloadSimulator:
    def __init__(self, n_vms=10, n_tasks=5000, time_steps=1000):
        self.n_vms = n_vms
        self.n_tasks = n_tasks
        self.time_steps = time_steps
        
    def generate_synthetic_data(self):
        """Generate synthetic cloud workload data"""
        np.random.seed(42)
        
        data = []
        timestamps = []
        
        for t in range(self.time_steps):
            # Time-based patterns (mimicking daily cycles)
            hour_of_day = t % 24
            day_of_week = (t // 24) % 7
            
            # Workload patterns
            if 9 <= hour_of_day <= 17:  # Business hours
                base_load = abs(np.random.normal(0.7, 0.15))
            elif 18 <= hour_of_day <= 22:  # Evening peak
                base_load = abs(np.random.normal(0.8, 0.2))
            else:  # Night hours
                base_load = abs(np.random.normal(0.3, 0.1))
            
            # Weekend adjustment
            if day_of_week >= 5:  # Weekend
                base_load = max(0.1, base_load * 0.6)
            
            # Add some random bursts
            if np.random.random() < 0.05:  # 5% chance of burst
                base_load = min(1.0, base_load + np.random.uniform(0.2, 0.5))
            
            # Generate VM metrics
            for vm_id in range(self.n_vms):
                # Add some variation between VMs
                vm_variation = 1.0 + (vm_id % 3) * 0.1
                vm_load = base_load * vm_variation
                vm_load = max(0.1, min(1.0, vm_load))
                
                # CPU utilization (0-100%)
                cpu_util = vm_load * 70 + np.random.normal(0, 10)
                cpu_util = max(5, min(100, cpu_util))
                
                # Memory utilization
                mem_util = vm_load * 65 + np.random.normal(0, 15)
                mem_util = max(5, min(100, mem_util))
                
                # Disk I/O
                disk_io = abs(np.random.poisson(vm_load * 1000))
                
                # Network traffic
                network_traffic = abs(np.random.exponential(vm_load * 500))
                
                # Queue length (pending tasks)
                queue_len = abs(np.random.poisson(vm_load * 5))
                
                # Response time (ms)
                response_time = 50 + vm_load * 200 + abs(np.random.normal(0, 30))
                
                # SLA violation risk (binary)
                sla_risk = 1 if (cpu_util > 85 or response_time > 300 or queue_len > 8) else 0
                
                data.append([
                    t, vm_id, cpu_util, mem_util, disk_io, 
                    network_traffic, queue_len, response_time, sla_risk,
                    hour_of_day, day_of_week
                ])
            
            timestamps.append(t)
        
        columns = [
            'timestamp', 'vm_id', 'cpu_util', 'memory_util', 'disk_io',
            'network_traffic', 'queue_length', 'response_time', 'sla_violation_risk',
            'hour_of_day', 'day_of_week'
        ]
        
        return pd.DataFrame(data, columns=columns)

class DynamicLoadBalancingAI:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.models = {}
        
    def create_lstm_model(self, input_dim):
        """Create LSTM-based prediction model"""
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, 
                       input_shape=(self.sequence_length, input_dim)),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)  # Predict only CPU utilization for simplicity
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_decision_model(self, input_dim):
        """Create model for load balancing decisions"""
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Load balancing decision
        ])
        
        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_sequences(self, data, feature_cols, target_col='cpu_util'):
        """Prepare sequences for time series prediction"""
        sequences = []
        targets = []
        
        # Use only selected features
        data_subset = data[feature_cols].values
        
        for i in range(len(data_subset) - self.sequence_length):
            seq = data_subset[i:i+self.sequence_length]
            target = data.iloc[i+self.sequence_length][target_col]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)

class AIBasedLoadBalancer:
    def __init__(self):
        self.simulator = CloudWorkloadSimulator()
        self.ai_model = DynamicLoadBalancingAI()
        self.data = None
        self.trained_models = {}
        self.results = {}
        self.feature_cols = None
        
    def generate_and_prepare_data(self):
        """Generate and preprocess data"""
        print("Generating synthetic cloud workload data...")
        self.data = self.simulator.generate_synthetic_data()
        
        print(f"Data generated: {len(self.data)} records")
        print(f"Sample data shape: {self.data.shape}")
        
        # Add derived features
        self.data['load_score'] = (
            0.4 * (self.data['cpu_util'] / 100) +
            0.3 * (self.data['memory_util'] / 100) +
            0.2 * (self.data['response_time'] / 500) +
            0.1 * (self.data['queue_length'] / 10)
        )
        
        # Define feature columns (excluding timestamp, vm_id, sla_violation_risk)
        self.feature_cols = ['cpu_util', 'memory_util', 'disk_io', 
                           'network_traffic', 'queue_length', 'response_time',
                           'load_score', 'hour_of_day', 'day_of_week']
        
        print(f"\nFeatures for training: {self.feature_cols}")
        print(f"Number of features: {len(self.feature_cols)}")
        
        # Normalize features
        scaled_features = self.ai_model.scaler.fit_transform(self.data[self.feature_cols])
        self.data[self.feature_cols] = scaled_features
        
        print(f"\nData preparation complete!")
        print(f"Feature statistics after normalization:")
        for col in self.feature_cols[:3]:  # Show first 3
            print(f"  {col}: mean={self.data[col].mean():.3f}, std={self.data[col].std():.3f}")
        
        return self.data
    
    def train_prediction_model(self):
        """Train LSTM model for resource prediction"""
        print("\n" + "="*50)
        print("Training Prediction Model (LSTM)")
        print("="*50)
        
        # Prepare sequences for one VM as example
        vm0_data = self.data[self.data['vm_id'] == 0]
        
        print(f"VM 0 data shape: {vm0_data.shape}")
        print(f"Features: {self.feature_cols}")
        print(f"Number of features: {len(self.feature_cols)}")
        
        # Prepare sequences
        sequences, targets = self.ai_model.prepare_sequences(
            vm0_data, self.feature_cols
        )
        
        print(f"\nSequences shape: {sequences.shape}")  # Should be (n_samples, 10, n_features)
        print(f"Targets shape: {targets.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, targets, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining samples: {X_train.shape}")
        print(f"Testing samples: {X_test.shape}")
        
        # Create model with correct input dimension
        input_dim = X_train.shape[2]  # Number of features
        print(f"\nCreating LSTM model with input dimension: {input_dim}")
        
        model = self.ai_model.create_lstm_model(input_dim)
        
        print("\nModel Summary:")
        model.summary()
        
        print("\nStarting model training...")
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=10,  # Reduced for faster training
            batch_size=32,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nModel Evaluation:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Make predictions
        y_pred = model.predict(X_test[:5], verbose=0)
        print(f"\nSample predictions (first 5):")
        for i in range(3):
            print(f"  True: {y_test[i]:.4f}, Predicted: {y_pred[i][0]:.4f}")
        
        self.trained_models['prediction'] = model
        self.results['prediction_metrics'] = {
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'history': history.history,
            'input_dim': input_dim
        }
        
        return model, history
    
    def train_decision_model(self):
        """Train decision model for load balancing"""
        print("\n" + "="*50)
        print("Training Decision Model")
        print("="*50)
        
        # Prepare decision data (balanced vs overloaded)
        decision_data = self.data.copy()
        
        # Use normalized values for threshold
        decision_data['needs_rebalance'] = (
            (decision_data['cpu_util'] > 0.8) |  # scaled threshold
            (decision_data['memory_util'] > 0.75) |
            (decision_data['response_time'] > 0.6)
        ).astype(int)
        
        print(f"Using features: {self.feature_cols}")
        print(f"Number of features: {len(self.feature_cols)}")
        
        X = decision_data[self.feature_cols]
        y = decision_data['needs_rebalance']
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target distribution:")
        target_counts = y.value_counts(normalize=True)
        for value, proportion in target_counts.items():
            print(f"  Class {value}: {proportion:.2%} ({y.value_counts()[value]} samples)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")
        
        # Create model with correct input dimension
        input_dim = X_train.shape[1]
        print(f"\nCreating decision model with input dimension: {input_dim}")
        
        model = self.ai_model.create_decision_model(input_dim)
        
        print("\nModel Summary:")
        model.summary()
        
        # Calculate class weights
        class_0_count = np.sum(y_train == 0)
        class_1_count = np.sum(y_train == 1)
        total = class_0_count + class_1_count
        
        class_weight = {
            0: total / (2 * class_0_count) if class_0_count > 0 else 1.0,
            1: total / (2 * class_1_count) if class_1_count > 0 else 1.0
        }
        
        print(f"\nClass weights: {class_weight}")
        
        print("\nStarting model training...")
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=10,  # Reduced for faster training
            batch_size=64,
            class_weight=class_weight,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nModel Evaluation:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Predictions
        y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
        
        from sklearn.metrics import classification_report, confusion_matrix
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:")
        print(f"[[TN={cm[0,0]} FP={cm[0,1]}]")
        print(f" [FN={cm[1,0]} TP={cm[1,1]}]]")
        
        self.trained_models['decision'] = model
        self.results['decision_metrics'] = {
            'test_accuracy': float(test_acc),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'history': history.history,
            'input_dim': input_dim
        }
        
        return model, history
    
    def simulate_load_balancing(self):
        """Simulate AI-driven load balancing"""
        print("\n" + "="*50)
        print("Simulating AI-Driven Load Balancing")
        print("="*50)
        
        # Group by timestamp to simulate system state
        simulation_results = []
        
        # Use a subset of timestamps for faster simulation
        timestamps = sorted(self.data['timestamp'].unique())
        timestamps = timestamps[:200]  # First 200 timestamps
        
        print(f"Simulating {len(timestamps)} timestamps...")
        
        action_counts = {"Redistribute load": 0, "Maintain current state": 0, "None": 0}
        
        for i, timestamp in enumerate(timestamps):
            timestamp_data = self.data[self.data['timestamp'] == timestamp]
            
            if len(timestamp_data) == 0:
                continue
                
            # Calculate system metrics
            avg_cpu = timestamp_data['cpu_util'].mean()
            avg_memory = timestamp_data['memory_util'].mean()
            avg_response = timestamp_data['response_time'].mean()
            sla_violations = timestamp_data['sla_violation_risk'].sum()
            total_vms = len(timestamp_data)
            
            action = "None"
            
            # Simulate AI decision if model is trained
            if self.trained_models.get('decision') and len(timestamp_data) > 0:
                # Take average features across all VMs for decision
                avg_features = timestamp_data[self.feature_cols].mean().values.reshape(1, -1)
                
                needs_rebalance = self.trained_models['decision'].predict(avg_features, verbose=0)[0][0]
                
                # Simulate load balancing action
                if needs_rebalance > 0.7:
                    action = "Redistribute load"
                    
                    # Simple redistribution simulation
                    overloaded_count = len(timestamp_data[timestamp_data['load_score'] > 0.8])
                    underloaded_count = len(timestamp_data[timestamp_data['load_score'] < 0.3])
                    
                    if overloaded_count > 0 and underloaded_count > 0:
                        # Estimate SLA improvement
                        improvement_factor = min(overloaded_count, underloaded_count) / total_vms
                        sla_violations = max(0, sla_violations * (1 - improvement_factor))
                else:
                    action = "Maintain current state"
            
            action_counts[action] += 1
            
            simulation_results.append({
                'timestamp': timestamp,
                'total_vms': total_vms,
                'avg_cpu_util': avg_cpu,
                'avg_memory_util': avg_memory,
                'avg_response_time': avg_response,
                'sla_violations': sla_violations,
                'action_taken': action
            })
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{len(timestamps)} timestamps")
        
        simulation_df = pd.DataFrame(simulation_results)
        self.results['simulation'] = simulation_df
        
        print(f"\nSimulation complete! {len(simulation_df)} records created.")
        print(f"\nAction Distribution:")
        for action, count in action_counts.items():
            print(f"  {action}: {count} times ({count/len(timestamps)*100:.1f}%)")
        
        return simulation_df
    
    def evaluate_performance(self):
        """Evaluate the performance of the AI model"""
        print("\n" + "="*50)
        print("Evaluating AI Model Performance")
        print("="*50)
        
        simulation_df = self.results.get('simulation')
        
        if simulation_df is not None and len(simulation_df) > 0:
            # Separate timestamps with and without load balancing
            balanced_timestamps = simulation_df[simulation_df['action_taken'] == "Redistribute load"]
            unbalanced_timestamps = simulation_df[simulation_df['action_taken'] != "Redistribute load"]
            
            if len(balanced_timestamps) > 0 and len(unbalanced_timestamps) > 0:
                # Compare metrics
                balanced_avg = balanced_timestamps.mean(numeric_only=True)
                unbalanced_avg = unbalanced_timestamps.mean(numeric_only=True)
                
                metrics = {
                    'cpu_improvement': float(unbalanced_avg['avg_cpu_util'] - balanced_avg['avg_cpu_util']),
                    'memory_improvement': float(unbalanced_avg['avg_memory_util'] - balanced_avg['avg_memory_util']),
                    'response_time_improvement': float(unbalanced_avg['avg_response_time'] - balanced_avg['avg_response_time']),
                    'sla_violation_reduction': float(unbalanced_avg['sla_violations'] - balanced_avg['sla_violations']),
                    'total_actions': int(len(balanced_timestamps)),
                    'action_percentage': float(len(balanced_timestamps) / len(simulation_df) * 100)
                }
                
                print("\nPerformance Comparison:")
                print("-" * 50)
                print(f"{'Metric':<25} {'Without AI':<12} {'With AI':<12} {'Improvement':<12}")
                print("-" * 50)
                print(f"{'Avg CPU Utilization':<25} {unbalanced_avg['avg_cpu_util']:<12.4f} {balanced_avg['avg_cpu_util']:<12.4f} {metrics['cpu_improvement']:<12.4f}")
                print(f"{'Avg Memory Utilization':<25} {unbalanced_avg['avg_memory_util']:<12.4f} {balanced_avg['avg_memory_util']:<12.4f} {metrics['memory_improvement']:<12.4f}")
                print(f"{'Avg Response Time':<25} {unbalanced_avg['avg_response_time']:<12.4f} {balanced_avg['avg_response_time']:<12.4f} {metrics['response_time_improvement']:<12.4f}")
                print(f"{'Avg SLA Violations':<25} {unbalanced_avg['sla_violations']:<12.2f} {balanced_avg['sla_violations']:<12.2f} {metrics['sla_violation_reduction']:<12.2f}")
                print("-" * 50)
                
                print(f"\nAI Actions: {metrics['total_actions']} times ({metrics['action_percentage']:.1f}% of timestamps)")
                
                # Calculate percentage improvements
                print(f"\nPercentage Improvements:")
                print(f"  CPU Utilization: {metrics['cpu_improvement']/abs(unbalanced_avg['avg_cpu_util'])*100:.1f}% better")
                print(f"  Memory Utilization: {metrics['memory_improvement']/abs(unbalanced_avg['avg_memory_util'])*100:.1f}% better")
                print(f"  Response Time: {metrics['response_time_improvement']/abs(unbalanced_avg['avg_response_time'])*100:.1f}% better")
                print(f"  SLA Violations: {metrics['sla_violation_reduction']/max(1, unbalanced_avg['sla_violations'])*100:.1f}% reduction")
                
                self.results['performance_metrics'] = metrics
                
                return metrics
            
            else:
                print("Not enough data for comparison (need both balanced and unbalanced timestamps)")
                return None
        
        print("No simulation data available for evaluation.")
        return None
    
    def save_models(self, path="models/"):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        print(f"\nSaving models to {path}...")
        
        for name, model in self.trained_models.items():
            model_path = f"{path}{name}_model.h5"
            model.save(model_path)
            print(f"Saved: {model_path}")
        
        # Save scaler
        scaler_path = f"{path}scaler.pkl"
        joblib.dump(self.ai_model.scaler, scaler_path)
        print(f"Saved: {scaler_path}")
        
        # Save feature columns
        feature_path = f"{path}feature_columns.txt"
        with open(feature_path, 'w') as f:
            f.write('\n'.join(self.feature_cols))
        print(f"Saved: {feature_path}")
        
        # Save results
        results_path = f"{path}results.json"
        
        # Convert results to serializable format
        results_serializable = {}
        for key, value in self.results.items():
            if isinstance(value, pd.DataFrame):
                results_serializable[key] = value.to_dict()
            elif isinstance(value, dict):
                # Handle history
                if 'history' in value:
                    value['history'] = {k: [float(x) for x in v] 
                                      for k, v in value['history'].items()}
                results_serializable[key] = value
            else:
                results_serializable[key] = str(value)
        
        import json
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Saved: {results_path}")
        print(f"\nAll models and results saved successfully!")
    
    def run_complete_pipeline(self):
        """Run the complete AI model pipeline"""
        print("\n" + "="*60)
        print("AI-Driven Dynamic Load Balancing Model Pipeline")
        print("="*60)
        
        try:
            # Step 1: Generate data
            print("\n[STEP 1] Data Generation")
            self.generate_and_prepare_data()
            
            # Step 2: Train prediction model
            print("\n[STEP 2] Training Prediction Model")
            self.train_prediction_model()
            
            # Step 3: Train decision model
            print("\n[STEP 3] Training Decision Model")
            self.train_decision_model()
            
            # Step 4: Simulate load balancing
            print("\n[STEP 4] Simulation")
            self.simulate_load_balancing()
            
            # Step 5: Evaluate performance
            print("\n[STEP 5] Performance Evaluation")
            self.evaluate_performance()
            
            # Step 6: Save models
            print("\n[STEP 6] Saving Results")
            self.save_models()
            
            print("\n" + "="*60)
            print("Pipeline completed successfully!")
            print("="*60)
            
            return self.results
            
        except Exception as e:
            print(f"\nError in pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def create_visualizations(results, data):
    """Create visualizations for research paper"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(15, 10))
        
        # 1. Workload patterns
        plt.subplot(2, 3, 1)
        hourly_load = data.groupby('hour_of_day')['cpu_util'].mean()
        plt.plot(hourly_load.index, hourly_load.values, marker='o', color='blue')
        plt.title('Average CPU Utilization by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('CPU Utilization (Normalized)')
        plt.grid(True, alpha=0.3)
        
        # 2. Prediction model performance
        plt.subplot(2, 3, 2)
        if 'prediction_metrics' in results:
            history = results['prediction_metrics']['history']
            plt.plot(history['loss'], label='Training Loss', color='red', linewidth=2)
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
            plt.title('Prediction Model Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 3. Decision model accuracy
        plt.subplot(2, 3, 3)
        if 'decision_metrics' in results:
            history = results['decision_metrics']['history']
            plt.plot(history['accuracy'], label='Accuracy', color='green', linewidth=2)
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'], label='Validation Accuracy', color='lightgreen', linewidth=2)
            plt.title('Decision Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. SLA violations over time
        plt.subplot(2, 3, 4)
        if 'simulation' in results:
            simulation_df = results['simulation']
            if isinstance(simulation_df, dict):
                simulation_df = pd.DataFrame(simulation_df)
            
            # Plot with different colors for different actions
            colors = {'Redistribute load': 'green', 'Maintain current state': 'blue', 'None': 'gray'}
            
            for action in colors.keys():
                subset = simulation_df[simulation_df['action_taken'] == action]
                if len(subset) > 0:
                    plt.scatter(subset['timestamp'], subset['sla_violations'], 
                               color=colors[action], alpha=0.6, label=action, s=20)
            
            plt.title('SLA Violations with AI Actions')
            plt.xlabel('Timestamp')
            plt.ylabel('Number of SLA Violations')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. Resource utilization distribution
        plt.subplot(2, 3, 5)
        plt.hist(data['cpu_util'], bins=30, alpha=0.7, label='CPU', color='blue', density=True)
        plt.hist(data['memory_util'], bins=30, alpha=0.7, label='Memory', color='green', density=True)
        plt.title('Resource Utilization Distribution')
        plt.xlabel('Utilization (Normalized)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Performance comparison
        plt.subplot(2, 3, 6)
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            
            # Create comparison bars
            labels = ['CPU', 'Memory', 'Response', 'SLA Violations']
            improvements = [
                metrics.get('cpu_improvement', 0),
                metrics.get('memory_improvement', 0),
                metrics.get('response_time_improvement', 0),
                metrics.get('sla_violation_reduction', 0)
            ]
            
            colors = ['skyblue', 'lightgreen', 'gold', 'lightcoral']
            bars = plt.bar(labels, improvements, color=colors)
            
            # Add value labels
            for bar, value in zip(bars, improvements):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height,
                        f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
            
            plt.title('Performance Improvements with AI')
            plt.ylabel('Improvement (Higher is better)')
            plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('load_balancing_results.png', dpi=300, bbox_inches='tight')
        print("\nVisualizations saved as 'load_balancing_results.png'")
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")

# Main execution
if __name__ == "__main__":
    print("Starting AI-Driven Dynamic Load Balancing System")
    print("Initializing...")
    
    # Initialize and run the complete pipeline
    load_balancer = AIBasedLoadBalancer()
    
    # Run complete pipeline
    results = load_balancer.run_complete_pipeline()
    
    if results:
        # Create visualizations
        print("\nGenerating visualizations...")
        create_visualizations(results, load_balancer.data)
        
        # Print summary for research paper
        print("\n" + "="*60)
        print("RESEARCH PAPER RESULTS SUMMARY")
        print("="*60)
        
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            print(f"\n1. Performance Improvements:")
            print(f"   - CPU Utilization Improvement: {metrics['cpu_improvement']:.4f} ({(metrics['cpu_improvement']/0.8*100):.1f}% relative)")
            print(f"   - Memory Utilization Improvement: {metrics['memory_improvement']:.4f} ({(metrics['memory_improvement']/0.75*100):.1f}% relative)")
            print(f"   - Response Time Reduction: {metrics['response_time_improvement']:.4f} ({(metrics['response_time_improvement']/0.6*100):.1f}% relative)")
            print(f"   - SLA Violations Reduced by: {metrics['sla_violation_reduction']:.2f} instances ({(metrics['sla_violation_reduction']/5*100):.1f}% relative)")
            print(f"   - Load Balancing Actions: {metrics['total_actions']} times ({metrics['action_percentage']:.1f}% of time)")
        
        if 'decision_metrics' in results:
            decision_metrics = results['decision_metrics']['classification_report']
            print(f"\n2. Model Performance:")
            print(f"   - Decision Model Accuracy: {decision_metrics['accuracy']:.2%}")
            print(f"   - Precision: {decision_metrics['1']['precision']:.2%}")
            print(f"   - Recall: {decision_metrics['1']['recall']:.2%}")
            print(f"   - F1-Score: {decision_metrics['1']['f1-score']:.2%}")
        
        if 'prediction_metrics' in results:
            pred_metrics = results['prediction_metrics']
            print(f"\n3. Prediction Model:")
            print(f"   - Test MAE: {pred_metrics['test_mae']:.4f}")
            print(f"   - Test Loss: {pred_metrics['test_loss']:.4f}")
        
        print("\n" + "="*60)
        print("All tasks completed successfully!")
        print("="*60)
        
        # Save final results to CSV for easy access
        if 'simulation' in results:
            if isinstance(results['simulation'], dict):
                df = pd.DataFrame(results['simulation'])
            else:
                df = results['simulation']
            
            df.to_csv('simulation_results.csv', index=False)
            print("\nSimulation results saved to 'simulation_results.csv'")
            
            # Print sample of results
            print("\nSample of simulation results:")
            print(df.head())
    else:
        print("\nPipeline failed to complete successfully.")

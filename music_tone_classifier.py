#!/usr/bin/env python3
"""
Music Tone Classifier using LAION CLAP model
This system can:
1. Classify audio files to tone effects and provide knob settings
2. Match text descriptions to appropriate tone effects and knob settings
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import librosa
from transformers import ClapModel, ClapProcessor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

class KnobParameterNet(nn.Module):
    """Neural Network for predicting all 17 knob parameters with shared representations"""
    
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64], output_dim=17, dropout_rate=0.3):
        super(KnobParameterNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with batch normalization and dropout
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer with sigmoid activation to constrain outputs to [0, 1]
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class MusicToneClassifier:
    def __init__(self, data_dir="Data"):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, "indexes", "index_train.csv")
        self.audio_dir = os.path.join(data_dir, "generated_audio", "train")
        self.json_dir = os.path.join(data_dir, "labels_normalised", "train")
        
        # Load CLAP model
        print("Loading CLAP model...")
        self.clap_model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
        self.clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
        
        # Initialize components
        self.df = None
        self.label_encoder = LabelEncoder()
        self.tone_classifier = None
        self.knob_regressor_net = None  # Neural network for knob parameters
        self.knob_scaler = StandardScaler()  # For input normalization
        self.tone_embeddings = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.knob_params = [
            'distortiondrive', 'overdrivedrive', 'distortiontone', 
            'eqbass', 'eqmid', 'eqtreble', 
            'chorusrate', 'chorusdepth', 'chorusmix', 
            'delaytime', 'delayfeedback', 'delaymix', 
            'reverbt60', 'reverbsize', 'reverbwet', 'reverbdamp', 
            'mastervolume'
        ]
        
        # Text descriptions for each tone type
        self.tone_descriptions = {
            'Clean': [
                "clean guitar tone", "pristine sound", "clear guitar", "clean amp", 
                "transparent tone", "studio clean", "jazz clean tone", "clean electric guitar"
            ],
            'Crunch': [
                "crunchy guitar", "crunch tone", "slight overdrive", "edge of breakup",
                "vintage crunch", "tube crunch", "bluesy crunch", "classic rock crunch"
            ],
            'High-Gain': [
                "heavy metal", "high gain distortion", "metal guitar", "aggressive tone",
                "brutal distortion", "heavy distortion", "modern metal", "saturated distortion"
            ],
            'Wellness': [
                "ambient guitar", "spacey tone", "ethereal guitar", "reverb heavy",
                "atmospheric guitar", "dreamy tone", "shoegaze guitar", "ambient soundscape"
            ]
        }
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} samples with {len(self.df['scenario'].unique())} tone types")
        return self.df
    
    def extract_audio_features(self, audio_path, max_samples=10):
        """Extract CLAP embeddings from audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=48000, duration=10.0)
            
            # Get CLAP audio embedding
            inputs = self.clap_processor(audios=audio, return_tensors="pt", sampling_rate=48000)
            
            with torch.no_grad():
                audio_embed = self.clap_model.get_audio_features(**inputs)
                
            return audio_embed.numpy().flatten()
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros(512)  # Return zero vector if processing fails
    
    def extract_text_features(self, text_list):
        """Extract CLAP embeddings from text descriptions"""
        inputs = self.clap_processor(text=text_list, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            text_embed = self.clap_model.get_text_features(**inputs)
            
        return text_embed.numpy()
    
    def prepare_training_data(self, max_samples_per_class=None):
        """Prepare training data with audio embeddings and target values"""
        print("Preparing training data...")
        
        # Limit samples if specified
        if max_samples_per_class:
            df_sampled = self.df.groupby('scenario').head(max_samples_per_class)
        else:
            df_sampled = self.df
            
        audio_features = []
        tone_labels = []
        knob_values = []
        
        for idx, row in df_sampled.iterrows():
            print(f"Processing {idx+1}/{len(df_sampled)}: {row['scenario']}")
            
            # Extract audio features
            # Handle both old format (audio_path column) and new format (basename column)
            if 'audio_path' in row:
                audio_filename = row['audio_path']
            elif 'basename' in row:
                # New format: basename already contains the filename, just add .wav
                audio_filename = f"{row['basename']}.wav"
            else:
                print(f"Error: No audio path column found in row {idx}")
                continue
                
            audio_path = os.path.join(self.audio_dir, audio_filename)
            
            if os.path.exists(audio_path):
                features = self.extract_audio_features(audio_path)
                audio_features.append(features)
                tone_labels.append(row['scenario'])
                
                # Get knob values from normalized columns
                knob_vals = []
                for param in self.knob_params:
                    knob_vals.append(row[f'n_{param}'])
                knob_values.append(knob_vals)
            else:
                print(f"Audio file not found: {audio_path}")
        
        return np.array(audio_features), np.array(tone_labels), np.array(knob_values)
    
    def train_models(self, max_samples_per_class=20):
        """Train tone classification and knob regression models"""
        print("Training models...")
        
        # Prepare data
        X_audio, y_tones, y_knobs = self.prepare_training_data(max_samples_per_class)
        
        if len(X_audio) == 0:
            raise ValueError("No training data available")
        
        # Encode tone labels
        y_tones_encoded = self.label_encoder.fit_transform(y_tones)
        
        # Split data
        X_train, X_test, y_tone_train, y_tone_test, y_knob_train, y_knob_test = train_test_split(
            X_audio, y_tones_encoded, y_knobs, test_size=0.2, random_state=42, stratify=y_tones_encoded
        )
        
        # Train tone classifier
        from sklearn.ensemble import RandomForestClassifier
        self.tone_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.tone_classifier.fit(X_train, y_tone_train)
        
        # Evaluate tone classifier
        y_pred = self.tone_classifier.predict(X_test)
        print("Tone Classification Report:")
        print(classification_report(y_tone_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Train neural network for knob parameters
        print("Training neural network for knob parameters...")
        self._train_knob_neural_network(X_train, X_test, y_knob_train, y_knob_test)
        
        # Create text embeddings for each tone type
        print("Creating text embeddings for tone matching...")
        for tone_type, descriptions in self.tone_descriptions.items():
            text_embeds = self.extract_text_features(descriptions)
            self.tone_embeddings[tone_type] = np.mean(text_embeds, axis=0)
    
    def _train_knob_neural_network(self, X_train, X_test, y_knob_train, y_knob_test, 
                                  epochs=200, batch_size=16, learning_rate=0.001):
        """Train neural network for knob parameter prediction"""
        
        # Normalize input features
        X_train_scaled = self.knob_scaler.fit_transform(X_train)
        X_test_scaled = self.knob_scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_knob_train).to(self.device)
        y_test_tensor = torch.FloatTensor(y_knob_test).to(self.device)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize neural network
        self.knob_regressor_net = KnobParameterNet(
            input_dim=X_train.shape[1], 
            hidden_dims=[256, 128, 64], 
            output_dim=len(self.knob_params),
            dropout_rate=0.3
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.knob_regressor_net.parameters(), 
                              lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
        
        # Training loop
        self.knob_regressor_net.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training neural network on {self.device}...")
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.knob_regressor_net(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_loss / num_batches
            
            # Validation
            self.knob_regressor_net.eval()
            with torch.no_grad():
                val_predictions = self.knob_regressor_net(X_test_tensor)
                val_loss = criterion(val_predictions, y_test_tensor).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 30:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            self.knob_regressor_net.train()
        
        # Final evaluation
        self.knob_regressor_net.eval()
        with torch.no_grad():
            final_predictions = self.knob_regressor_net(X_test_tensor).cpu().numpy()
            
            print("\nParameter-wise Performance:")
            total_mse = 0
            for i, param in enumerate(self.knob_params):
                param_mse = mean_squared_error(y_knob_test[:, i], final_predictions[:, i])
                total_mse += param_mse
                print(f"{param}: MSE = {param_mse:.6f}")
            
            avg_mse = total_mse / len(self.knob_params)
            print(f"\nAverage MSE across all parameters: {avg_mse:.6f}")
            
            # Calculate R² score for overall model performance
            from sklearn.metrics import r2_score
            r2 = r2_score(y_knob_test, final_predictions)
            print(f"Overall R² Score: {r2:.4f}")
    
    def classify_audio(self, audio_path):
        """Classify audio file and predict knob settings"""
        # Extract audio features
        audio_features = self.extract_audio_features(audio_path)
        audio_features = audio_features.reshape(1, -1)
        
        # Predict tone type
        tone_pred = self.tone_classifier.predict(audio_features)[0]
        tone_name = self.label_encoder.inverse_transform([tone_pred])[0]
        tone_prob = self.tone_classifier.predict_proba(audio_features)[0]
        confidence = np.max(tone_prob)
        
        # Predict knob settings using neural network
        knob_settings = {}
        if self.knob_regressor_net is not None:
            # Normalize input features
            audio_features_scaled = self.knob_scaler.transform(audio_features)
            audio_tensor = torch.FloatTensor(audio_features_scaled).to(self.device)
            
            self.knob_regressor_net.eval()
            with torch.no_grad():
                knob_predictions = self.knob_regressor_net(audio_tensor).cpu().numpy()[0]
            
            for i, param in enumerate(self.knob_params):
                knob_settings[param] = float(knob_predictions[i])  # Already constrained by sigmoid
        else:
            # Fallback to zeros if model not trained
            for param in self.knob_params:
                knob_settings[param] = 0.5
        
        return {
            'tone_type': tone_name,
            'confidence': confidence,
            'knob_settings': knob_settings
        }
    
    def match_text_to_tone(self, text_description):
        """
        Match text description directly to knob parameters using neural network.
        This allows for infinite variations based on text input, not just 4 predefined tones.
        """
        # Extract text embedding from CLAP
        text_embed = self.extract_text_features([text_description])[0]
        text_embed = text_embed.reshape(1, -1)
        
        # Predict knob settings directly using neural network
        knob_settings = {}
        if self.knob_regressor_net is not None:
            # Normalize input features (same as audio)
            text_embed_scaled = self.knob_scaler.transform(text_embed)
            text_tensor = torch.FloatTensor(text_embed_scaled).to(self.device)
            
            self.knob_regressor_net.eval()
            with torch.no_grad():
                knob_predictions = self.knob_regressor_net(text_tensor).cpu().numpy()[0]
            
            for i, param in enumerate(self.knob_params):
                knob_settings[param] = float(knob_predictions[i])
        else:
            # Fallback: use tone classification if neural network not available
            return self._match_text_to_tone_fallback(text_description)
        
        # Also calculate tone type for reference (optional)
        similarities = {}
        for tone_type, tone_embed in self.tone_embeddings.items():
            similarity = np.dot(text_embed.flatten(), tone_embed) / (
                np.linalg.norm(text_embed) * np.linalg.norm(tone_embed)
            )
            similarities[tone_type] = similarity
        
        best_tone = max(similarities, key=similarities.get)
        confidence = similarities[best_tone]
        
        return {
            'tone_type': best_tone,  # For reference only
            'confidence': confidence,
            'knob_settings': knob_settings,
            'all_similarities': similarities
        }
    
    def _match_text_to_tone_fallback(self, text_description):
        """Fallback method using tone classification (old approach)"""
        text_embed = self.extract_text_features([text_description])[0]
        
        similarities = {}
        for tone_type, tone_embed in self.tone_embeddings.items():
            similarity = np.dot(text_embed, tone_embed) / (
                np.linalg.norm(text_embed) * np.linalg.norm(tone_embed)
            )
            similarities[tone_type] = similarity
        
        best_tone = max(similarities, key=similarities.get)
        confidence = similarities[best_tone]
        
        # Get average knob settings for this tone type
        tone_samples = self.df[self.df['scenario'] == best_tone]
        knob_settings = {}
        for param in self.knob_params:
            avg_value = tone_samples[f'n_{param}'].mean()
            knob_settings[param] = avg_value
        
        return {
            'tone_type': best_tone,
            'confidence': confidence,
            'knob_settings': knob_settings,
            'all_similarities': similarities
        }
    
    def save_models(self, save_dir="models"):
        """Save trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        with open(os.path.join(save_dir, 'tone_classifier.pkl'), 'wb') as f:
            pickle.dump(self.tone_classifier, f)
        
        with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save neural network state dict
        if self.knob_regressor_net is not None:
            torch.save(self.knob_regressor_net.state_dict(), 
                      os.path.join(save_dir, 'knob_regressor_net.pth'))
        
        # Save scaler
        with open(os.path.join(save_dir, 'knob_scaler.pkl'), 'wb') as f:
            pickle.dump(self.knob_scaler, f)
        
        with open(os.path.join(save_dir, 'tone_embeddings.pkl'), 'wb') as f:
            pickle.dump(self.tone_embeddings, f)
        
        print(f"Models saved to {save_dir}/")
    
    def load_models(self, save_dir="models"):
        """Load pre-trained models"""
        with open(os.path.join(save_dir, 'tone_classifier.pkl'), 'rb') as f:
            self.tone_classifier = pickle.load(f)
        
        with open(os.path.join(save_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load neural network
        net_path = os.path.join(save_dir, 'knob_regressor_net.pth')
        if os.path.exists(net_path):
            self.knob_regressor_net = KnobParameterNet(
                input_dim=512, 
                hidden_dims=[256, 128, 64], 
                output_dim=len(self.knob_params)
            ).to(self.device)
            self.knob_regressor_net.load_state_dict(torch.load(net_path, map_location=self.device))
            self.knob_regressor_net.eval()
        
        # Load scaler
        with open(os.path.join(save_dir, 'knob_scaler.pkl'), 'rb') as f:
            self.knob_scaler = pickle.load(f)
        
        with open(os.path.join(save_dir, 'tone_embeddings.pkl'), 'rb') as f:
            self.tone_embeddings = pickle.load(f)
        
        # Load dataset for text-to-tone matching
        if self.df is None:
            self.load_data()
        
        print(f"Models loaded from {save_dir}/")

def main():
    # Initialize classifier
    classifier = MusicToneClassifier()
    
    # Load data
    classifier.load_data()
    
    # Train models (using subset for faster training)
    classifier.train_models(max_samples_per_class=15)
    
    # Save models
    classifier.save_models()
    
    print("\nTraining completed! Models saved.")
    
    # Test with a sample audio file
    if len(classifier.df) > 0:
        sample_audio = os.path.join(classifier.audio_dir, classifier.df.iloc[0]['audio_path'])
        if os.path.exists(sample_audio):
            print(f"\nTesting with sample audio: {sample_audio}")
            result = classifier.classify_audio(sample_audio)
            print(f"Predicted tone: {result['tone_type']} (confidence: {result['confidence']:.3f})")
            print("Knob settings:")
            for param, value in result['knob_settings'].items():
                print(f"  {param}: {value:.3f}")
    
    # Test text matching
    print(f"\nTesting text matching with 'heavy metal':")
    text_result = classifier.match_text_to_tone("heavy metal")
    print(f"Matched tone: {text_result['tone_type']} (confidence: {text_result['confidence']:.3f})")
    print("Recommended knob settings:")
    for param, value in text_result['knob_settings'].items():
        print(f"  {param}: {value:.3f}")

if __name__ == "__main__":
    main()

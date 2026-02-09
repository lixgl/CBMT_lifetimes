"""
CBMT-lifetimes: Customer-Base Multi-Task Transformer for Cohort Analysis

A unified package for cohort-based customer lifetime value analysis using 
a multi-task Transformer model as described in the research paper 
"Multi-Task Learning for Customer Base Analysis".

Main Functions:
    - visualize_cohorts(): Visualize ROPC or AOV trends from raw transaction data
    - train_and_forecast(): Train CBMT model and generate forecasts
    - visualize_forecast(): Visualize historical + forecasted metrics

Usage:
    from CBMT_lifetimes import CohortLifetimes
    
    pipeline = CohortLifetimes()
    
    # Explore raw data
    pipeline.visualize_cohorts(df, metric='ropc')
    
    # Train & forecast
    results = pipeline.train_and_forecast(df, forecast_weeks=13)
    
    # Visualize results
    pipeline.visualize_forecast(results, metric='total_revenue')
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CBMTConfig:
    """Configuration for the CBMT model."""
    
    # Model architecture
    d_model: int = 32
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    ff_hidden: int = 64
    dropout: float = 0.1
    lookback_window: int = 20
    
    # Training
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    lr_shared: float = 1e-4
    lr_heads: float = 1e-3
    
    # Task weights
    w_acquisition: float = 4.0
    w_ropc: float = 4.0
    w_aov: float = 1.0
    w_sales: float = 4.0
    w_consistency: float = 1.0
    
    # Data splits (weeks)
    validation_weeks: int = 12
    holdout_weeks: int = 12
    forecast_horizon: int = 13
    
    # Compute
    device: str = "auto"
    
    # Numerical features to normalize
    numerical_features: List[str] = field(default_factory=lambda: [
        'tenure', 'acquisitions', 'repeat_orders', 'ropc', 'aov',
        'total_orders', 'total_revenue', 'calendar_trend', 
        'calendar_trend_sq', 'tenure_sq'
    ])
    
    # Categorical features for embedding
    categorical_features: List[str] = field(default_factory=lambda: [
        'week_of_year', 'acquisition_month', 'cohort_id'
    ])
    
    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# =============================================================================
# Data Wrangling
# =============================================================================

class DataWrangler:
    """Transforms raw transaction data into cohort panel format."""
    
    def __init__(self):
        self._cohort_initial_acquisitions: Dict = {}
    
    def wrangle(
        self,
        df: pd.DataFrame,
        customer_id: str = 'customer_id',
        transaction_date: str = 'transaction_date',
        transaction_amount: str = 'amount',
        binary_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert transaction-level data to cohort-week panel.
        
        Args:
            df: Raw transaction data
            customer_id: Column name for customer identifier
            transaction_date: Column name for transaction date
            transaction_amount: Column name for transaction amount
            binary_features: Optional list of binary feature columns to pass through
        """
        df = df.copy()
        df[transaction_date] = pd.to_datetime(df[transaction_date])
        
        # Assign cohort (week of first purchase)
        first_purchase = df.groupby(customer_id)[transaction_date].min().reset_index()
        first_purchase.columns = [customer_id, 'first_purchase_date']
        first_purchase['cohort_week'] = first_purchase['first_purchase_date'].dt.to_period('W').dt.start_time
        
        df = df.merge(first_purchase[[customer_id, 'cohort_week']], on=customer_id)
        df['calendar_week'] = df[transaction_date].dt.to_period('W').dt.start_time
        
        # Get all unique cohorts and calendar weeks
        cohorts = sorted(df['cohort_week'].unique())
        calendar_weeks = sorted(df['calendar_week'].unique())
        
        # Create dense panel
        panel_rows = []
        
        for cohort in cohorts:
            cohort_customers = df[df['cohort_week'] == cohort][customer_id].unique()
            n_acquired = len(cohort_customers)
            self._cohort_initial_acquisitions[cohort] = n_acquired
            
            for cal_week in calendar_weeks:
                if cal_week < cohort:
                    continue
                    
                tenure = (cal_week - cohort).days // 7
                
                # Filter transactions for this cohort-week
                mask = (df['cohort_week'] == cohort) & (df['calendar_week'] == cal_week)
                week_data = df[mask]
                
                # Acquisitions (only at tenure=0)
                acquisitions = n_acquired if tenure == 0 else 0
                
                # Repeat orders and metrics
                if len(week_data) > 0:
                    unique_customers = week_data[customer_id].nunique()
                    total_orders = len(week_data)
                    repeat_orders = total_orders - acquisitions if tenure == 0 else total_orders
                    total_revenue = week_data[transaction_amount].sum()
                    aov = total_revenue / total_orders if total_orders > 0 else 0
                else:
                    unique_customers = 0
                    total_orders = 0
                    repeat_orders = 0
                    total_revenue = 0
                    aov = 0
                
                ropc = repeat_orders / n_acquired if n_acquired > 0 else 0
                
                row = {
                    'cohort_week': cohort,
                    'calendar_week': cal_week,
                    'tenure': tenure,
                    'acquisitions': acquisitions,
                    'repeat_orders': repeat_orders,
                    'ropc': ropc,
                    'aov': aov,
                    'total_orders': total_orders,
                    'total_revenue': total_revenue,
                    'week_of_year': cal_week.isocalendar().week,
                    'acquisition_month': cohort.month,
                }
                
                # Add binary features if provided
                if binary_features:
                    for feat in binary_features:
                        if feat in week_data.columns and len(week_data) > 0:
                            row[feat] = int(week_data[feat].max())
                        else:
                            row[feat] = 0
                
                panel_rows.append(row)
        
        panel = pd.DataFrame(panel_rows)
        
        # Add derived features
        min_week = panel['calendar_week'].min()
        panel['calendar_trend'] = (panel['calendar_week'] - min_week).dt.days // 7
        panel['calendar_trend_sq'] = panel['calendar_trend'] ** 2
        panel['tenure_sq'] = panel['tenure'] ** 2
        
        # Assign cohort IDs
        cohort_to_id = {c: i + 1 for i, c in enumerate(cohorts)}
        panel['cohort_id'] = panel['cohort_week'].map(cohort_to_id)
        
        return panel.sort_values(['cohort_week', 'calendar_week']).reset_index(drop=True)
    
    def get_cohort_acquisitions(self) -> Dict:
        return self._cohort_initial_acquisitions.copy()


# =============================================================================
# Normalization
# =============================================================================

class Normalizer:
    """Min-max normalization for numerical features."""
    
    def __init__(self):
        self._params: Dict[str, Dict[str, float]] = {}
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'Normalizer':
        for col in columns:
            if col in df.columns:
                self._params[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Normalizer must be fit before transform")
        
        result = df.copy()
        for col, params in self._params.items():
            if col in result.columns:
                range_val = params['max'] - params['min']
                if range_val > 0:
                    result[col] = (result[col] - params['min']) / range_val
                else:
                    result[col] = 0.0
        return result
    
    def inverse_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        result = df.copy()
        for col in columns:
            if col in result.columns and col in self._params:
                params = self._params[col]
                result[col] = result[col] * (params['max'] - params['min']) + params['min']
        return result
    
    def get_params(self) -> Dict:
        return self._params.copy()
    
    def set_params(self, params: Dict):
        self._params = params.copy()
        self._is_fitted = True


# =============================================================================
# PyTorch Dataset
# =============================================================================

class CohortDataset(Dataset):
    """PyTorch Dataset for cohort panel data."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: CBMTConfig,
        embedding_dims: Dict[str, int],
        is_train: bool = True
    ):
        self.config = config
        self.embedding_dims = embedding_dims
        
        df = df.copy()
        df['cohort_week'] = pd.to_datetime(df['cohort_week'])
        df['calendar_week'] = pd.to_datetime(df['calendar_week'])
        df = df.sort_values(['cohort_week', 'calendar_week']).reset_index(drop=True)
        
        self.cohorts = sorted(df['cohort_week'].unique())
        self.cohort_to_id = {c: i + 1 for i, c in enumerate(self.cohorts)}
        
        self.samples = []
        self.data_by_cohort = {}
        self._prepare_data(df, is_train)
    
    def _prepare_data(self, df: pd.DataFrame, is_train: bool):
        lookback = self.config.lookback_window
        
        for cohort in self.cohorts:
            cohort_data = df[df['cohort_week'] == cohort].sort_values('calendar_week')
            
            # Numerical features
            num_cols = [c for c in self.config.numerical_features if c in cohort_data.columns]
            numerical = torch.tensor(cohort_data[num_cols].values, dtype=torch.float32)
            
            # Categorical features
            cat_data = []
            for col in self.config.categorical_features:
                if col in cohort_data.columns:
                    cat_data.append(cohort_data[col].values)
            categorical = torch.tensor(np.array(cat_data).T, dtype=torch.long) if cat_data else None
            
            # Targets
            targets = torch.tensor(
                cohort_data[['acquisitions', 'ropc', 'aov', 'total_revenue']].values,
                dtype=torch.float32
            )
            
            self.data_by_cohort[cohort] = {
                'numerical': numerical,
                'categorical': categorical,
                'targets': targets
            }
            
            # Build samples
            n_weeks = len(cohort_data)
            for end_idx in range(lookback, n_weeks):
                self.samples.append((cohort, end_idx))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cohort, end_idx = self.samples[idx]
        lookback = self.config.lookback_window
        data = self.data_by_cohort[cohort]
        start_idx = end_idx - lookback
        
        numerical_seq = data['numerical'][start_idx:end_idx]
        categorical_seq = data['categorical'][start_idx:end_idx] if data['categorical'] is not None else torch.zeros(lookback, 3, dtype=torch.long)
        target = data['targets'][end_idx]
        cohort_id = torch.tensor([self.cohort_to_id[cohort]], dtype=torch.long)
        
        return numerical_seq, categorical_seq, target, cohort_id


# =============================================================================
# CBMT Model
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CBMTModel(nn.Module):
    """Customer-Base Multi-Task Transformer model."""
    
    def __init__(self, config: CBMTConfig, embedding_dims: Dict[str, int], numerical_dim: int):
        super().__init__()
        self.config = config
        
        # Embedding layers with dynamic dimensions
        self.week_embed = nn.Embedding(54, embedding_dims.get('week_of_year', 8))
        self.month_embed = nn.Embedding(13, embedding_dims.get('acquisition_month', 4))
        self.cohort_embed = nn.Embedding(
            embedding_dims.get('cohort_id_cardinality', 500) + 1,
            embedding_dims.get('cohort_id', 13)
        )
        
        # Calculate total input dimension
        total_embed_dim = (
            embedding_dims.get('week_of_year', 8) +
            embedding_dims.get('acquisition_month', 4) +
            embedding_dims.get('cohort_id', 13)
        )
        input_dim = numerical_dim + total_embed_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, config.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, dropout=config.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_hidden,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_hidden,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)
        
        # Task heads (3-layer FFNNs)
        def make_head():
            return nn.Sequential(
                nn.Linear(config.d_model, config.ff_hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.ff_hidden, config.ff_hidden // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.ff_hidden // 2, 1)
            )
        
        self.acq_head = make_head()
        self.ropc_head = make_head()
        self.aov_head = make_head()
        self.sales_head = make_head()
    
    def forward(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = numerical.shape
        
        # Embeddings
        week_emb = self.week_embed(categorical[:, :, 0].clamp(0, 53))
        month_emb = self.month_embed(categorical[:, :, 1].clamp(0, 12))
        cohort_emb = self.cohort_embed(categorical[:, :, 2].clamp(0, self.cohort_embed.num_embeddings - 1))
        
        # Concatenate
        x = torch.cat([numerical, week_emb, month_emb, cohort_emb], dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # Encoder
        memory = self.encoder(x)
        
        # Decoder
        decoder_input = x[:, -1:, :]
        decoded = self.decoder(decoder_input, memory)
        decoded = decoded.squeeze(1)
        
        # Task heads
        return (
            self.acq_head(decoded),
            self.ropc_head(decoded),
            self.aov_head(decoded),
            self.sales_head(decoded)
        )


# =============================================================================
# Main Class
# =============================================================================

class CohortLifetimes:
    """
    Customer-Base Multi-Task Transformer for Cohort Lifetime Analysis.
    
    A unified interface for:
    1. Visualizing cohort metrics from raw transaction data
    2. Training a multi-task Transformer and generating forecasts
    3. Visualizing historical + forecasted metrics
    """
    
    def __init__(self):
        self._wrangler = DataWrangler()
        self._normalizer = Normalizer()
        self._config = CBMTConfig()
        self._model: Optional[CBMTModel] = None
        self._embedding_dims: Dict[str, int] = {}
        self._is_fitted = False
        self._panel_data: Optional[pd.DataFrame] = None
        self._last_historical_week: Optional[pd.Timestamp] = None
    
    # =========================================================================
    # Function 1: Visualize Cohorts (Pre-Model)
    # =========================================================================
    
    def visualize_cohorts(
        self,
        data: pd.DataFrame,
        customer_id: str = 'customer_id',
        transaction_date: str = 'transaction_date',
        transaction_amount: str = 'amount',
        metric: str = 'ropc',
        cohorts: Optional[List[str]] = None,
        date_range: Optional[Tuple[str, str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize ROPC or AOV trends from raw transaction data.
        
        Args:
            data: Raw transaction DataFrame
            customer_id: Column name for customer identifier
            transaction_date: Column name for transaction date
            transaction_amount: Column name for transaction amount
            metric: 'ropc' or 'aov'
            cohorts: List of cohort dates to display (None = all)
            date_range: Tuple of (start_date, end_date) for calendar weeks
            save_path: Optional path to save image
            
        Returns:
            matplotlib Figure object
        """
        # Wrangle data
        panel = self._wrangler.wrangle(
            data,
            customer_id=customer_id,
            transaction_date=transaction_date,
            transaction_amount=transaction_amount
        )
        
        return self._create_spaghetti_chart(
            panel, metric, cohorts, date_range, save_path, is_forecast=False
        )
    
    # =========================================================================
    # Function 2: Train and Forecast
    # =========================================================================
    
    def train_and_forecast(
        self,
        data: pd.DataFrame,
        customer_id: str = 'customer_id',
        transaction_date: str = 'transaction_date',
        transaction_amount: str = 'amount',
        binary_features: Optional[Union[str, List[str]]] = None,
        forecast_weeks: int = 13,
        train_weeks: Optional[int] = None,
        val_weeks: int = 12,
        holdout_weeks: int = 12,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Train CBMT model and generate forecasts.
        
        Args:
            data: Raw transaction DataFrame
            customer_id: Column name for customer identifier
            transaction_date: Column name for transaction date
            transaction_amount: Column name for transaction amount
            binary_features: Optional binary feature column(s)
            forecast_weeks: Number of weeks to forecast
            train_weeks: Training period weeks (None = auto)
            val_weeks: Validation period weeks
            holdout_weeks: Holdout period weeks
            save_path: Optional path to save combined CSV
            
        Returns:
            Combined DataFrame with historical + forecast data
        """
        # Handle binary features
        if isinstance(binary_features, str):
            binary_features = [binary_features]
        
        # Wrangle data
        print("Wrangling transaction data...")
        panel = self._wrangler.wrangle(
            data,
            customer_id=customer_id,
            transaction_date=transaction_date,
            transaction_amount=transaction_amount,
            binary_features=binary_features
        )
        
        self._panel_data = panel.copy()
        
        # Update config
        self._config.validation_weeks = val_weeks
        self._config.holdout_weeks = holdout_weeks
        self._config.forecast_horizon = forecast_weeks
        
        # Add binary features to numerical features if provided
        if binary_features:
            for feat in binary_features:
                if feat not in self._config.numerical_features:
                    self._config.numerical_features.append(feat)
        
        # Compute dynamic embedding dimensions
        self._compute_embedding_dims(panel)
        
        # Split data
        train_df, val_df = self._split_data(panel, val_weeks, holdout_weeks)
        
        # Normalize
        print("Normalizing features...")
        self._normalizer.fit(train_df, self._config.numerical_features)
        train_normalized = self._normalizer.transform(train_df)
        
        # Combine train+val for validation with lookback
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        train_val_normalized = self._normalizer.transform(train_val_df)
        
        # Create datasets
        train_dataset = CohortDataset(train_normalized, self._config, self._embedding_dims, is_train=True)
        val_dataset = self._create_val_dataset(train_val_normalized, val_df['calendar_week'].unique())
        
        # Train
        print(f"\nTraining CBMT model...")
        self._train_model(train_dataset, val_dataset)
        
        # Store last historical week
        self._last_historical_week = panel['calendar_week'].max()
        
        # Generate forecasts
        print(f"\nGenerating {forecast_weeks}-week forecast...")
        forecast_df = self._predict(panel, forecast_weeks)
        
        # Combine
        panel['is_forecast'] = 0
        forecast_df['is_forecast'] = 1
        combined = pd.concat([panel, forecast_df], ignore_index=True)
        combined = combined.sort_values(['cohort_week', 'calendar_week']).reset_index(drop=True)
        
        if save_path:
            combined.to_csv(save_path, index=False)
            print(f"Results saved to: {save_path}")
        
        return combined
    
    # =========================================================================
    # Function 3: Visualize Forecast
    # =========================================================================
    
    def visualize_forecast(
        self,
        data: pd.DataFrame,
        metric: str = 'total_revenue',
        cohorts: Optional[List[str]] = None,
        date_range: Optional[Tuple[str, str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize historical + forecasted metrics with split line.
        
        Args:
            data: Combined DataFrame from train_and_forecast()
            metric: 'ropc', 'aov', or 'total_revenue'
            cohorts: List of cohort dates to display (None = all)
            date_range: Tuple of (start_date, end_date) for calendar weeks
            save_path: Optional path to save image
            
        Returns:
            matplotlib Figure object
        """
        return self._create_spaghetti_chart(
            data, metric, cohorts, date_range, save_path, is_forecast=True
        )
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _compute_embedding_dims(self, df: pd.DataFrame):
        """Compute embedding dimensions as ceil(sqrt(cardinality))."""
        self._embedding_dims = {
            'week_of_year': max(2, int(np.ceil(np.sqrt(53)))),  # 8
            'acquisition_month': max(2, int(np.ceil(np.sqrt(12)))),  # 4
            'cohort_id': max(2, int(np.ceil(np.sqrt(df['cohort_id'].nunique())))),
            'cohort_id_cardinality': df['cohort_id'].max()
        }
        print(f"Embedding dimensions: {self._embedding_dims}")
    
    def _split_data(
        self,
        df: pd.DataFrame,
        val_weeks: int,
        holdout_weeks: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and validation sets."""
        df = df.copy()
        df['calendar_week'] = pd.to_datetime(df['calendar_week'])
        
        weeks = sorted(df['calendar_week'].unique())
        n_weeks = len(weeks)
        
        holdout_start = n_weeks - holdout_weeks
        val_start = holdout_start - val_weeks
        
        train_weeks_list = weeks[:val_start]
        val_weeks_list = weeks[val_start:holdout_start]
        
        train_df = df[df['calendar_week'].isin(train_weeks_list)]
        val_df = df[df['calendar_week'].isin(val_weeks_list)]
        
        print(f"Data split: Train={len(train_weeks_list)} weeks, Val={len(val_weeks_list)} weeks, Holdout={holdout_weeks} weeks")
        
        return train_df, val_df
    
    def _create_val_dataset(self, df: pd.DataFrame, val_weeks: np.ndarray) -> CohortDataset:
        """Create validation dataset with proper lookback history."""
        # This is a simplified version - uses full history for lookback
        return CohortDataset(df, self._config, self._embedding_dims, is_train=False)
    
    def _train_model(self, train_dataset: CohortDataset, val_dataset: CohortDataset):
        """Train the CBMT model."""
        device = self._config.get_device()
        print(f"Training on device: {device}")
        
        # Get numerical dimension
        num_cols = [c for c in self._config.numerical_features 
                    if c in self._panel_data.columns]
        numerical_dim = len(num_cols)
        
        # Initialize model
        self._model = CBMTModel(self._config, self._embedding_dims, numerical_dim).to(device)
        
        # Optimizer
        optimizer = torch.optim.AdamW([
            {'params': self._model.encoder.parameters(), 'lr': self._config.lr_shared},
            {'params': self._model.decoder.parameters(), 'lr': self._config.lr_shared},
            {'params': self._model.acq_head.parameters(), 'lr': self._config.lr_heads},
            {'params': self._model.ropc_head.parameters(), 'lr': self._config.lr_heads},
            {'params': self._model.aov_head.parameters(), 'lr': self._config.lr_heads},
            {'params': self._model.sales_head.parameters(), 'lr': self._config.lr_heads},
        ])
        
        train_loader = DataLoader(train_dataset, batch_size=self._config.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self._config.batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        for epoch in range(self._config.max_epochs):
            # Training
            self._model.train()
            train_losses = []
            
            for numerical, categorical, targets, _ in train_loader:
                numerical = numerical.to(device)
                categorical = categorical.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                predictions = self._model(numerical, categorical)
                loss = self._compute_loss(predictions, targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            self._model.eval()
            val_losses = []
            
            with torch.no_grad():
                for numerical, categorical, targets, _ in val_loader:
                    numerical = numerical.to(device)
                    categorical = categorical.to(device)
                    targets = targets.to(device)
                    predictions = self._model(numerical, categorical)
                    loss = self._compute_loss(predictions, targets)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or patience_counter == 0:
                print(f"Epoch {epoch + 1:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
            
            if patience_counter >= self._config.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Restore best model
        if best_state:
            self._model.load_state_dict(best_state)
        
        self._is_fitted = True
        print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    
    def _compute_loss(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute multi-task loss."""
        acq_pred, ropc_pred, aov_pred, sales_pred = predictions
        
        acq_true = targets[:, 0:1]
        ropc_true = targets[:, 1:2]
        aov_true = targets[:, 2:3]
        sales_true = targets[:, 3:4]
        
        mse = nn.MSELoss()
        
        l_acq = mse(acq_pred, acq_true)
        l_ropc = mse(ropc_pred, ropc_true)
        l_aov = mse(aov_pred, aov_true)
        l_sales = mse(sales_pred, sales_true)
        
        # Consistency loss
        implied_sales = acq_pred * ropc_pred * aov_pred
        l_consistency = mse(implied_sales, sales_true)
        
        return (
            self._config.w_acquisition * l_acq +
            self._config.w_ropc * l_ropc +
            self._config.w_aov * l_aov +
            self._config.w_sales * l_sales +
            self._config.w_consistency * l_consistency
        )
    
    def _predict(self, df: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
        """Generate forecasts using walk-forward inference."""
        device = self._config.get_device()
        self._model.eval()
        
        df_original = df.copy()
        df_original['cohort_week'] = pd.to_datetime(df_original['cohort_week'])
        df_original['calendar_week'] = pd.to_datetime(df_original['calendar_week'])
        
        df_normalized = self._normalizer.transform(df)
        df_normalized['cohort_week'] = pd.to_datetime(df_normalized['cohort_week'])
        df_normalized['calendar_week'] = pd.to_datetime(df_normalized['calendar_week'])
        
        last_date = df_normalized['calendar_week'].max()
        cohorts = sorted(df_normalized['cohort_week'].unique())
        cohort_acq = self._wrangler.get_cohort_acquisitions()
        
        forecast_weeks = [last_date + pd.Timedelta(weeks=i+1) for i in range(horizon_weeks)]
        num_cols = [c for c in self._config.numerical_features if c in df_normalized.columns]
        cat_cols = self._config.categorical_features
        
        forecast_rows = []
        
        with torch.no_grad():
            for forecast_week in forecast_weeks:
                week_of_year = forecast_week.isocalendar().week
                
                for cohort in cohorts:
                    cohort_data = df_normalized[df_normalized['cohort_week'] == cohort].copy()
                    cohort_start = pd.Timestamp(cohort)
                    tenure = (forecast_week - cohort_start).days // 7
                    
                    if tenure < 0 or len(cohort_data) < self._config.lookback_window:
                        continue
                    
                    window_data = cohort_data.tail(self._config.lookback_window)
                    
                    numerical = torch.tensor(window_data[num_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
                    categorical = torch.tensor(window_data[cat_cols].values, dtype=torch.long).unsqueeze(0).to(device)
                    
                    acq_norm, ropc_norm, aov_norm, revenue_norm = self._model(numerical, categorical)
                    
                    forecast_rows.append({
                        'cohort_week': cohort,
                        'calendar_week': forecast_week,
                        'tenure': tenure,
                        'acquisitions': acq_norm.item(),
                        'ropc': ropc_norm.item(),
                        'aov': aov_norm.item(),
                        'total_revenue': revenue_norm.item(),
                        'week_of_year': week_of_year,
                        'acquisition_month': cohort_start.month,
                        'cohort_id': cohorts.index(cohort) + 1
                    })
                
                # Add to normalized df for next iteration
                if forecast_rows:
                    new_rows = [r for r in forecast_rows if r['calendar_week'] == forecast_week]
                    if new_rows:
                        new_df = pd.DataFrame(new_rows)
                        for col in df_normalized.columns:
                            if col not in new_df.columns:
                                new_df[col] = 0
                        df_normalized = pd.concat([df_normalized, new_df[df_normalized.columns]], ignore_index=True)
        
        forecast_df = pd.DataFrame(forecast_rows)
        
        if len(forecast_df) == 0:
            return pd.DataFrame()
        
        # Inverse transform
        forecast_df = self._normalizer.inverse_transform(
            forecast_df, ['acquisitions', 'ropc', 'aov', 'total_revenue']
        )
        
        # Force acquisitions=0 for tenure>0
        forecast_df.loc[forecast_df['tenure'] > 0, 'acquisitions'] = 0
        
        # Derive repeat_orders using cohort's initial acquisitions
        forecast_df['cohort_initial_acq'] = forecast_df['cohort_week'].map(cohort_acq).fillna(0)
        forecast_df['repeat_orders'] = forecast_df['cohort_initial_acq'] * forecast_df['ropc']
        forecast_df['total_orders'] = forecast_df['cohort_initial_acq'] + forecast_df['repeat_orders']
        forecast_df = forecast_df.drop(columns=['cohort_initial_acq'])
        
        # Add derived columns
        min_week = df_original['calendar_week'].min()
        forecast_df['calendar_trend'] = (forecast_df['calendar_week'] - min_week).dt.days // 7
        forecast_df['calendar_trend_sq'] = forecast_df['calendar_trend'] ** 2
        forecast_df['tenure_sq'] = forecast_df['tenure'] ** 2
        
        print(f"Generated {len(forecast_df)} forecast rows")
        
        return forecast_df
    
    def _create_spaghetti_chart(
        self,
        df: pd.DataFrame,
        metric: str,
        cohorts: Optional[List[str]],
        date_range: Optional[Tuple[str, str]],
        save_path: Optional[str],
        is_forecast: bool
    ) -> plt.Figure:
        """Create neon spaghetti chart visualization."""
        df = df.copy()
        df['cohort_week'] = pd.to_datetime(df['cohort_week'])
        df['calendar_week'] = pd.to_datetime(df['calendar_week'])
        
        # Filter cohorts
        if cohorts:
            cohorts_dt = [pd.to_datetime(c) for c in cohorts]
            df = df[df['cohort_week'].isin(cohorts_dt)]
        
        # Filter date range
        if date_range:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df[(df['calendar_week'] >= start) & (df['calendar_week'] <= end)]
        
        all_cohorts = sorted(df['cohort_week'].unique())
        n_cohorts = len(all_cohorts)
        
        # Color scheme
        cmap = plt.cm.plasma
        colors = [cmap(i / max(1, n_cohorts - 1)) for i in range(n_cohorts)]
        
        # Dark theme
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')
        
        for idx, cohort in enumerate(all_cohorts):
            cohort_data = df[df['cohort_week'] == cohort].sort_values('calendar_week')
            
            if len(cohort_data) == 0:
                continue
            
            x = cohort_data['calendar_week']
            y = cohort_data[metric]
            
            ax.plot(x, y, color=colors[idx], alpha=0.8, linewidth=1.2)
        
        # Add vertical line for forecast split
        if is_forecast and self._last_historical_week is not None:
            ax.axvline(
                x=self._last_historical_week,
                color='#ffffff',
                linestyle='--',
                linewidth=2,
                alpha=0.7,
                label='Forecast Start'
            )
            ax.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='#333333')
        
        # Labels
        metric_labels = {
            'ropc': 'Repeat Orders per Customer (ROPC)',
            'aov': 'Average Order Value (AOV)',
            'total_revenue': 'Total Revenue',
            'acquisitions': 'Acquisitions'
        }
        
        ax.set_xlabel('Calendar Week', fontsize=12, color='#cccccc')
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12, color='#cccccc')
        ax.set_title(
            f'{metric_labels.get(metric, metric)} by Cohort',
            fontsize=14, color='#ffffff', pad=20
        )
        
        # Grid
        ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
        ax.tick_params(colors='#888888')
        
        for spine in ax.spines.values():
            spine.set_color('#333333')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0a0a0a', edgecolor='none', bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        plt.show()
        return fig


# =============================================================================
# Demo
# =============================================================================

if __name__ == '__main__':
    print("CBMT-lifetimes package loaded successfully.")
    print("\nUsage:")
    print("  from CBMT_lifetimes import CohortLifetimes")
    print("  pipeline = CohortLifetimes()")
    print("  pipeline.visualize_cohorts(df, metric='ropc')")
    print("  results = pipeline.train_and_forecast(df, forecast_weeks=13)")
    print("  pipeline.visualize_forecast(results, metric='total_revenue')")

import pandas as pd
from app.core.config import Config

class DatasetLoader:
    """Handle loading and basic validation of raw dataset"""
    
    def __init__(self):
        self.config = Config()
    
    def load_raw_data(self):
        """Load the Natural Questions dataset"""
        print(f"Loading dataset from {self.config.RAW_DATA_PATH}...")
        df = pd.read_csv(self.config.RAW_DATA_PATH)
        print(f"Loaded {len(df)} rows")
        return df
    
    def validate_data(self, df):
        """Basic validation checks"""
        required_cols = ['question', 'long_answers', 'short_answers']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        print("✅ Data validation passed")
        return True
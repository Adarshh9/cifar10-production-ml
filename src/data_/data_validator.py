"""Data validation using Great Expectations."""
import great_expectations as ge
from great_expectations.dataset import PandasDataset
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data quality and schema."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate dataframe schema and quality."""
        try:
            # Convert to Great Expectations dataset
            ge_df = ge.from_pandas(df)
            
            # Define expectations (customize for your data)
            ge_df.expect_table_row_count_to_be_between(min_value=10, max_value=1000000)
            ge_df.expect_column_to_exist('feature')
            ge_df.expect_column_to_exist('label')
            
            # Validate column types
            ge_df.expect_column_values_to_be_of_type('label', 'int64')
            
            # Validate label range (e.g., for classification)
            ge_df.expect_column_values_to_be_between('label', min_value=0, max_value=9)
            
            # Get validation results
            validation_result = ge_df.validate()
            
            if validation_result['success']:
                logger.info("✅ Data validation passed")
                return True
            else:
                logger.error("❌ Data validation failed")
                logger.error(validation_result)
                return False
                
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def validate_all_splits(self) -> bool:
        """Validate all data splits."""
        splits = ['train', 'val', 'test']
        all_valid = True
        
        for split in splits:
            file_path = self.data_dir / f"{split}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                if not self.validate_schema(df):
                    all_valid = False
                    logger.error(f"Validation failed for {split} split")
            else:
                logger.warning(f"{split}.csv not found")
        
        return all_valid

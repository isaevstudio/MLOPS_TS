from pydantic_settings import BaseSettings
from pydantic import Field, validator
from pathlib import Path



class AppConfig(BaseSettings):
    finetuned_model_path: Path = Field(..., description="Path to the fine-tuned FinBERT model")
    df_predict_path: Path = Field(..., description="Path to the csv for predictions")
    future_imitate: Path = Field(..., description="Path for csv that imitate the future predictions")
    saved_model: Path = Field(..., description="Path to the directory containing saved model - TSMixer.pkl")
    prices_by_day: Path = Field(..., description="Path to the csv containing prices by day for prediction")
    news_by_day: Path = Field(..., description="Path to the csv containing news by day for prediction")
    model_file: Path = Field(..., description="Path to the specific model file")
    integrated_data_path: Path = Field(..., description="Path to integrated data")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("finetuned_model_path", 
               "df_predict_path", 
               "future_imitate", 
               "saved_model",
               "prices_by_day",
               "news_by_day",
                "model_file",
                "integrated_data_path")
    
    def validate_paths(cls, path):
        if not path.exists():
            raise ValueError(f"The path '{path}' does not exist.")
        return path

    @property
    def model_file(self) -> Path:
        """Combine the saved_model path with the the model file name."""
        return self.saved_model / "m_TSMixer.pkl"


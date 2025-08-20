import os
import joblib
from dotenv import load_dotenv


# Load .env file
load_dotenv(override=True)

# Variables
APP_NAME = os.getenv('APP_NAME')
VERSION = os.getenv('VERSION')
API_SECRET_KEY = os.getenv('API_SECRET_KEY')


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_FOLDER_PATH = os.path.join(BASE_DIR, "assets")

# Load models
preprocessor = joblib.load(os.path.join(ASSETS_FOLDER_PATH, "preprocessor.pkl"))
forest_model = joblib.load(os.path.join(ASSETS_FOLDER_PATH, "forest_tuned.pkl"))
xgboost_model = joblib.load(os.path.join(ASSETS_FOLDER_PATH, "xgb_tuned.pkl"))
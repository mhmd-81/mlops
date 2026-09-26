import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

base_dir = Path(__file__).parent.parent
env_dir = base_dir / 'src' / '.env'
load_dotenv(env_dir)
data_path = os.getenv("TEST_DATA_PATH")

data = pd.read_csv(data_path)
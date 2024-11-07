# main.py
from src.preprocess import load_data, clean_data, resample_data
from src.analysis import plot_hourly_trends
from src.model import prepare_features, train_model
from src.utils import save_model

# Load and preprocess the data
data_path = 'data/METR-LA.csv'
df = load_data(data_path)
df = clean_data(df)
hourly_data = resample_data(df)

# Analyze data trends
plot_hourly_trends(hourly_data)

# Prepare data for modeling and train a model
X, y = prepare_features(hourly_data)
model = train_model(X, y)

# Save the model
save_model(model)

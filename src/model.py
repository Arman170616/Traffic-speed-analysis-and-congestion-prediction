# src/model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def prepare_features(df):
    """Prepare features and labels for model training."""
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    X = df[['Hour', 'DayOfWeek']].join(df.mean(axis=1).rename('AverageSpeed'))
    y = (X['AverageSpeed'] < X['AverageSpeed'].median()).astype(int)  # 1 for congestion
    return X, y

def train_model(X, y):
    """Train a Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model

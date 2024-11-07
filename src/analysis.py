# src/analysis.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hourly_trends(hourly_data):
    """Plot hourly trends in traffic speed."""
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=hourly_data.mean(axis=1))
    plt.title("Average Hourly Traffic Speed")
    plt.xlabel("Timestamp")
    plt.ylabel("Average Speed")
    plt.show()

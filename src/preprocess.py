import pandas as pd

def load_data():
    df = pd.read_csv("data/vehicle_sensor_data.csv")
    print("Dataset Loaded Successfully.")
    print(df.head())
    return df


def clean_data(df):
    df = df.dropna()   # Remove rows with missing values
    return df

#We will remove values that are unrealistically far from normal.
def remove_outliers(df):
    for col in ["rpm", "speed", "coolant_temp", "engine_load", "vibration", "battery_voltage"]:
        df = df[(df[col] >= df[col].quantile(0.01)) & (df[col] <= df[col].quantile(0.99))]
    return df  


# Scaling
from sklearn.preprocessing import StandardScaler

def scale_features(df):
    scaler = StandardScaler()
    feature_cols = ["rpm", "speed", "coolant_temp", "engine_load", "vibration", "battery_voltage"]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

# Train / Validation / Test
from sklearn.model_selection import train_test_split

def split_data(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# main block
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = remove_outliers(df)
    df, scaler = scale_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print("Preprocessing Completed.")
    print("Train Size:", len(X_train))
    print("Validation Size:", len(X_val))
    print("Test Size:", len(X_test))



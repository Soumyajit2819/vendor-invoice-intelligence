import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split


# 1️⃣ Load Data
def load_vendor_invoice_data(db_path):
    """
    Load vendor_invoice data from SQLite database
    """
    # create connection
    conn = sqlite3.connect(db_path)
    
    query = "SELECT * FROM vendor_invoice"
    df = pd.read_sql_query(query, conn)
    
    conn.close()  # close connection
    
    # basic cleaning
    
    return df


# 2️⃣ Prepare Features
def prepare_features(df):
    """
    Select features and target variable
    """
    X = df[['Dollars']]   # only Dollars
    y = df['Freight']
    
    return X, y


# 3️⃣ Train-Test Split
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
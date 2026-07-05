import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_path):
    # Memuat dataset
    df = pd.read_csv(input_path)
    
    # Definisikan kolom kategorikal dan numerik
    categorical_col = ['Stage_fear', 'Drained_after_socializing']
    numerical_col = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    # Menangani missing values
    numeric_imputer = SimpleImputer(strategy='median')
    df[numerical_col] = numeric_imputer.fit_transform(df[numerical_col])
    
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_col] = categorical_imputer.fit_transform(df[categorical_col])
    
    # Encoding data
    df['Personality'] = (df['Personality'] == 'Introvert').astype(float)
    df[categorical_col] = (df[categorical_col] == 'Yes').astype(float)
    
    # Feature engineering
    df['social_alone_ratio'] = (df['Social_event_attendance'] + df['Going_outside']) / (df['Time_spent_Alone'] + 1)
    df['friends_and_posts'] = df['Friends_circle_size'] * df['Post_frequency']
    df['drained_by_social'] = df['Drained_after_socializing'] * df['Social_event_attendance']
    
    # Definisikan semua kolom numerik, termasuk yang baru
    all_numerical_col = numerical_col + ['social_alone_ratio', 'friends_and_posts', 'drained_by_social']
    
    # Normalisasi fitur numerik
    scaler = StandardScaler()
    df[all_numerical_col] = scaler.fit_transform(df[all_numerical_col])
    
    # Menyimpan dataset
    df.to_csv(output_path, index=False)
    print(f"Dataset yang sudah diproses disimpan sebagai {output_path}")
    
    return df

if __name__ == "__main__":
    input_path = 'personality_dataset_raw.csv'
    output_path = 'preprocessing/personality_dataset_preprocessing.csv'
    preprocess_data(input_path, output_path)

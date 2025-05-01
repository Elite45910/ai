import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
 
# โหลด dataset
def load_data():
    # ใช้ Spotify Dataset จาก Kaggle
    # ลิงก์: https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset
    df = pd.read_csv('data/data.csv')  # แก้ไข path ตามที่เก็บไฟล์
    return df
 
# เตรียมข้อมูล
def preprocess_data(df):
    # เลือก features ที่ใช้ในการแนะนำ
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
   
    # จัดการ missing values
    df = df.dropna(subset=features)
   
    # Normalize features
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    return df, X_scaled, scaler, features
 
# สร้างโมเดล K-Means
def train_kmeans(X_scaled, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    return kmeans
 
# แนะนำเพลง
def recommend_songs(song_name, df, X_scaled, kmeans, scaler, features, top_n=5):
    try:
        # ค้นหาเพลง
        song_idx = df[df['name'].str.lower() == song_name.lower()].index[0]
       
        # ได้รับ cluster ของเพลง
        song_cluster = kmeans.predict(X_scaled[song_idx].reshape(1, -1))[0]
       
        # หาเพลงใน cluster เดียวกัน
        cluster_indices = df[kmeans.labels_ == song_cluster].index
       
        # คำนวณความคล้ายคลึงด้วย cosine similarity
        song_features = X_scaled[song_idx].reshape(1, -1)
        similarities = cosine_similarity(song_features, X_scaled[cluster_indices])[0]
       
        # จัดเรียงตามความคล้ายคลึง
        similar_indices = cluster_indices[np.argsort(similarities)[::-1]][:top_n+1]
       
        # แสดงผลแนะนำ
        print(f"\nเพลงที่แนะนำสำหรับ '{song_name}':")
        recommendations = []
        for idx in similar_indices:
            if idx != song_idx:  # ข้ามเพลงตัวเอง
                song = {
                    'name': df.iloc[idx]['name'],
                    'artists': df.iloc[idx]['artists'],
                    'year': df.iloc[idx]['year']
                }
                recommendations.append(song)
                print(f"- {song['name']} โดย {song['artists']} ({song['year']})")
       
        return recommendations
   
    except IndexError:
        print(f"ไม่พบเพลง '{song_name}' ใน dataset")
        return []
 
def main():
    # โหลดและเตรียมข้อมูล
    print("กำลังโหลดข้อมูล...")
    df = load_data()
    df, X_scaled, scaler, features = preprocess_data(df)
   
    # ฝึกโมเดล
    print("กำลังฝึกโมเดล K-Means...")
    kmeans = train_kmeans(X_scaled)
   
    # ทดสอบแนะนำเพลง
    test_song = input("กรุณาใส่ชื่อเพลงที่ต้องการแนะนำ: ")
    recommend_songs(test_song, df, X_scaled, kmeans, scaler, features)
 
if __name__ == "__main__":
    main()
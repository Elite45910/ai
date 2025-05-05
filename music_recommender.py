import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox
import warnings
warnings.filterwarnings('ignore')

# -------------------- โหลดและเตรียมข้อมูล --------------------

def load_data():
    df = pd.read_csv('data/data.csv')  # เปลี่ยน path ตามที่อยู่จริง
    return df

def preprocess_data(df):
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    df = df.dropna(subset=features)
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled, scaler, features

def train_kmeans(X_scaled, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    return kmeans

# -------------------- แนะนำเพลงคล้ายกัน (จากกลุ่ม K-Means) --------------------

def recommend_similar_songs(song_name, df, X_scaled, kmeans, top_n=5):
    try:
        song_idx = df[df['name'].str.lower() == song_name.lower()].index[0]
        song_cluster = kmeans.predict(X_scaled[song_idx].reshape(1, -1))[0]
        cluster_indices = df[kmeans.labels_ == song_cluster].index

        song_features = X_scaled[song_idx].reshape(1, -1)
        similarities = cosine_similarity(song_features, X_scaled[cluster_indices])[0]
        similar_indices = cluster_indices[np.argsort(similarities)[::-1]][:top_n+1]

        recommendations = []
        for idx in similar_indices:
            if idx != song_idx:
                recommendations.append({
                    'name': df.iloc[idx]['name'],
                    'artists': df.iloc[idx]['artists'],
                    'year': df.iloc[idx]['year']
                })
        return recommendations
    except:
        return []

# -------------------- แนะนำเพลงที่ Mood คล้ายกัน (Cosine Similarity) --------------------

def recommend_mood_songs(song_name, df, scaler, features, top_n=5):
    try:
        song_row = df[df['name'].str.lower() == song_name.lower()]
        if song_row.empty:
            return []

        song_features = song_row[features]
        song_scaled = scaler.transform(song_features)
        all_scaled = scaler.transform(df[features])

        similarities = cosine_similarity(song_scaled, all_scaled)[0]
        similar_indices = np.argsort(similarities)[::-1][:top_n+1]

        recommendations = []
        for idx in similar_indices:
            if df.iloc[idx]['name'].lower() != song_name.lower():
                recommendations.append({
                    'name': df.iloc[idx]['name'],
                    'artists': df.iloc[idx]['artists'],
                    'year': df.iloc[idx]['year']
                })
        return recommendations
    except:
        return []

# -------------------- GUI --------------------

class MusicRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎶 ระบบแนะนำเพลง AI")
        self.root.geometry("650x500")
        self.root.configure(bg="#1e1e2e")

        tk.Label(root, text="🎧 Music Recommender", font=("Arial", 20, "bold"), bg="#1e1e2e", fg="white").pack(pady=20)

        frame = tk.Frame(root, bg="#1e1e2e")
        frame.pack()

        tk.Label(frame, text="ชื่อเพลง:", font=("Arial", 14), bg="#1e1e2e", fg="white").grid(row=0, column=0, padx=10)
        self.song_entry = tk.Entry(frame, font=("Arial", 14), width=30)
        self.song_entry.grid(row=0, column=1)

        self.mode = tk.StringVar(value="similar")
        tk.Radiobutton(root, text="🎧 เพลงที่คล้ายกัน (จากกลุ่ม)", variable=self.mode, value="similar",
                       font=("Arial", 12), bg="#1e1e2e", fg="white", selectcolor="#444").pack()
        tk.Radiobutton(root, text="🎵 เพลงที่ Mood คล้ายกัน", variable=self.mode, value="mood",
                       font=("Arial", 12), bg="#1e1e2e", fg="white", selectcolor="#444").pack()

        tk.Button(root, text="แนะนำเพลง", font=("Arial", 14), bg="#5e60ce", fg="white",
                  command=self.recommend).pack(pady=15)

        self.result_box = tk.Text(root, height=12, width=75, font=("Arial", 12), bg="#2a2a3b", fg="white")
        self.result_box.pack(pady=10)

        # เตรียมข้อมูล
        self.df = load_data()
        self.df, self.X_scaled, self.scaler, self.features = preprocess_data(self.df)
        self.kmeans = train_kmeans(self.X_scaled)

    def recommend(self):
        self.result_box.delete("1.0", tk.END)
        song_name = self.song_entry.get().strip()

        if not song_name:
            messagebox.showerror("❌ ข้อผิดพลาด", "กรุณากรอกชื่อเพลง")
            return

        if self.mode.get() == "similar":
            results = recommend_similar_songs(song_name, self.df, self.X_scaled, self.kmeans)
            title = f"\n🎧 เพลงที่คล้ายกับ '{song_name}':\n"
        else:
            results = recommend_mood_songs(song_name, self.df, self.scaler, self.features)
            title = f"\n🎵 เพลงที่ Mood คล้ายกับ '{song_name}':\n"

        if not results:
            self.result_box.insert(tk.END, f"ไม่พบเพลง '{song_name}' หรือไม่มีคำแนะนำ")
        else:
            self.result_box.insert(tk.END, title)
            for song in results:
                self.result_box.insert(tk.END, f"- {song['name']} โดย {song['artists']} ({song['year']})\n")

# -------------------- รันโปรแกรม --------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = MusicRecommenderApp(root)
    root.mainloop()

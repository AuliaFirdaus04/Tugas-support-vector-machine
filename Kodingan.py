import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Menampilkan seluruh baris tabel support vector di terminal tanpa terpotong
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# 1. Membaca dataset dan menyiapkan fitur serta target biner
df = pd.read_csv('Mall_Customers.csv')
df['Target'] = (df['Spending Score (1-100)'] > 50).astype(int)
df['Gender_Enc'] = df['Gender'].map({'Male': 0, 'Female': 1})

X = df[['Age', 'Annual Income (k$)', 'Gender_Enc']]
y = df['Target']

# 2. Split 80% data latih dan 20% data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Pengujian SVM untuk setiap nilai C
nilai_C = [0.01, 0.1, 1, 10]

for c in nilai_C:
    print("=" * 65)
    print(f"PARAMETER C = {c}")
    print("=" * 65)
    
    model_svm = SVC(C=c, random_state=42)
    model_svm.fit(X_train, y_train)
    
    y_pred = model_svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi Data Uji : {acc * 100:.2f}%\n")
    
    # Hasil klasifikasi 5 teratas dan 5 terbawah
    hasil_df = X_test.copy()
    hasil_df['Target_Asli'] = y_test.values
    hasil_df['Hasil_Prediksi'] = y_pred
    
    print("--- 5 Hasil Klasifikasi Teratas ---")
    print(hasil_df.head(5))
    
    print("\n--- 5 Hasil Klasifikasi Terbawah ---")
    print(hasil_df.tail(5))
    
    # Jumlah support vector setiap C
    print(f"\n--- Jumlah Support Vector (C = {c}) ---")
    print(f"Total Support Vector : {len(model_svm.support_)} data")
    print(f"Per Kelas            : Kelas 0 = {model_svm.n_support_[0]}, Kelas 1 = {model_svm.n_support_[1]}")
    
    # Data yang dianggap support vector setiap C
    sv_df = pd.DataFrame(model_svm.support_vectors_, columns=['Age', 'Annual Income (k$)', 'Gender_Enc'])
    sv_df['Indeks_Data_Latih'] = model_svm.support_
    print(f"\n--- Data yang Dianggap Support Vector (C = {c}) ---")
    print(sv_df)
    print("\n")
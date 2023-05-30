import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Fungsi untuk ekstraksi fitur dari gambar
def ekstraksi_fitur(image):
    # Ubah gambar ke citra grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Deteksi tepi menggunakan operator Canny
    edges = cv2.Canny(gray, 50, 150)
    
    # Tampilkan citra tepi
    cv2.imshow('Citra Tepi', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return edges

# Fungsi untuk mendeteksi watermark dan menentukan apakah uang asli atau palsu
def deteksi_watermark(image_path):
    # Baca gambar
    image = cv2.imread(image_path)
    
    # Deteksi tepi menggunakan operator Canny
    edges = ekstraksi_fitur(image)
    
    # Dilasi untuk mempertajam tepi
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Temukan kontur pada gambar
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Periksa setiap kontur
    for contour in contours:
        # Hitung luas kontur
        area = cv2.contourArea(contour)
        
        # Jika luas kontur melebihi ambang tertentu, anggap sebagai watermark
        if area > 500:
            # Gambar persegi di sekitar kontur
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, 'Watermark', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Tampilkan gambar hasil deteksi
    cv2.imshow('Deteksi Watermark', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Contoh penggunaan fungsi
image_paths = ('gambar1.jpg', 'gambar2.jpg', 'gambar3.jpg', 'gambar4.jpg')
X = []
y = []

for path in image_paths:
    #deteksi_watermark(path)
    X.append(ekstraksi_fitur(cv2.imread(path)))
    # Ganti dengan label yang sesuai
    # Jika gambar tersebut adalah uang asli, beri label 1, jika palsu, beri label 0
    # Contoh: uang asli, uang palsu, uang palsu
    if 'gambar1.jpg' in path:
        y.append(1)
    elif 'gambar2.jpg' in path:
        y.append(1)
    elif 'gambar3.jpg' in path:
        y.append(1)
    elif 'gambar4.jpg' in path:
        y.append(1)
    else:
        y.append(0)
        
# Mengubah X dan y menjadi array numpy
X = np.array(X, dtype=object)
y = np.array(y)

# Memisahkan menjadi subset pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat objek SVC dan melatih model
model = SVC()

# Normalisasi fitur pada subset pelatihan
X_train_normalized = []
for features in X_train:
    normalized_features = (features - np.min(features)) / (np.max(features) - np.min(features))
    X_train_normalized.append(normalized_features)

# Ubah menjadi array numpy
X_train_normalized = np.array(X_train_normalized)

# Melatih model
model.fit(X_train_normalized.reshape(X_train_normalized.shape[0], -1), y_train)

# Memprediksi label pada subset pengujian
X_test_normalized = []
for features in X_test:
    normalized_features = (features - np.min(features)) / (np.max(features) - np.min(features))
    X_test_normalized.append(normalized_features)

# Ubah menjadi array numpy
X_test_normalized = np.array(X_test_normalized)

y_pred = model.predict(X_test_normalized.reshape(X_test_normalized.shape[0], -1))

# Ekstraksi fitur pada gambar baru
new_image_path = 'new_image.jpg'
new_image = cv2.imread(new_image_path)
new_image_features = ekstraksi_fitur(new_image)

# Normalisasi fitur pada gambar baru
new_image_normalized = (new_image_features - np.min(new_image_features)) / (np.max(new_image_features) - np.min(new_image_features))

# Prediksi label
prediction = model.predict([new_image_normalized.flatten()])

if prediction[0] == 1:
    print("Uang tersebut adalah uang asli.")
else:
    print("Uang tersebut adalah uang palsu.")

# Menghitung akurasi prediksi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)

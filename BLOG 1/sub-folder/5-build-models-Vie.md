### 5. Xây dựng một Mô hình Học Máy Cơ bản

Chúng ta sẽ dùng Python và scikit-learn, thư viện này rất thân thiện với người mới và xử lý hầu hết phần phức tạp mà không cần đào sâu vào toán học ngay từ đầu. Mục tiêu là làm cho phần này thực tế để bạn có thể tự thử và thấy cách các khái niệm từ các phần trước kết hợp trong code.

#### 5.1 Công cụ Bạn Cần

Trước khi bắt tay vào, hãy chuẩn bị nhé. Đây là những thứ cơ bản thôi – không có gì cầu kỳ, và tất cả đều miễn phí hoặc dễ cài.

- **Python**: Phiên bản 3.8 trở lên. Nếu chưa có, tải từ python.org. Đây là ngôn ngữ phổ biến nhất cho học máy nhờ sự đơn giản và cộng đồng hỗ trợ lớn.
- **Môi trường**: Mình recommend Jupyter Notebook (cài qua Anaconda cho tiện) vì nó cho phép chạy code theo ô, xem kết quả ngay, và xen lẫn giải thích. Hoặc dùng Google Colab cũng hay – online, miễn phí, không cần cài đặt; chỉ cần đăng nhập Google là code được. Còn tiện chia sẻ notebook nữa.
- **Thư viện**: Đây là các package Python giúp tăng sức mạnh cho học máy. Cài một lần qua terminal hoặc Colab bằng lệnh:
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn
  ```
  - **numpy**: Để xử lý mảng số và tính toán hiệu quả (như Excel phiên bản siêu cấp cho dữ liệu số).
  - **pandas**: Giúp làm việc với dữ liệu dễ dàng, từ tải, làm sạch đến khám phá bảng biểu.
  - **matplotlib và seaborn**: Để vẽ biểu đồ và hình ảnh – seaborn là phiên bản đẹp đẽ hơn của matplotlib.
  - **scikit-learn**: Nhân vật chính. Nó có sẵn dữ liệu, thuật toán và công cụ để huấn luyện/đánh giá mô hình. Chưa cần đến neural network; đây là học máy cổ điển.

Nếu mới cài đặt, search "cài Jupyter Notebook" hoặc dùng Colab để bỏ qua bước đó.

#### 5.2 Ví dụ Vấn đề – Phân loại Hoa Iris

Với ví dụ này, chúng ta sẽ dùng bộ dữ liệu Iris, giống như "Hello World" trong học máy. Nó được thu thập bởi nhà sinh học Ronald Fisher năm 1936 và xuất hiện trong vô số bài hướng dẫn.

- **Tại sao chọn bộ dữ liệu này?** Nó nhỏ (chỉ 150 mẫu), sạch sẽ (không giá trị thiếu hay lộn xộn), và cân bằng (50 mẫu mỗi loài). Nhờ vậy, ta tập trung vào quy trình mà không bị mắc kẹt ở khâu làm sạch dữ liệu. Thực tế thì dữ liệu thường rối hơn, nhưng bắt đầu đơn giản giúp xây dựng tự tin.
- **Đặc trưng (đầu vào)**: Bốn thông số đo từ hoa:
  - Chiều dài đài hoa (lá ngoài dài, đơn vị cm)
  - Chiều rộng đài hoa
  - Chiều dài cánh hoa (phần màu sắc bên trong)
  - Chiều rộng cánh hoa
  Những cái này đều là số, rất phù hợp cho thuật toán học máy.
- **Mục tiêu (đầu ra)**: Loài hoa – một trong ba loại:
  - Setosa (dễ phân biệt, thường tách biệt trên biểu đồ)
  - Versicolor
  - Virginica (hai loại này chồng chéo hơn, tạo chút thử thách)
- **Loại nhiệm vụ**: Đây là phân loại có giám sát – ta có dữ liệu đã gắn nhãn (biết loài của từng mẫu), và mô hình học cách mapping đặc trưng sang nhãn.
- **Ứng dụng thực tế**: Tưởng tượng như nhận diện loại cây từ ảnh hoặc đo lường trong nông nghiệp, hoặc tương tự trong y tế (như phân loại khối u từ ảnh chụp).

Bộ dữ liệu được tích hợp sẵn trong scikit-learn, nên chỉ cần một dòng lệnh để tải. Nếu muốn xem ngoài code, tìm trên Kaggle hoặc UCI ML Repository.

#### 5.3 Mã Từng Bước

Bây giờ đến phần thú vị: code. Mình sẽ đưa toàn bộ script kèm chú thích giải thích. Nó theo đúng quy trình 5 bước. Copy vào Jupyter/Colab và chạy từng ô – bạn sẽ thấy bảng dữ liệu, biểu đồ, kết quả hiện ra.

Mình chọn K-Nearest Neighbors (KNN) vì nó dễ hiểu: phân loại điểm mới dựa trên "k" láng giềng gần nhất trong dữ liệu và lấy phiếu đa số. Không hộp đen bí ẩn; có thể hình dung cách nó hoạt động.

```python
# 1. Import các thứ cần thiết – như hộp dụng cụ cho dữ liệu và học máy
import numpy as np  # Xử lý mảng số và toán học
import pandas as pd  # Dataframe (giống bảng tính)
import matplotlib.pyplot as plt  # Vẽ biểu đồ cơ bản
import seaborn as sns  # Biểu đồ đẹp hơn

from sklearn.datasets import load_iris  # Bộ dữ liệu sẵn
from sklearn.model_selection import train_test_split  # Chia dữ liệu
from sklearn.preprocessing import StandardScaler  # Chuẩn hóa đặc trưng
from sklearn.neighbors import KNeighborsClassifier  # Thuật toán của ta
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Đánh giá

# 2. Tải và khám phá dữ liệu (Bước 2: Thu thập & Tiền xử lý Dữ liệu)
iris = load_iris()  # Tải dưới dạng bunch object
X = iris.data       # Đặc trưng: 150 hàng x 4 cột (đo lường)
y = iris.target     # Nhãn: 150 giá trị (0=Setosa, 1=Versicolor, 2=Virginica)

# Chuyển sang DataFrame pandas để dễ xem và phân tích
df = pd.DataFrame(X, columns=iris.feature_names)  # Thêm tên cột
df['species'] = pd.Categorical.from_codes(y, iris.target_names)  # Thêm tên loài

# Kiểm tra nhanh: 5 hàng đầu và tóm tắt
print("5 hàng dữ liệu đầu tiên:")
print(df.head())
print("\nTóm tắt dữ liệu:")
print(df.describe())  # Thống kê như mean, min, max
print("\nSố lượng mỗi loài:")
print(df['species'].value_counts())  # Nên 50 mỗi loài – cân bằng!

# Vẽ để xem mối quan hệ (ví dụ, cánh hoa phân biệt loài tốt)
sns.pairplot(df, hue='species', palette='husl')  # Biểu đồ phân tán cho các cặp đặc trưng, màu theo loài
plt.suptitle("Pairplot Dữ liệu Iris")
plt.show()

# 3. Chuẩn bị dữ liệu: chia và chuẩn hóa (vẫn Bước 2)
# Chia: 80% huấn luyện, 20% kiểm tra – random_state để lặp lại kết quả
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Stratify giữ cân bằng loài
)

# Chuẩn hóa: KNN dùng khoảng cách, nên scale về mean=0, std=1 tránh đặc trưng lớn lấn át
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit và transform trên train
X_test = scaler.transform(X_test)        # Chỉ transform trên test (tránh leak dữ liệu)

# 4. Huấn luyện mô hình (Bước 3: Chọn & Huấn luyện Mô hình)
# KNN: phân loại dựa trên phiếu của k láng giềng gần nhất
model = KNeighborsClassifier(n_neighbors=5)  # Bắt đầu với k=5 – mặc định hay
model.fit(X_train, y_train)  # Đây là lúc mô hình "học" – lưu dữ liệu huấn luyện

# 5. Đánh giá (Bước 4: Đánh giá Mô hình)
y_pred = model.predict(X_test)  # Dự đoán trên test

# Chỉ số cơ bản: độ chính xác (tỷ lệ đúng)
acc = accuracy_score(y_test, y_pred)
print(f"Độ chính xác trên tập test: {acc:.3f} (tức {acc*100:.1f}%)")

# Báo cáo chi tiết: precision (tránh false positive), recall (tránh false negative), f1 (cân bằng)
print("\nBáo cáo Phân loại:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Ma trận nhầm lẫn: xem lỗi ở đâu (ví dụ, Versicolor nhầm thành Virginica)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Nhãn Dự đoán')
plt.ylabel('Nhãn Thực')
plt.title('Ma trận Nhầm lẫn – Mô hình Làm Tốt đến Đâu?')
plt.show()

# 6. Cải thiện (Bước 5: Cải thiện & Triển khai)
# Tinh chỉnh k: thử từ 1-14, xem k nào tốt nhất
accuracies = []
for k in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)  # Score cho độ chính xác
    accuracies.append(acc)

# Vẽ để xem – đỉnh là k tối ưu
plt.plot(range(1,15), accuracies, marker='o', linestyle='--')
plt.xlabel('Số Láng giềng (k)')
plt.ylabel('Độ chính xác Test')
plt.title('Tinh chỉnh k để Tăng Hiệu suất')
plt.grid(True)
plt.show()

best_k = np.argmax(accuracies) + 1  # +1 vì range từ 1
print(f"k Tốt nhất = {best_k} với độ chính xác = {max(accuracies):.3f}")

# Để triển khai: lưu mô hình vào file (bỏ comment để dùng)
# import joblib
# joblib.dump(model, 'iris_model.pkl')  # Load sau bằng joblib.load()
```

#### Giải thích Chi tiết các Bước

Mình phân tích kỹ hơn để bạn nắm rõ lý do làm từng bước.

- **Tải và Khám phá Dữ liệu**: Bắt đầu bằng tải – không cần crawl web ở đây. In head/describe giúp phát hiện vấn đề (như outlier). Pairplot quan trọng: cho thấy Setosa dễ tách, nhưng Versicolor/Virginica chồng chéo ở vài đặc trưng, nên mô hình phải học sự khác biệt tinh tế.
  
- **Chia và Chuẩn hóa**: Chia train/test tránh overfitting (mô hình nhớ thuộc lòng thay vì tổng quát). Stratify giữ tỷ lệ loài cân bằng. Chuẩn hóa cần thiết cho KNN vì dựa trên khoảng cách – nếu không, đặc trưng lớn (cm vs mm) sẽ lấn át.

- **Huấn luyện**: .fit() là bước chính – với KNN, nó chỉ lưu dữ liệu. Dự đoán nhanh, xảy ra lúc query.

- **Đánh giá**: Độ chính xác đơn giản nhưng không phải lúc nào cũng hay (nếu dữ liệu lệch). Báo cáo phân loại thêm chi tiết: precision (dự đoán Setosa đúng bao nhiêu?), recall (bắt hết Setosa chưa?). Ma trận nhầm lẫn hiển thị lỗi trực quan – lý tưởng đường chéo cao, ngoài thấp.

- **Cải thiện**: Đây là tune hyperparameter. k nhỏ quá thì overfit nhiễu, lớn quá thì underfit pattern. Ta loop và plot để tìm điểm ngọt (thường 3-7 ở đây). Trong dự án thực, dùng GridSearchCV tự động.

**Kết quả Thường thấy**: Với chia này, độ chính xác hay đạt 1.000 (100%), vì Iris dễ. Với dữ liệu khó hơn, 80-90% là tốt. Chạy lại với random_state khác có thể giảm nhẹ – đó là biến động.

**Ý tưởng Mở rộng**:
- Thay KNN bằng LogisticRegression hoặc DecisionTreeClassifier (import từ sklearn).
- Thêm cross-validation: Dùng cross_val_score để đánh giá chắc chắn hơn.
- Dữ liệu thực: Lấy từ Kaggle, như dự đoán sống sót trên Titanic.
- Triển khai: Bọc vào app Flask làm web interface (nhập đo lường, nhận loài).
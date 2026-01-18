# Học machine learning cơ bản
![Machinea_Learning_broad](https://mindlabinc.ca/wp-content/uploads/2024/05/Machine-Learning.webp)
Trước khi Machine Learning được phổ biến rộng rãi, có phải bạn từng nghĩ những người máy có nhận thức chỉ xuất hiện trên màn ảnh Hollywood hay hoạt hình?

Giờ đây cú "*twist*" của Machine Learning đã hiện hữu trong đời sống của chúng ta. Đây là thời đại của Machine Learning, khi chỉ cần lướt điện thoại, nhập tìm kiếm, lưu, thả tim, thích,... đều được máy ghi lại dữ liệu.

> "Machine learning is not magic, and it’s not menace. It’s mathematics with purpose"
> -- ScienceNewsToday

Tạm dịch: "Học máy không phải phép màu, cũng không phải mối đe dọa của con người. Đây là toán học có mục đích"

## 1.Giới thiệu tổng quan về Machine Learning
Giai đoạn khởi nguồn của Machine Learning bắt đầu từ những năm 1940-1950, Alan Turing đã đặt ra câu hỏi: "Liệu máy tính có thể suy nghĩ như con người không?" đã len lỏi trong giới khoa học. \
Năm 1943, McCulloch và Pitts đã đề xuất mô hình neuron nhân tạo đầu tiên. Đây chính là bước ngoặt quan trọng nhất của máy tính và đặt nền móng cho Machine Learning sau này.

### 1.1. Định nghĩa cơ bản (Machine Learning là gì, v..)
**Machine Learning (Học Máy) là hệ thống máy tính học hỏi từ những điểm dữ liệu thông qua quá trình huấn luyện**. Khi máy đã học và cải thiện từ mẫu, máy tạo ra các mô hình thuật toán có thể suy luận những dữ liệu mới, từ đó đề xuất dự đoán với độ chính xác cao. 

Ví dụ: Machine Learning phân tích thói quen lướt mạng thông qua nội dung thường xem. Sau đó đề xuất video, sản phẩm mới có nội dung tương tự.

### 1.2. Lập trình truyền thống vs Machine Learning
Hệ thống trong lập trình truyền thống hoạt động theo câu lệnh điều kiện đơn thuần được viết sẵn như if-else, case A-B,... Vậy bản chất của lập trình truyền thống là máy chỉ làm đúng những gì được viết, có quy tắc ổn định trong quản lý và tính toán. Muốn thay đổi lập trình theo lối cũ thì phải "*sửa*" code.

Machine Learning so với lập trình truyền thống thì không cần mô tả bằng thuật toán rõ ràng. **Cốt lõi xây dựng mô hình Machine Learning nằm ở dữ liệu**. Machine Learning cho phép hệ thống vận dụng *INPUT* và dự đoán *OUTPUT*, thích nghi và cải thiện theo thời gian khi có thêm dữ liệu mới với độ chính xác dựa trên xác suất.

### 1.3. Phân biệt AI/DL/Machine Learning
|   | Artificial Intelligence | Deep Learning | Machine Learning |
|---|-----|-----|-----|
| Phạm vi | Lĩnh vực rộng nhất  | Một phần của AI | Một nhánh chuyên sâu của Machine Learning |
| Mục tiêu  | Mọi hệ thống có thể thực hiện nhiệm vụ cần trí tuệ của con người | Học từ dữ liệu và cải thiện theo thời gian | Nhận diện mẫu phức tạp |
| Dữ liệu |   | Xử lý dữ liệu có cấu trúc | Xử lý dữ liệu không có cấu trúc phức tạp hơn như ảnh/video (mạng neurons nhiều tầng) |
### 1.4. Vì sao Machine Learning lại quan trọng

Các nhà khoa học bắt đầu tạo ra các chương trình để máy tính phân tích lượng lớn dữ liệu, thích nghi với các sai số do môi trường biến đổi liên tục và giải quyết những bài toán phức tạp vượt quá khả năng của con người. \
Các công nghệ lớn hiện nay đều áp dụng Machine Learning như mô hình dự đoán khí hậu, chẩn đoán y khoa, giao thông tự vận hành theo thời gian thực,... 

#### Tổng quan vai trò của Machine Learning:
**1. Đưa ra quyết định**

    - Phân loại dữ liệu phi tuyến tính, mối quan hệ phức tạp.
    - Nhận dạng các mẫu dữ liệu
Yếu tố cho khả năng dự đoán trên nhiều lĩnh vực khác nhau.
**2. Dự đoán xu hướng và kết quả** 
    - Cá nhân hóa và tùy chỉnh qua trải nghiểm, điều chỉnh dịch vụ, đề xuất
    - Hệ thống đề xuất, gợi ý nội dung tương tự 
**3. Ứng dụng rộng rãi**
    - Giáo dục: Phát hiện dấu hiệu điểm bất thường
    - Ngân hàng: phát hiện giao dịch bất thường
    - Ứng dụng: Spotify gợi ý bài hát dựa trên thói quen nghe nhạc
    - Kinh doanh: Dự đoán nhu cầu mua hàng hóa, tối ưu hóa quy trình sản xuất
    - Y tế: trợ lý thuốc cá nhân, phân tích ảnh chụp X-quang, CT
## 2. Các loại thuật toán máy học
![Các loại thuật toán máy học](/static/uploads/20260118_204736_c834b889.png)

Các loại thuật toán máy học hiện nay rất đa dạng để phục vụ nhiều bài toán cụ thể. Do đó trong phần này sẽ giải thích ngắn gọn về hai nhóm chính là học giám sát học không giám sát và các thuật toán phổ biến là học bán giám sát và học tăng cường.

### 2.1. Học có giám sát

Học có giám sát là việc lấy các dữ liệu được dán nhãn để huấn luyện mô hình sao cho mô hình này có khả năng dự đoán được mối quan hệ giữa đầu vào và đầu ra.

![Ví dụ về học có giám sát](/static/uploads/20260118_204909_2b8164f8.jpg)

Ví dụ trên có các hình ảnh chia làm hai loại: 
- Một hình được gán nhãn **táo**
- Một hình được gán nhãn **dâu tây**
Mô hình sau khi được huấn luyện sẽ dự đoán được các hình ảnh mới (không được đưa vào huấn luyện) là **táo** hoặc **dâu tây**. 

### 2.2. Học không giám sát

Học không giám sát có dữ liệu huấn luyện bao gồm dữ liệu đầu vào và không có đầu ra tương ứng (dữ liệu không được gán nhãn). 

Thuật toán này trích xuất những thông tin quan trọng giữa các điểm dữ liệu và phân tách ra thành các nhóm. 

![Ví dụ về học không giám sát](/static/uploads/20260118_204931_2fe2b3b0.webp)

Ví dụ trên có các hình ảnh không có nhãn và mô hình dựa vào đó để tìm ra các nhóm có các thông tin liên quan tới nhau:
- Hình quả táo có hình dạng tương xứng
- Hình quả chanh và cam đều có một quả còn nguyên vẹn và một quả cắt nửa.
- Hình quả dưa hấu với một miếng dưa hấu đưa vào nhóm riêng

## 2.3. Học bán giám sát

Thực tế dữ liệu dán nhãn thì khan hiếm và khó thực hiện (chi phí cao và tốn thời gian), và dữ liệu không dán nhãn thì nhiều vô số, là nơi mà học bán giám sát sẽ tỏa sáng.

Học bán giám sát là sự kết hợp giữa học có giám sát và không giám sát vì thuật toán này bao gồm cả dữ liệu có nhãn và không có nhãn. 

Mục đích cuối cùng của mô hình thuật toán này là dự đoán được đầu ra tốt hơn khi mà học có giám sát và học không giám sát chỉ sử dụng 1 loại dữ liệu (có nhãn hoặc không nhãn).

Ví dụ về phân loại tin tức trên các trang báo hoặc các trang mạng xã hội:
- Có 200 bài báo được gắn nhãn chủ đề (thể thao, kinh tế, giải trí, ...)
- Có 100,000 bài báo chưa phân loại
- Học cấu trúc ngôn ngữ và chủ đề từ cả hai nguồn dữ liệu

## 2.4. Học tăng cường

Học tăng cường là loại mô hình thuật toán cho phép các tác nhân (agent) tự động (automation) liên tục đưa ra các quyết định bằng cách giao tiếp với môi trường xung quanh để tự củng cố hành vi. Hay nói cách khác là học qua thử-sai và phản hồi từ môi trường

Ví dụ về xe tự lái:
- Trạng thái: Vị trí xe, tốc độ, làn đường, xe xung quanh
- Hành động: Rẽ trái/phải, tăng/giảm tốc, phanh
- Phần thưởng: +10 di chuyển an toàn, -100 va chạm, -5 chuyển làn không cần thiết
- Xe học cách lái an toàn qua hàng triệu lần mô phỏng
## 3. Machine Learning Algorithms
Mỗi phương thức học giống như một “triết lý giáo dục” khác nhau dành cho máy tính. Nếu coi Machine Learning là một hành trình giải quyết vấn đề, thì các phương thức học chính là “chiến lược”, còn thuật toán chính là “vũ khí” cụ thể. Tùy thuộc vào đặc điểm dữ liệu và mục tiêu dự báo, chúng ta sẽ chọn ra loại vũ khí phù hợp nhất.
### 3.1. Nhóm hồi quy (Regression) – dự báo giá trị số.
Mục tiêu: tìm ra quy luật để dự báo một biến số liên tục.
- Linear Regression (hồi quy tuyến tính): Không đơn thuần là tìm đường thẳng, nó còn giúp chúng ta hiểu mức độ ảnh hưởng của từng biến đầu vào. Ví dụ điển hình của thuật toán này là dự đoán giá nhà, cụ thể, ta có thể biết chính xác diện thích tăng thêm 1m2 thì tổng giá trị căn nhà tăng bao nhiêu triệu đồng.
- Ridge & Lasso Regression: Đây là các phiên bản “nâng cấp” của hồi quy tuyến tính. Chúng them vào các thành phần toán học để ngăn chặn hiện tượng Overfitting (mô hình có kết quả tốt với dữ liệu cũ nhưng dự báo hiệu quả thấp với dữ liệu mới), giuos mô hình bền bỉ hơn.
- Random Forest Regression: Thay vì dùng 1 hàm toán học duy nhất, thuật toán này kết hợp kết quả từ hàng trăm “cây quyết định” khác nhau để đưa ra con số cuối cùng, hiệu quả với dữ liệu có nhiều biến phức tạp.
### 3.2. Nhóm phân loại (Classification) – Định danh đối tượng
Mục tiêu: Xác định dữ liệu thuộc về nhóm nào trong các nhóm đã cho trước.
- Logistic Regression: Trái với tên gọi, đây là “ông vua” của phân loại nhị phân (0 và 1). Thuật toán này rất phổ biến trong y khoa để dự đoán bệnh nhân có mắc bệnh hay không dựa trên các chỉ số xét nghiệm.
- Support Vector Machine (SVM): Thuật toán này không đơn thuần chia ranh giới mà cố gắng tạo một “vùng đệm” (margin) rộng nhất có thể giữa hai nhóm. Để dễ hình dung, bạn có thể tưởng tượng bạn xây một con mương rộng nhất có thể để ngăn cách đội quân hai nước đối đầu.
- Naïve Bayes: Dựa trên định lý xác suất Bayes, thuật toán này cực kỳ nhanh và hiệu quả cho các bài toán phân loại văn bản, chẳng hạn như lọc email rác (spam) hoặc phân tích tâm lý khách hàng qua bình luận.
- Neural Networks: Lấy cảm hứng từ mạng nơ-ron thần kinh trong não bộ của con người, thuật toán này có thể học được những quy luật phức tạp (nhận diện khuôn mặt/âm thanh, phân tích sóng não). Đây là nền tảng của Deep Learning – công cụ đứng sau những chatbot AI đang hiện hành.
### 3.3. Nhóm Phân cụm (Clustering) – Khám phá cấu trúc ngầm
Mục tiêu: Tự động gom nhóm dữ liệu dựa trên sự tương đồng mà không cần nhãn.
- K-Means: Thuật toán hoạt động bằng cách chọn điểm “k” trung tâm và “lôi kéo” các điểm dữ liệu xung quanh về phía mình. Qua nhiều vòng lặp, các cụm sẽ dần hình thành rõ nét.
- Gaussian Mixture Models (GMM): Khác với K-Means (phân cụm theo hình tròn), GMM linh hoạt hơn khi coi mỗi cụm là một phân phối hình oval, cho phép các cụm chồng lấn lên nhau một cách tự nhiên hơn.
- Principle Component Analysis (PCA): Mặc dù thường được dùng để giảm chiều dữ liệu, PCA giúp ta nhìn ra những thành phần “cốt lõi” nhất của dữ liệu, từ đó hỗ trợ việc phân cụm trở nên chính xác và trực quan hơn trên đồ thị.
## 4. Quy trình các bước của một dự án Machine Learning

### 4.1. Định nghĩa bài toán

Đây là bước nền tảng quyết định sự thành bại, giúp tránh việc giải quyết sai vấn đề. Cần làm rõ 3 yếu tố:

* **Mục tiêu:** Xác định loại bài toán là Dự đoán (Regression), Phân loại (Classification) hay Gom nhóm (Clustering).
* **Giá trị thực tế:** Kết quả giải quyết nỗi đau nào cho người dùng/doanh nghiệp.
* **Thước đo thành công:** Chọn metrics (chỉ số đánh giá) đúng ngay từ đầu để nghiệm thu khách quan.

### 4.2. Thu thập và Xử lý dữ liệu

Giai đoạn tốn nhiều thời gian nhất (60-80%), tuân theo nguyên tắc "Garbage In, Garbage Out".

**4.2.1. Thu thập dữ liệu**

* **Nguồn:** Nội bộ (SQL, Log), Công khai (Kaggle), Web Scraping (Code), API, hoặc IoT/Sensors.
* **Định dạng:** Có cấu trúc (Excel, SQL), Bán cấu trúc (JSON, XML), Phi cấu trúc (Text, Ảnh, Video).

**4.2.2. Tiền xử lý dữ liệu (4 bước)**

1. **Làm sạch:** Xử lý dữ liệu thiếu (gán/xóa), loại bỏ trùng lặp, sửa lỗi định dạng để đảm bảo tính nhất quán.
2. **Tích hợp:** Kết hợp dữ liệu từ nhiều nguồn, khớp lược đồ (schema mapping) và khử trùng lặp giữa các nguồn.
3. **Chuyển đổi:** Chuẩn hóa (Scaling) cho các thuật toán đo khoảng cách, Mã hóa biến phân loại (One-hot/Label encoding), và Trích xuất đặc trưng (Feature engineering).
4. **Giảm thiểu:** Giảm chiều dữ liệu (PCA), lựa chọn đặc trưng quan trọng hoặc lấy mẫu đại diện để tăng tốc độ huấn luyện.

### 4.3. Lựa chọn và Huấn luyện mô hình

Mục tiêu là tìm mô hình cân bằng, tránh quá đơn giản (kém chính xác) hoặc quá phức tạp (overfit).

* **Hiểu bài toán:** Xác định dữ liệu là số hay phân loại, có nhãn hay không.
* **Gợi ý mô hình:**
* *Hồi quy (Regression):* Linear Regression, Decision Trees, Random Forest, Neural Networks.
* *Phân loại (Classification):* Logistic Regression, SVM, k-NN, Neural Networks.
* *Phân cụm (Clustering):* k-Means, Hierarchical Clustering, DBSCAN.



### 4.4. Đánh giá mô hình

Chia dữ liệu thành **Tập Huấn luyện (Train)** để học và **Tập Kiểm tra (Test)** để đánh giá trên dữ liệu mới. Nên dùng kỹ thuật **Kiểm định chéo (k-fold cross-validation)** để đánh giá khách quan hơn.

**Chỉ số đánh giá (Metrics):**

* *Hồi quy:* MSE, MAE, R-squared.
* *Phân loại:* Accuracy, Precision, Recall, F1-score.

### 4.5. Các Kỹ thuật Tối ưu Lựa chọn Mô hình

* **Grid Search (Tìm kiếm lưới):** Thử nghiệm tất cả tổ hợp tham số. Chính xác nhưng tốn tài nguyên.
* **Random Search (Tìm kiếm ngẫu nhiên):** Thử nghiệm tập con ngẫu nhiên. Nhanh hơn và hiệu quả tương đương Grid Search.
* **Bayesian Optimization:** Sử dụng xác suất để dự đoán và chọn tham số tốt nhất một cách thông minh.
* **Dựa trên kiểm định chéo:** Chọn mô hình có hiệu suất trung bình tốt nhất qua nhiều lần chia dữ liệu để tránh overfit.
## 5. Xây dựng một Mô hình Học Máy Cơ bản

Chúng ta sẽ dùng Python và scikit-learn, thư viện này rất thân thiện với người mới và xử lý hầu hết phần phức tạp mà không cần đào sâu vào toán học ngay từ đầu. Mục tiêu là làm cho phần này thực tế để bạn có thể tự thử và thấy cách các khái niệm từ các phần trước kết hợp trong code.

### 5.1 Công cụ Bạn Cần

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

### 5.2 Ví dụ Vấn đề – Phân loại Hoa Iris

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

### 5.3 Mã Từng Bước

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

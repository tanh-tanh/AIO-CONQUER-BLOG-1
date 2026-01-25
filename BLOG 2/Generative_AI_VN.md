# Lộ Trình Học Generative AI

Thị trường AI đang bùng nổ với tốc độ chưa từng có. Báo cáo của McKinsey năm 2024 cho thấy nhu cầu tuyển dụng AI Engineer tăng 312% chỉ trong 18 tháng. Nhưng có một khoảng cách lớn: 73% người muốn chuyển sang lĩnh vực này không biết bắt đầu từ đâu.

Mục này được tạo ra nhằm mục đích giúp hiểu rõ bối cảnh hiện tại và có cái nhìn tổng quất về kiến thức cần học trong lộ trình Generative AI.

---

## Nền Tảng Đầu Tiên: Python

### Tại sao Python là điểm khởi đầu

Dữ liệu từ khảo sát Stack Overflow 2024 cho thấy 94% thư viện AI được viết bằng Python. Đây không phải sự lựa chọn ngẫu nhiên - Python có cú pháp đơn giản, thư viện phong phú, và cộng đồng lớn nhất trong lĩnh vực này.
### Kiến thức cần học

**Cú pháp cơ bản**

- Biến và kiểu dữ liệu
- Vòng lặp và câu điều kiện
- Hàm và lớp (class)
- Xử lý lỗi cơ bản

**Thư viện xử lý dữ liệu**

- NumPy: Làm việc với mảng số và ma trận
- Pandas: Xử lý dữ liệu dạng bảng
- Matplotlib: Vẽ biểu đồ và trực quan hóa

**Framework web**

- Flask hoặc FastAPI để sau này triển khai mô hình lên web

**Thời gian đầu tư:** 40-50 giờ thực hành

Một phát hiện quan trọng: Những người dành quá nhiều thời gian học Python hoàn hảo thường bị chậm tiến độ tổng thể. Mục tiêu ở giai đoạn này là đủ tốt để đọc và viết code AI, không cần trở thành chuyên gia Python.

---

## Machine Learning Cơ Bản: Máy Tính Hiểu Văn Bản Như Thế Nào

### Vấn đề cốt lõi

Máy tính chỉ hiểu số, không hiểu chữ. Mọi kỹ thuật xử lý ngôn ngữ tự nhiên đều bắt đầu bằng việc chuyển đổi văn bản thành các con số mà máy có thể xử lý.

### Các phương pháp chính

**One-Hot Encoding**

Phương pháp đơn giản nhất, xuất hiện từ thập niên 1950. Mỗi từ được biểu diễn bằng một vector có một vị trí bằng 1, còn lại bằng 0.

Ví dụ:

```
"mèo" → [1, 0, 0]
"chó" → [0, 1, 0]
"gà"  → [0, 0, 1]
```

Hạn chế lớn: Không thể hiện được mối quan hệ giữa các từ. "Mèo" và "chó" đều là động vật nhưng vector của chúng không liên quan gì đến nhau.

**Túi Từ (Bag of Words)**

Đếm số lần mỗi từ xuất hiện trong văn bản. Phương pháp này vẫn được dùng cho các bài toán phân loại đơn giản.

**TF-IDF (term frequency-inverse document frequency)**

Phát triển từ thập niên 1970-1990, TF-IDF đánh giá tầm quan trọng của từ dựa trên:

- Từ xuất hiện nhiều trong văn bản hiện tại
- Nhưng ít xuất hiện trong các văn bản khác

Ứng dụng: Vẫn được sử dụng trong công cụ tìm kiếm và lọc thư rác.

**Word2Vec - Bước đột phá năm 2013**

Đây là thời điểm chuyển mình quan trọng. Word2Vec chuyển từ thành các vector dày đặc sao cho các từ có nghĩa gần nhau có vector gần nhau trong không gian.

Điều đặc biệt: Có thể thực hiện phép toán với từ

```
"Vua" - "Nam" + "Nữ" ≈ "Nữ hoàng"
```

Kể từ Word2Vec, mọi phương pháp hiện đại đều dựa trên ý tưởng nhúng từ (word embedding) này.

---

## Deep Learning Cơ Bản: Mạng Neural Hoạt Động Ra Sao

### Lịch sử ngắn gọn

Mạng neural không phải khái niệm mới - Warren McCulloch và Walter Pitts đề xuất từ năm 1943. Nhưng chỉ sau 2012 với AlexNet, công nghệ này mới thực sự khả thi nhờ GPU và dữ liệu lớn.

### Cấu trúc cơ bản

Một mạng neural gồm ba phần:

- **Lớp đầu vào:** Nhận dữ liệu
- **Lớp ẩn:** Học các đặc trưng
- **Lớp đầu ra:** Đưa ra dự đoán

### Hai quá trình quan trọng

**Lan truyền tiến (Forward Propagation)**

Dữ liệu đi từ đầu vào đến đầu ra, qua từng lớp, tạo ra dự đoán cuối cùng.

**Lan truyền ngược (Backpropagation)**

So sánh dự đoán với kết quả thực tế, tính toán sai số, rồi điều chỉnh các trọng số trong mạng. Thuật toán này được formalize bởi Rumelhart và cộng sự năm 1986, là nền tảng của mọi mô hình học sâu hiện đại.

### Hàm kích hoạt

Các hàm này tạo nên tính phi tuyến - khả năng học các mẫu phức tạp. Nghiên cứu của Goodfellow (2016) chỉ ra tầm quan trọng của chúng:

- **ReLU** (2011): Đơn giản nhưng hiệu quả, là lựa chọn mặc định
- **Sigmoid:** Dùng cho phân loại nhị phân
- **Softmax:** Phân loại nhiều lớp

### Hàm mất mát và bộ tối ưu

Hàm mất mát đo lường sai số. Bộ tối ưu (như SGD, Adam) quyết định cách cập nhật trọng số.

Adam optimizer (Kingma & Ba, 2014) hiện là chuẩn mực trong hầu hết ứng dụng do khả năng tự điều chỉnh tốc độ học.

---

## NLP Nâng Cao: Từ RNN Đến Transformer

### Mạng Neural Hồi Tiếp (RNN)

RNN ra đời năm 1986, được thiết kế để xử lý dữ liệu tuần tự - văn bản, âm thanh, chuỗi thời gian. Khác với mạng thông thường, RNN có kết nối quay ngược, tạo thành "bộ nhớ".

**Vấn đề:** Mất dần độ chính xác (vanishing gradient) khi chuỗi dài - mạng "quên" thông tin từ xa.

### LSTM - Bộ Nhớ Ngắn Hạn Dài

Hochreiter & Schmidhuber giới thiệu LSTM năm 1997 để giải quyết vấn đề trên. LSTM có cấu trúc "cổng":

- Cổng quên: Quyết định bỏ thông tin nào
- Cổng đầu vào: Quyết định lưu thông tin nào
- Cổng đầu ra: Quyết định xuất thông tin gì

LSTM thống trị NLP từ 2013 đến 2017, cho đến khi Transformer xuất hiện.

### Transformer: Kiến Trúc Thay Đổi Cuộc Chơi

**"Attention Is All You Need"** - paper năm 2017 của Vaswani và cộng sự

Đây là bài báo quan trọng nhất của NLP thập kỷ qua. Transformer loại bỏ việc xử lý tuần tự, thay bằng cơ chế chú ý (attention).

**Self-Attention**

Cho phép mỗi vị trí trong chuỗi "nhìn" tất cả vị trí khác cùng lúc. Khác với RNN xử lý từng từ một, Transformer xử lý song song - nhanh hơn đáng kể.

Ví dụ: Trong câu "Con mèo đuổi theo con chuột"

- Từ "mèo" cần chú ý đến "đuổi theo" (hành động) và "chuột" (đối tượng)
- Cơ chế attention tự động học được điều này

**Multi-Head Attention**

Thay vì một cơ chế chú ý, dùng nhiều "đầu" để học các khía cạnh khác nhau của mối quan hệ.

**Positional Encoding**

Vì không xử lý tuần tự, cần thêm thông tin về vị trí của từ trong câu.

**Tác động:**

Kể từ 2017, Transformer đã tạo ra:

- BERT (2018): Mô hình tiền huấn luyện hai chiều
- GPT series (2018-2024): Mô hình ngôn ngữ tự hồi quy
- T5, BART, và hàng trăm biến thể khác

Mọi mô hình ngôn ngữ lớn hiện nay (GPT-4, Claude, Gemini) đều dựa trên kiến trúc Transformer.

---

## Làm Việc Với Mô Hình Ngôn Ngữ Lớn

### Bối cảnh mới

Thời đại huấn luyện mô hình từ đầu đã qua. Nghiên cứu của Stanford 2024 cho thấy 89% ứng dụng AI sử dụng các mô hình đã được huấn luyện sẵn.

### LangChain - Framework kết nối

LangChain giúp đơn giản hóa việc làm việc với các mô hình ngôn ngữ lớn. Theo thống kê GitHub, đây là framework phát triển nhanh nhất trong hệ sinh thái AI (2023-2024).

**Thành phần chính:**

- **Chains (Chuỗi):** Kết nối các bước xử lý
- **Agents (Tác nhân):** Mô hình tự quyết định công cụ nào cần dùng
- **Memory (Bộ nhớ):** Lưu trữ lịch sử hội thoại
- **Retrievers (Bộ truy xuất):** Tìm kiếm thông tin liên quan

### RAG - Tăng Cường Bằng Truy Xuất

RAG được giới thiệu bởi Lewis và cộng sự năm 2020, giải quyết vấn đề lớn: các mô hình ngôn ngữ không biết về dữ liệu riêng của bạn và có thể tạo ra thông tin sai lệch (hiện tượng ảo giác).

**Cách hoạt động:**

1. Chuyển tài liệu thành các vector nhúng
2. Lưu vào cơ sở dữ liệu vector
3. Khi người dùng hỏi, tìm tài liệu liên quan
4. Đưa tài liệu và câu hỏi cho mô hình
5. Mô hình tạo câu trả lời dựa trên tài liệu thực tế

**Kết quả từ nghiên cứu:**

- Giảm ảo giác (illusion) 60-80% (Microsoft Research, 2023)
- Tăng độ chính xác về sự thật đáng kể
- Cho phép mô hình truy cập dữ liệu độc quyền

Đây là kiến trúc đằng sau phần lớn chatbot doanh nghiệp hiện nay.

---

## Cơ Sở Dữ Liệu Vector

### Tại sao cần loại database mới

Cơ sở dữ liệu truyền thống tìm kiếm khớp chính xác. Cơ sở dữ liệu vector tìm kiếm theo ngữ nghĩa - tìm ý nghĩa, không phải từ khóa.

### Các lựa chọn chính

**ChromaDB**

- Mã nguồn mở, dễ cài đặt
- Phù hợp cho thử nghiệm và ứng dụng vừa nhỏ
- Tích hợp tốt với LangChain

**FAISS (Tìm Kiếm Tương Đồng của Facebook AI)**

- Hiệu năng mức production
- Có thể xử lý hàng tỷ vector
- Được sử dụng bởi Facebook, Spotify và các tập đoàn công nghệ lớn

**Pinecone**

- Dịch vụ được quản lý, tự động mở rộng
- Đánh đổi: Phải trả phí nhưng không cần quản lý hạ tầng

**So sánh hiệu năng:**

- ChromaDB: 1 triệu vector, thời gian truy vấn ~2-3 giây
- FAISS: 1 triệu vector, thời gian truy vấn ~100-200 mili giây
- Pinecone: Tương đương FAISS nhưng được quản lý toàn phần

Lựa chọn phụ thuộc vào quy mô và nguồn lực.

---

## Tinh Chỉnh Mô Hình

### Tùy chỉnh cho lĩnh vực cụ thể

Các mô hình được huấn luyện sẵn giỏi các tác vụ chung. Nhưng với các lĩnh vực chuyên biệt (y khoa, luật, tiếng Việt đặc thù), tinh chỉnh là cần thiết.

### LoRA

LoRA được giới thiệu bởi Hu và cộng sự năm 2021, là bước đột phá về hiệu quả. Thay vì cập nhật toàn bộ tham số (tốn kém), LoRA đóng băng trọng số đã huấn luyện và chỉ thêm các ma trận có thể huấn luyện.

**Kết quả:**

- Giảm 99% tham số cần huấn luyện 
- Giảm yêu cầu bộ nhớ 3 lần
- Thời gian huấn luyện nhanh hơn 2-3 lần
- Hiệu năng tương đương tinh chỉnh toàn phần

### QLoRA

Dettmers và cộng sự năm 2023 kết hợp lượng tử hóa (quantization) với LoRA. Cho phép tinh chỉnh mô hình 70 tỷ tham số trên một GPU duy nhất.

### Yêu cầu về dữ liệu

Nghiên cứu từ nhiều bài báo cho thấy:

- 100-500 mẫu: Mức tối thiểu khả thi
- 1,000-5,000: Kết quả tốt
- 10,000+: Lợi ích giảm dần

Chất lượng dữ liệu quan trọng hơn số lượng. 500 mẫu chất lượng cao, đa dạng tốt hơn 5,000 mẫu lặp lại.

---

## Triển Khai Production

### Từ notebook đến ứng dụng thực tế

Theo khảo sát với 45 nhóm AI, 67% dự án thất bại ở giai đoạn triển khai. Không phải vì mô hình không tốt - mà vì các thách thức về hạ tầng và vận hành.

### Các lựa chọn triển khai

**HuggingFace Spaces**

- Có gói miễn phí
- Phù hợp cho demo và dự án nhỏ
- Tích hợp Gradio/Streamlit
- Hạn chế: Tài nguyên có giới hạn

**AWS (Amazon Web Services)**

- Bedrock: Dịch vụ mô hình ngôn ngữ được quản lý
- SageMaker: Triển khai mô hình tùy chỉnh
- Lambda: Suy luận không máy chủ
- Đánh đổi: Phức tạp nhưng linh hoạt

**Azure OpenAI Service**

- Cấp độ doanh nghiệp
- Tính năng tuân thủ tích hợp sẵn
- Truy cập trực tiếp GPT-4
- Chi phí: Cao hơn nhưng dự đoán được

**Phân tích chi phí từ dự án thực tế:**

- HuggingFace: $0-50/tháng
- AWS: $100-1,000/tháng tùy sử dụng
- Azure: $200-1,500/tháng cấp doanh nghiệp

### Giám sát và tối ưu

**LangSmith** (Công cụ giám sát của LangChain)

- Theo dõi mỗi lần gọi mô hình
- Debug các chuỗi xử lý
- Đo độ trễ và chi phí
- Kiểm thử A/B với prompts

AI production không kết thúc ở triển khai. Giám sát liên tục và cải tiến là chìa khóa thành công.

---
Lộ trình trên là toàn bộ những kiến thức cơ bản và tổng quát để bắt đầu với Generative AI. Nếu muốn có một lộ trình chi tiết và đi sâu vào từng phần thì các trang như roadmap.sh, Opencourse từ các trường nổi tiếng (MIT, Standford, ...) hay các khóa học trực tuyến khác (Coursera, Udemy, Youtube, ...) cũng rất đa dạng và đầy đủ. 

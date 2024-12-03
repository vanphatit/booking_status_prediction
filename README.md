
# Hotel Reservation Prediction

Dự án này sử dụng mô hình cây quyết định CART (Classification and Regression Tree) để dự đoán trạng thái đặt phòng khách sạn dựa trên dữ liệu đầu vào. Dự án bao gồm các bước từ tiền xử lý dữ liệu, cân bằng dữ liệu, xây dựng mô hình, đến kiểm thử và đánh giá hiệu suất mô hình.

## Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [Cấu trúc dự án](#cấu-trúc-dự-án)
3. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
4. [Kết quả](#kết-quả)
5. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
6. [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Giới thiệu

Mục tiêu của dự án là xây dựng một mô hình dự đoán trạng thái đặt phòng (canceled hoặc confirmed) dựa trên dữ liệu đầu vào. Dự án sử dụng cây quyết định CART để giải quyết bài toán phân loại, với các bước như:
- Tiền xử lý dữ liệu: làm sạch và cân bằng dữ liệu bằng SMOTE-Tomek.
- Huấn luyện và tối ưu mô hình Decision Tree.
- Đánh giá hiệu suất dựa trên các chỉ số như độ chính xác (accuracy), F1-Score, và ROC-AUC.

---

## Cấu trúc dự án

- `main.py`: File chứa mã nguồn chính để huấn luyện mô hình.
- `test_model.py`: File kiểm thử mô hình trên tập dữ liệu mới.
- `Hotel Reservations.csv`: Dataset gốc sử dụng cho dự án.
- `test_data.csv`: Dữ liệu test được tách ra từ dữ liệu huấn luyện.
- `test_results.csv`: Kết quả dự đoán của mô hình trên tập test.
- `model.pkl`: Mô hình Decision Tree đã được lưu trữ.
- `images/`: Thư mục chứa các biểu đồ minh họa (ma trận nhầm lẫn, biểu đồ ROC, trọng số đặc trưng, v.v.).

---

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python (phiên bản >= 3.8). Cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình
Chạy file `main.py` để huấn luyện mô hình và lưu kết quả:
```bash
python main.py
```

### 3. Kiểm thử mô hình
Chạy file `test_model.py` để kiểm tra hiệu suất mô hình trên tập dữ liệu test:
```bash
python test_model.py
```

---

## Kết quả

1. **Hiệu suất mô hình**:
   - Độ chính xác: 89.25%
   - F1-Score trung bình: 0.89
   - AUC: 0.90

2. **Biểu đồ minh họa**:
   - Ma trận nhầm lẫn (Confusion Matrix): Thể hiện số lượng mẫu dự đoán đúng/sai.
   - Biểu đồ ROC: Hiệu quả của mô hình trong việc phân biệt hai lớp.
   - Trọng số đặc trưng: Phân tích tầm quan trọng của các đặc trưng đầu vào.

---

## Yêu cầu hệ thống

- Python >= 3.8
- Các thư viện cần thiết:
  ```plaintext
  imbalanced-learn==0.12.4
  matplotlib==3.9.2
  numpy==2.1.3
  pandas==2.2.3
  scikit-learn==1.5.2
  seaborn==0.13.2
  joblib==1.4.2
  ```

---

## Tài liệu tham khảo

1. **Sách**:
   - Aurelien Geron, *Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow*, O’Reilly, 2019.

2. **Trang web**:
   - [CART Classification and Regression Tree in Machine Learning](https://www.geeksforgeeks.org/cart-classification-and-regression-tree-in-machine-learning/)
   - [Decision Trees with CART Algorithm](https://medium.com/geekculture/decision-trees-with-cart-algorithm-7e179acee8ff)
   - [Kaggle Dataset: Hotel Reservations Classification Dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)

3. **Các tài liệu khác**:
   - Các tài liệu tham khảo trong quá trình xây dựng và phát triển đề tài.

---

## **Tác giả**
Dự án được thực hiện bởi [Lê Văn Phát - 22110196]. Các vấn đề thắc mắc hoặc góp ý, vui lòng liên hệ qua [vanphat15it@gmail.com].
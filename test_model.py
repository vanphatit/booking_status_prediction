import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc file test data
test_data = pd.read_csv('test_data.csv')

# 2. Load mô hình đã lưu
model_filename = 'model.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

print("Mô hình đã được load từ file:", model_filename)

# 3. Chuẩn bị dữ liệu
X_test = test_data.drop('booking_status', axis=1)  # Loại bỏ cột nhãn
y_test = test_data['booking_status']  # Lấy cột nhãn

# 4. Dự đoán với mô hình đã load
y_pred = loaded_model.predict(X_test)
y_prob = loaded_model.predict_proba(X_test)[:, 1]

# 5. Tính toán tỷ lệ dự đoán
accuracy = (y_pred == y_test).mean()
print(f"Độ chính xác trên tập test: {accuracy:.2%}")

# Tạo bảng thống kê
results_summary = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
summary_table = results_summary.value_counts().unstack(fill_value=0)
summary_table.columns = ['Predicted as 0', 'Predicted as 1']
summary_table.index = ['Actual 0', 'Actual 1']

print("\n### Tóm tắt kết quả ###")
print(summary_table)

# 6. Xuất kết quả ra file CSV
output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
output.to_csv('test_results.csv', index=False)
print("Kết quả đã được lưu vào file: test_results.csv")

# 7. Biểu đồ Confusion Matrix
conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', cbar=False)
plt.title('Ma trận nhầm lẫn', fontsize=16)
plt.xlabel('Predicted label', fontsize=14)
plt.ylabel('Actual label', fontsize=14)
plt.show()

# 8. Biểu đồ ROC
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc="lower right")
plt.show()
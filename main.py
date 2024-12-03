# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTETomek

# 1. Đọc dataset
data = pd.read_csv('Hotel Reservations.csv')

# 2. Làm sạch dữ liệuD
data = data.dropna()

print(f"Số lượng mẫu ban đầu: {data.shape[0]}")


# Loại bỏ ngoại lệ (dựa trên IQR)
numeric_columns = ['lead_time', 'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'avg_price_per_room', 'no_of_special_requests']
for col in numeric_columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data[col] >= Q1 - 1.5 * IQR) & (data[col] <= Q3 + 1.5 * IQR)]

# Mã hóa dữ liệu phân loại
label_encoder = LabelEncoder()
data['room_type_reserved'] = label_encoder.fit_transform(data['room_type_reserved'])
data['market_segment_type'] = label_encoder.fit_transform(data['market_segment_type'])
data['booking_status'] = label_encoder.fit_transform(data['booking_status'])

# Feature Engineering
data['total_nights'] = data['no_of_weekend_nights'] + data['no_of_week_nights']
data['price_per_person'] = data['avg_price_per_room'] / (data['no_of_adults'] + data['no_of_children'] + 1)

# Chọn các biến đầu vào và mục tiêu
X = data[['lead_time', 'room_type_reserved', 'market_segment_type', 'no_of_adults', 'no_of_children',
          'avg_price_per_room', 'no_of_special_requests', 'total_nights', 'price_per_person',
          'required_car_parking_space', 'no_of_previous_cancellations']]
y = data['booking_status']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Cân bằng dữ liệu bằng SMOTE-Tomek
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

# Tách dữ liệu thành train/test (90:10)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)

# Xuất dữ liệu test ra file CSV
test_data = X_test.copy()
test_data['booking_status'] = y_test  # Thêm cột mục tiêu vào dữ liệu test
test_data.to_csv('test_data.csv', index=False)

print("Dữ liệu test đã được lưu thành file: test_data.csv")

# 3. Tối ưu hóa Decision Tree với GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'min_impurity_decrease': [0.0, 0.01, 0.02],
    'ccp_alpha': [0.0, 0.01, 0.02]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Tìm mô hình tối ưu
best_model = grid_search.best_estimator_

# Dự đoán với tập test
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 4. Đánh giá mô hình
print("### Báo cáo hiệu suất mô hình ###")
print(classification_report(y_test, y_pred))
print(f"Độ chính xác: {accuracy_score(y_test, y_pred)}")

# Ma trận nhầm lẫn với nhãn rõ ràng hơn
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Canceled', 'Confirmed'], 
            yticklabels=['Canceled', 'Confirmed'])
plt.title('Ma trận nhầm lẫn', fontsize=16)
plt.xlabel('Predicted label', fontsize=14)
plt.ylabel('True label', fontsize=14)
plt.show()

# Biểu đồ ROC và AUC
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

# 5. Biểu đồ trọng số của các đặc trưng
feature_importances = best_model.feature_importances_
features = X.columns  # Tên các đặc trưng
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Vẽ biểu đồ
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Trọng số của các đặc trưng', fontsize=16) 
plt.xlabel('Mức độ quan trọng', fontsize=14) 
plt.ylabel('Đặc trưng', fontsize=14) 
plt.show()

# 6. Biểu đồ phân phối của biến mục tiêu (booking_status)
plt.figure(figsize=(8, 6)) 
sns.countplot(data=pd.DataFrame(y_resampled, columns=['booking_status']), x='booking_status', palette='viridis') 
plt.title('Phân phối biến mục tiêu sau khi cân bằng', fontsize=16) 
plt.xlabel('Booking Status (0: Canceled, 1: Confirmed)', fontsize=14) 
plt.ylabel('Số lượng', fontsize=14) 
plt.show()

# 7. Biểu đồ phân tán các đặc trưng chính
plt.figure(figsize=(10, 6)) 
sns.scatterplot(data=pd.concat([X_resampled, y_resampled], axis=1), x='lead_time', y='price_per_person', hue=y_resampled, palette='viridis', alpha=0.7) 
plt.title('Mối quan hệ giữa lead_time và price_per_person', fontsize=16) 
plt.xlabel('Lead Time', fontsize=14) 
plt.ylabel('Price Per Person', fontsize=14) 
plt.legend(title='Booking Status (0: Canceled, 1: Confirmed)', fontsize=12) 
plt.show()

# 8. Biểu đồ tương quan giữa các đặc trưng (Correlation Heatmap)
plt.figure(figsize=(12, 10)) 
correlation_matrix = pd.concat([X_resampled, y_resampled], axis=1).corr() 
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True) 
plt.title('Ma trận tương quan giữa các đặc trưng', fontsize=16) 
plt.show()

# 9. Boxplot cho các đặc trưng quan trọng
plt.figure(figsize=(10, 6)) 
sns.boxplot(data=pd.concat([X_resampled, y_resampled], axis=1), x=y_resampled, y='lead_time', palette='viridis') 
plt.title('Boxplot của Lead Time theo Booking Status', fontsize=16) 
plt.xlabel('Booking Status (0: Canceled, 1: Confirmed)', fontsize=14) 
plt.ylabel('Lead Time', fontsize=14) 
plt.show()

plt.figure(figsize=(10, 6)) 
sns.boxplot(data=pd.concat([X_resampled, y_resampled], axis=1), x=y_resampled, y='price_per_person', palette='viridis') 
plt.title('Boxplot của Price Per Person theo Booking Status', fontsize=16) 
plt.xlabel('Booking Status (0: Canceled, 1: Confirmed)', fontsize=14) 
plt.ylabel('Price Per Person', fontsize=14) 
plt.show()

# 10. Histogram của các đặc trưng liên tục
plt.figure(figsize=(10, 6)) 
sns.histplot(data=X_resampled, x='lead_time', bins=30, kde=True, color='purple') 
plt.title('Histogram của Lead Time', fontsize=16) 
plt.xlabel('Lead Time', fontsize=14) 
plt.ylabel('Số lượng', fontsize=14) 
plt.show()

plt.figure(figsize=(10, 6)) 
sns.histplot(data=X_resampled, x='price_per_person', bins=30, kde=True, color='green') 
plt.title('Histogram của Price Per Person', fontsize=16) 
plt.xlabel('Price Per Person', fontsize=14) 
plt.ylabel('Số lượng', fontsize=14) 
plt.show()

# 11. Biểu đồ Cây quyết định (Tree Depth Analysis)
plt.figure(figsize=(12, 8)) 
plot_tree(best_model, filled=True, feature_names=X.columns, class_names=['Canceled', 'Confirmed'], fontsize=10) 
plt.title('Cây quyết định - Decision Tree', fontsize=16) 
plt.show()

# 12. Biểu đồ hình tròn (Pie Chart) cho trọng số đặc trưng
plt.figure(figsize=(10, 8)) 
plt.pie(importance_df['Importance'], labels=importance_df['Feature'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', n_colors=len(importance_df))) 
plt.title('Tầm quan trọng của các đặc trưng', fontsize=16) 
plt.show()

# 13. Lưu mô hình đã tối ưu
model_filename = 'model.pkl' 
with open(model_filename, 'wb') as file: pickle.dump(best_model, file)

print(f"Mô hình đã được lưu thành file: {model_filename}")
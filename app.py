# ================================
# 1. Import thư viện
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
import pickle
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# ================================
# 2. Cấu hình trang
# ================================
st.set_page_config(
    page_title="DỰ BÁO CÔNG SUẤT PHÁT - NHÀ MÁY ĐIỆN GIÓ BT1",
    page_icon="🌬️",
    layout="wide"
)

# ================================
# 3. Tiêu đề
# ================================
st.title("🌬️ DỰ BÁO CÔNG SUẤT PHÁT - NHÀ MÁY ĐIỆN GIÓ BT1")
st.markdown("---")

# ================================
# 4. Khởi tạo session state
# ================================
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'df_valid' not in st.session_state:
    st.session_state.df_valid = None
if 'training_info' not in st.session_state:
    st.session_state.training_info = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None

# ================================
# 5. Đường dẫn lưu mô hình
# ================================
MODEL_PATH = "model_xgboost.pkl"
SCALER_PATH = "scaler.pkl"
INFO_PATH = "training_info.pkl"

# ================================
# 6. Hàm load mô hình đã lưu
# ================================
def load_saved_model():
    """Tải mô hình đã lưu từ file"""
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            with open(INFO_PATH, 'rb') as f:
                training_info = pickle.load(f)
            
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.training_info = training_info
            st.session_state.model_trained = True
            return True
    except Exception as e:
        st.warning(f"Không thể load mô hình cũ: {e}")
    return False

# ================================
# 7. Hàm lưu mô hình
# ================================
def save_model(model, scaler, training_info):
    """Lưu mô hình và scaler vào file"""
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(INFO_PATH, 'wb') as f:
        pickle.dump(training_info, f)

# ================================
# 8. Hàm xóa mô hình
# ================================
def delete_model():
    """Xóa tất cả file mô hình"""
    try:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            os.remove(SCALER_PATH)
        if os.path.exists(INFO_PATH):
            os.remove(INFO_PATH)
        st.session_state.model_trained = False
        st.session_state.model = None
        st.session_state.scaler = None
        st.session_state.training_info = None
        st.session_state.df_valid = None
        st.session_state.df_original = None
        return True
    except Exception as e:
        st.error(f"Lỗi khi xóa: {e}")
        return False

# ================================
# 9. Hàm vẽ biểu đồ phân bố
# ================================
def plot_distributions(df_valid, df_original):
    """Vẽ biểu đồ phân bố tốc độ gió và nhiệt độ"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram tốc độ gió
    axes[0, 0].hist(df_valid['WindSpeed'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df_valid['WindSpeed'].mean(), color='red', linestyle='--', linewidth=2, label=f'TB: {df_valid["WindSpeed"].mean():.2f} m/s')
    axes[0, 0].set_xlabel('Tốc độ gió (m/s)')
    axes[0, 0].set_ylabel('Tần suất')
    axes[0, 0].set_title('Phân bố tốc độ gió (dữ liệu đạt)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Histogram nhiệt độ
    axes[0, 1].hist(df_valid['Temp'], bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(df_valid['Temp'].mean(), color='red', linestyle='--', linewidth=2, label=f'TB: {df_valid["Temp"].mean():.2f} °C')
    axes[0, 1].set_xlabel('Nhiệt độ (°C)')
    axes[0, 1].set_ylabel('Tần suất')
    axes[0, 1].set_title('Phân bố nhiệt độ (dữ liệu đạt)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Boxplot tốc độ gió
    wind_data = [df_original['WindSpeed'].dropna(), df_valid['WindSpeed']]
    bp1 = axes[1, 0].boxplot(wind_data, labels=['Trước lọc', 'Sau lọc'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightcoral')
    bp1['boxes'][1].set_facecolor('lightgreen')
    axes[1, 0].set_ylabel('Tốc độ gió (m/s)')
    axes[1, 0].set_title('So sánh tốc độ gió trước và sau khi lọc nhiễu')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Boxplot nhiệt độ
    temp_data = [df_original['Temp'].dropna(), df_valid['Temp']]
    bp2 = axes[1, 1].boxplot(temp_data, labels=['Trước lọc', 'Sau lọc'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightcoral')
    bp2['boxes'][1].set_facecolor('lightgreen')
    axes[1, 1].set_ylabel('Nhiệt độ (°C)')
    axes[1, 1].set_title('So sánh nhiệt độ trước và sau khi lọc nhiễu')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_power_distribution(df_valid):
    """Vẽ biểu đồ phân bố công suất"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram công suất
    axes[0].hist(df_valid['Power'], bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[0].axvline(df_valid['Power'].mean(), color='red', linestyle='--', linewidth=2, label=f'TB: {df_valid["Power"].mean():.2f} kW')
    axes[0].set_xlabel('Công suất (kW)')
    axes[0].set_ylabel('Tần suất')
    axes[0].set_title('Phân bố công suất phát')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Công suất theo tốc độ gió
    axes[1].scatter(df_valid['WindSpeed'], df_valid['Power'], alpha=0.5, c='steelblue', s=10)
    axes[1].set_xlabel('Tốc độ gió (m/s)')
    axes[1].set_ylabel('Công suất (kW)')
    axes[1].set_title('Quan hệ giữa tốc độ gió và công suất')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ================================
# 10. Sidebar
# ================================
with st.sidebar:
    st.header("📘 Hướng dẫn")
    st.markdown("""
    **Các bước thực hiện:**
    
    **🔄 LẦN ĐẦU TIÊN (Chỉ 1 lần):**
    1. 📁 Upload file dữ liệu lịch sử (CSV)
    2. 🚀 Huấn luyện mô hình
    3. 💾 Mô hình sẽ được lưu lại
    
    **⚡ CÁC LẦN SAU:**
    1. 📊 Nhập dữ liệu dự báo
    2. 🚀 Dự báo ngay (không cần upload lại)
    
    **Yêu cầu file dữ liệu:**
    - Định dạng: CSV
    - Các cột: `PCTimeStamp`, `WindSpeed`, `Temp`, `Power`
    
    **📝 Cách nhập dữ liệu:**
    - Mỗi dòng là 1 giá trị số
    - Không để dòng trống
    """)
    
    st.markdown("---")
    
    # Nút xóa mô hình (để huấn luyện lại từ đầu)
    if st.button("🗑️ XÓA MÔ HÌNH (Huấn luyện lại từ đầu)", use_container_width=True, type="secondary"):
        if delete_model():
            st.success("✅ Đã xóa mô hình cũ! Hãy upload dữ liệu mới để huấn luyện.")
            st.rerun()
    
    st.markdown("---")
    
    # Hiển thị thông tin mô hình đã lưu
    if st.session_state.model_trained and st.session_state.training_info:
        st.info("💡 **Mô hình đã sẵn sàng**")
        st.write(f"📅 Huấn luyện: {st.session_state.training_info.get('date', 'N/A')}")
        st.write(f"📊 Số mẫu đạt: {st.session_state.training_info.get('n_valid', 0):,}")
        st.write(f"📈 R²: {st.session_state.training_info.get('r2', 0):.4f}")
        st.write(f"📉 MAE: {st.session_state.training_info.get('mae', 0):.2f} kW")

# ================================
# 11. Thử load mô hình đã lưu
# ================================
if not st.session_state.model_trained:
    load_saved_model()

# ================================
# 12. Upload và huấn luyện (CHỈ KHI CHƯA CÓ MÔ HÌNH)
# ================================
if not st.session_state.model_trained:
    st.header("📁 Bước 1: Tải dữ liệu lịch sử (LẦN ĐẦU TIÊN)")
    st.warning("⚠️ Bạn chỉ cần làm bước này MỘT LẦN DUY NHẤT.")
    
    uploaded_file = st.file_uploader(
        "Chọn file CSV dữ liệu lịch sử",
        type=['csv']
    )
    
    if uploaded_file is not None:
        # Đọc file
        try:
            df = pd.read_csv(uploaded_file, delimiter='\t', encoding='utf-8')
            if len(df.columns) == 1:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        
        # Đặt tên cột
        if len(df.columns) == 4:
            df.columns = ['PCTimeStamp', 'WindSpeed', 'Temp', 'Power']
        else:
            st.error("❌ File không đúng định dạng. Cần 4 cột!")
            st.stop()
        
        # Lưu dữ liệu gốc
        df_original = df.copy()
        
        # Xử lý thời gian
        df['PCTimeStamp'] = pd.to_datetime(df['PCTimeStamp'], errors='coerce')
        df = df.dropna(subset=['WindSpeed', 'Temp', 'Power'])
        
        # Lọc dữ liệu đạt
        valid_power = df['Power'] >= 0
        valid_wind = (df['WindSpeed'] >= 0.1) & (df['WindSpeed'] <= 26)
        valid_temp = (df['Temp'] >= 0) & (df['Temp'] <= 50)
        
        df_valid = df[valid_power & valid_wind & valid_temp]
        
        # ================================
        # HIỂN THỊ THỐNG KÊ CHI TIẾT
        # ================================
        st.subheader("📊 Thống kê dữ liệu đầu vào")
        
        # Hàng 1: Số liệu mẫu
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Tổng số mẫu", f"{len(df):,}")
        with col2:
            st.metric("✅ Mẫu đạt", f"{len(df_valid):,}", delta=f"{(len(df_valid)/len(df)*100):.1f}%")
        with col3:
            st.metric("❌ Mẫu nhiễu", f"{len(df)-len(df_valid):,}")
        with col4:
            st.metric("📈 Tỷ lệ đạt", f"{(len(df_valid)/len(df)*100):.1f}%")
        
        # Hàng 2: Thống kê tốc độ gió
        st.subheader("🌬️ Thống kê tốc độ gió (m/s)")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Trung bình", f"{df_valid['WindSpeed'].mean():.2f}")
        with col2:
            st.metric("Trung vị", f"{df_valid['WindSpeed'].median():.2f}")
        with col3:
            st.metric("Max", f"{df_valid['WindSpeed'].max():.2f}")
        with col4:
            st.metric("Min", f"{df_valid['WindSpeed'].min():.2f}")
        with col5:
            st.metric("Độ lệch chuẩn", f"{df_valid['WindSpeed'].std():.2f}")
        
        # Hàng 3: Thống kê nhiệt độ
        st.subheader("🌡️ Thống kê nhiệt độ (°C)")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Trung bình", f"{df_valid['Temp'].mean():.2f}")
        with col2:
            st.metric("Trung vị", f"{df_valid['Temp'].median():.2f}")
        with col3:
            st.metric("Max", f"{df_valid['Temp'].max():.2f}")
        with col4:
            st.metric("Min", f"{df_valid['Temp'].min():.2f}")
        with col5:
            st.metric("Độ lệch chuẩn", f"{df_valid['Temp'].std():.2f}")
        
        # Hàng 4: Thống kê công suất
        st.subheader("⚡ Thống kê công suất (kW)")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Trung bình", f"{df_valid['Power'].mean():.2f}")
        with col2:
            st.metric("Trung vị", f"{df_valid['Power'].median():.2f}")
        with col3:
            st.metric("Max", f"{df_valid['Power'].max():.2f}")
        with col4:
            st.metric("Min", f"{df_valid['Power'].min():.2f}")
        with col5:
            st.metric("Độ lệch chuẩn", f"{df_valid['Power'].std():.2f}")
        
        # Biểu đồ phân bố
        st.subheader("📊 Biểu đồ phân bố dữ liệu")
        
        fig1 = plot_distributions(df_valid, df_original)
        st.pyplot(fig1)
        
        fig2 = plot_power_distribution(df_valid)
        st.pyplot(fig2)
        
        # Nút huấn luyện
        if st.button("🚀 HUẤN LUYỆN VÀ LƯU MÔ HÌNH", type="primary"):
            with st.spinner("Đang huấn luyện mô hình XGBoost..."):
                # Chuẩn bị dữ liệu
                X = df_valid[['WindSpeed', 'Temp']].values
                y = df_valid['Power'].values
                
                # Chia train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=True
                )
                
                # Chuẩn hóa
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Huấn luyện
                model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0
                )
                model.fit(X_train_scaled, y_train)
                
                # Đánh giá
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Lưu mô hình
                training_info = {
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'n_samples': len(df),
                    'n_valid': len(df_valid),
                    'n_invalid': len(df) - len(df_valid),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'wind_mean': float(df_valid['WindSpeed'].mean()),
                    'temp_mean': float(df_valid['Temp'].mean()),
                    'power_mean': float(df_valid['Power'].mean())
                }
                save_model(model, scaler, training_info)
                
                # Lưu vào session state
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.df_valid = df_valid
                st.session_state.df_original = df_original
                st.session_state.training_info = training_info
                st.session_state.model_trained = True
                
                # Hiển thị kết quả
                st.subheader("🎯 Kết quả huấn luyện")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("MAE", f"{mae:.2f} kW")
                with c2:
                    st.metric("RMSE", f"{rmse:.2f} kW")
                with c3:
                    st.metric("R²", f"{r2:.4f}")
                
                st.success("✅ Huấn luyện thành công! Mô hình đã được lưu.")
                st.info("💡 Từ lần sau, bạn chỉ cần nhập dữ liệu dự báo, không cần upload lại!")
                
                # So sánh dự đoán vs thực tế
                st.subheader("📊 Đánh giá mô hình")
                fig_eval, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test, y_pred, alpha=0.5, c='steelblue', s=20)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel('Thực tế (kW)')
                ax.set_ylabel('Dự báo (kW)')
                ax.set_title('So sánh dự báo vs thực tế')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_eval)
                
                st.rerun()

else:
    # ================================
    # 13. Dự báo (KHI ĐÃ CÓ MÔ HÌNH)
    # ================================
    st.header("🔧 Bước 2: Dự báo công suất")
    
    # Hiển thị thông tin mô hình
    info = st.session_state.training_info
    st.success(f"✅ Mô hình đã sẵn sàng!")
    st.info(f"📅 Huấn luyện: {info['date']}\n\n📊 Số mẫu: {info['n_valid']:,}\n\n📈 R² = {info['r2']:.4f}, MAE = {info['mae']:.2f} kW")
    
    # ================================
    # 13.1 LỰA CHỌN SỐ CHU KỲ
    # ================================
    st.subheader("📅 Chọn số chu kỳ dự báo")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🕐 24 chu kỳ", use_container_width=True):
            st.session_state.n_periods = 24
    with c2:
        if st.button("🕑 48 chu kỳ", use_container_width=True):
            st.session_state.n_periods = 48
    with c3:
        if st.button("🕒 96 chu kỳ", use_container_width=True):
            st.session_state.n_periods = 96
    
    if 'n_periods' not in st.session_state:
        st.session_state.n_periods = 96
    
    st.info(f"📌 Đang chọn: {st.session_state.n_periods} chu kỳ")
    st.markdown("---")
    
    # ================================
    # 13.2 NHẬP DỮ LIỆU
    # ================================
    st.subheader(f"📝 Nhập dữ liệu cho {st.session_state.n_periods} chu kỳ")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌬️ Tốc độ gió (m/s)**")
        wind_text = st.text_area(
            "Nhập tốc độ gió:",
            height=300,
            placeholder=f"Nhập {st.session_state.n_periods} giá trị, mỗi giá trị 1 dòng",
            key="wind_input"
        )
    
    with col2:
        st.markdown("**🌡️ Nhiệt độ (°C)**")
        temp_text = st.text_area(
            "Nhập nhiệt độ:",
            height=300,
            placeholder=f"Nhập {st.session_state.n_periods} giá trị, mỗi giá trị 1 dòng",
            key="temp_input"
        )
    
    # ================================
    # 13.3 XỬ LÝ DỰ BÁO
    # ================================
    if wind_text and temp_text:
        # Parse dữ liệu
        wind_lines = [line.strip() for line in wind_text.strip().split('\n') if line.strip()]
        temp_lines = [line.strip() for line in temp_text.strip().split('\n') if line.strip()]
        
        if len(wind_lines) != st.session_state.n_periods:
            st.error(f"❌ Gió: Cần {st.session_state.n_periods} giá trị, có {len(wind_lines)}")
        elif len(temp_lines) != st.session_state.n_periods:
            st.error(f"❌ Nhiệt: Cần {st.session_state.n_periods} giá trị, có {len(temp_lines)}")
        else:
            try:
                wind_values = [float(x) for x in wind_lines]
                temp_values = [float(x) for x in temp_lines]
                
                st.success(f"✅ Đã nhập {len(wind_values)} giá trị!")
                
                if st.button("🚀 DỰ BÁO NGAY", type="primary", use_container_width=True):
                    with st.spinner("Đang dự báo..."):
                        forecast_input = np.array(list(zip(wind_values, temp_values)))
                        forecast_input_scaled = st.session_state.scaler.transform(forecast_input)
                        forecast_power = st.session_state.model.predict(forecast_input_scaled)
                        forecast_power[forecast_power < 0] = 0
                        forecast_power[np.array(wind_values) > 25] = 0
                        forecast_power[np.array(wind_values) <= 3] = 0
                    
                    st.success("✅ Dự báo hoàn tất!")
                    
                    # Thống kê
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Công suất TB", f"{np.mean(forecast_power):.2f} kW")
                    with c2:
                        st.metric("Công suất max", f"{np.max(forecast_power):.2f} kW")
                    with c3:
                        st.metric("Công suất min", f"{np.min(forecast_power):.2f} kW")
                    with c4:
                        if st.session_state.n_periods == 24:
                            time_step = 1.0
                        elif st.session_state.n_periods == 48:
                            time_step = 0.5
                        else:
                            time_step = 0.25
                        total_energy = np.sum(forecast_power) * time_step
                        st.metric("Tổng sản lượng 24h", f"{total_energy:.2f} kWh")
                    
                    # Vẽ biểu đồ
                    st.subheader("📊 Biểu đồ dự báo")
                    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
                    
                    axes[0].plot(range(1, st.session_state.n_periods+1), forecast_power, 'b-o', markersize=3, linewidth=1)
                    axes[0].set_title("Dự báo công suất")
                    axes[0].set_xlabel("Chu kỳ")
                    axes[0].set_ylabel("Công suất (kW)")
                    axes[0].grid(True, alpha=0.3)
                    axes[0].fill_between(range(1, st.session_state.n_periods+1), forecast_power, alpha=0.2, color='b')
                    
                    axes[1].plot(range(1, st.session_state.n_periods+1), wind_values, 'g-s', markersize=3, linewidth=1)
                    axes[1].set_title("Tốc độ gió đầu vào")
                    axes[1].set_xlabel("Chu kỳ")
                    axes[1].set_ylabel("Gió (m/s)")
                    axes[1].grid(True, alpha=0.3)
                    
                    axes[2].plot(range(1, st.session_state.n_periods+1), temp_values, 'r-^', markersize=3, linewidth=1)
                    axes[2].set_title("Nhiệt độ đầu vào")
                    axes[2].set_xlabel("Chu kỳ")
                    axes[2].set_ylabel("Nhiệt (°C)")
                    axes[2].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Bảng kết quả
                    st.subheader("📋 Bảng kết quả")
                    result_df = pd.DataFrame({
                        'Period': range(1, st.session_state.n_periods+1),
                        'WindSpeed': [f"{x:.2f}" for x in wind_values],
                        'Temperature': [f"{x:.2f}" for x in temp_values],
                        'Power_kW': [f"{x:.2f}" for x in forecast_power]
                    })
                    st.dataframe(result_df, height=400, use_container_width=True)
                    
                    # Tải file
                    csv = result_df.to_csv(index=False)
                    filename = f"forecast_{st.session_state.n_periods}periods.csv"
                    st.download_button("📥 Tải CSV", data=csv, file_name=filename, mime="text/csv")
                    
            except ValueError as e:
                st.error(f"❌ Lỗi: {e}. Vui lòng nhập số hợp lệ.")

# ================================
# 14. Footer
# ================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🌬️ Hệ thống dự báo công suất tuabin gió | XGBoost"
    "</div>",
    unsafe_allow_html=True
)

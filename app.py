# ================================
# 1. Import thư viện
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
# 8. Sidebar
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
    - Dấu phân cách: Tab hoặc dấu phẩy
    
    **📝 Cách nhập dữ liệu:**
    - Mỗi dòng là 1 giá trị số
    - Không để dòng trống
    - Số thực (vd: 2.5, 3.1, 4.2)
    """)
    
    st.markdown("---")
    
    # Hiển thị thông tin mô hình đã lưu
    if st.session_state.model_trained and st.session_state.training_info:
        st.info(f"💡 **Mô hình đã sẵn sàng**\n\n"
                f"📅 Huấn luyện: {st.session_state.training_info['date']}\n\n"
                f"📊 Số mẫu: {st.session_state.training_info['n_samples']}\n\n"
                f"🎯 R² Score: {st.session_state.training_info['r2']:.3f}")
    
    st.markdown("---")
    
    # Nút xóa mô hình (để huấn luyện lại từ đầu)
    if st.session_state.model_trained:
        if st.button("🗑️ Xóa mô hình (Huấn luyện lại)", use_container_width=True):
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
                st.success("✅ Đã xóa mô hình cũ. Hãy upload dữ liệu mới để huấn luyện!")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi khi xóa: {e}")

# ================================
# 9. Thử load mô hình đã lưu
# ================================
if not st.session_state.model_trained:
    load_saved_model()

# ================================
# 10. Upload và huấn luyện (CHỈ KHI CHƯA CÓ MÔ HÌNH)
# ================================
if not st.session_state.model_trained:
    st.header("📁 Bước 1: Tải dữ liệu lịch sử (LẦN ĐẦU TIÊN)")
    st.warning("⚠️ **Lưu ý:** Bạn chỉ cần làm bước này MỘT LẦN DUY NHẤT. Sau khi huấn luyện, mô hình sẽ được lưu lại để dùng cho các lần sau.")
    
    uploaded_file = st.file_uploader(
        "Chọn file CSV dữ liệu lịch sử (có các cột: PCTimeStamp, WindSpeed, Temp, Power)",
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
        
        # Xử lý thời gian
        df['PCTimeStamp'] = pd.to_datetime(df['PCTimeStamp'], errors='coerce')
        df = df.dropna(subset=['WindSpeed', 'Temp', 'Power'])
        
        # Lọc dữ liệu đạt
        valid_power = df['Power'] >= 0
        valid_wind = (df['WindSpeed'] >= 0.1) & (df['WindSpeed'] <= 26)
        valid_temp = (df['Temp'] >= 0) & (df['Temp'] <= 50)
        
        df_valid = df[valid_power & valid_wind & valid_temp]
        
        # Hiển thị thông tin
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Tổng số mẫu", len(df))
        with col2:
            st.metric("✅ Mẫu đạt", len(df_valid), delta=f"{(len(df_valid)/len(df)*100):.1f}%")
        with col3:
            st.metric("❌ Mẫu nhiễu", len(df)-len(df_valid))
        
        # Nút huấn luyện
        if st.button("🚀 Huấn luyện và LƯU mô hình", type="primary"):
            with st.spinner("Đang huấn luyện mô hình XGBoost với 300 cây..."):
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
                r2 = model.score(X_test_scaled, y_test)
                
                # Lưu mô hình
                training_info = {
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'n_samples': len(df_valid),
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
                save_model(model, scaler, training_info)
                
                # Lưu vào session state
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.df_valid = df_valid
                st.session_state.training_info = training_info
                st.session_state.model_trained = True
                
                # Hiển thị kết quả
                st.success("✅ Huấn luyện và LƯU mô hình thành công!")
                st.info("💡 **Từ lần sau, bạn chỉ cần nhập dữ liệu dự báo, không cần upload file lịch sử nữa!**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE", f"{mae:.2f} kW")
                with col2:
                    st.metric("RMSE", f"{rmse:.2f} kW")
                with col3:
                    st.metric("R² Score", f"{r2:.3f}")
                
                # Vẽ biểu đồ phân bố
                st.subheader("📊 Phân tích dữ liệu")
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Công suất theo gió
                wind_bins = np.arange(0, 26, 2)
                wind_labels = [f"{int(wind_bins[i])}-{int(wind_bins[i+1])}" for i in range(len(wind_bins)-1)]
                df_valid_copy = df_valid.copy()
                df_valid_copy['WindBin'] = pd.cut(df_valid_copy['WindSpeed'], bins=wind_bins, labels=wind_labels)
                avg_power = df_valid_copy.groupby('WindBin', observed=True)['Power'].mean()
                
                axes[0].bar(range(len(avg_power)), avg_power.values, color='steelblue', alpha=0.7)
                axes[0].set_xlabel("Tốc độ gió (m/s)")
                axes[0].set_ylabel("Công suất trung bình (kW)")
                axes[0].set_title("Công suất theo tốc độ gió")
                axes[0].set_xticks(range(len(avg_power)))
                axes[0].set_xticklabels(avg_power.index, rotation=45)
                axes[0].grid(True, alpha=0.3)
                
                # Số lượng mẫu
                count_by_wind = df_valid_copy.groupby('WindBin', observed=True)['WindSpeed'].count()
                axes[1].bar(range(len(count_by_wind)), count_by_wind.values, color='coral', alpha=0.7)
                axes[1].set_xlabel("Tốc độ gió (m/s)")
                axes[1].set_ylabel("Số lượng mẫu")
                axes[1].set_title("Phân bố dữ liệu")
                axes[1].set_xticks(range(len(count_by_wind)))
                axes[1].set_xticklabels(count_by_wind.index, rotation=45)
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.rerun()

else:
    # ================================
    # 11. Dự báo (KHI ĐÃ CÓ MÔ HÌNH)
    # ================================
    st.header("🔧 Dự báo công suất")
    st.success(f"✅ Mô hình đã sẵn sàng! Huấn luyện ngày: {st.session_state.training_info['date']}")
    
    # ================================
    # 11.1 LỰA CHỌN SỐ CHU KỲ DỰ BÁO
    # ================================
    st.subheader("📅 Chọn số chu kỳ dự báo")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        period_24 = st.button("🕐 24 chu kỳ (24 giờ)", use_container_width=True)
    with col2:
        period_48 = st.button("🕑 48 chu kỳ (24 giờ)", use_container_width=True)
    with col3:
        period_96 = st.button("🕒 96 chu kỳ (24 giờ)", use_container_width=True)
    
    # Khởi tạo số chu kỳ mặc định
    if 'n_periods' not in st.session_state:
        st.session_state.n_periods = 96
    
    if period_24:
        st.session_state.n_periods = 24
        st.success("✅ Đã chọn dự báo 24 chu kỳ (24 giờ)")
    elif period_48:
        st.session_state.n_periods = 48
        st.success("✅ Đã chọn dự báo 48 chu kỳ (24 giờ)")
    elif period_96:
        st.session_state.n_periods = 96
        st.success("✅ Đã chọn dự báo 96 chu kỳ (24 giờ)")
    
    st.info(f"📌 **Hiện tại đang chọn: {st.session_state.n_periods} chu kỳ**")
    
    st.markdown("---")
    
    # ================================
    # 11.2 NHẬP DỮ LIỆU DỰ BÁO
    # ================================
    st.subheader(f"📝 Nhập dữ liệu dự báo cho {st.session_state.n_periods} chu kỳ")
    st.markdown("**✏️ Hướng dẫn:** Mỗi dòng là 1 giá trị (dễ dàng copy/paste từ Excel)")
    st.warning("⚠️ **Lưu ý:** Không để dòng trống, mỗi dòng chỉ chứa 1 số")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌬️ Tốc độ gió (m/s)**")
        st.caption(f"Nhập đúng {st.session_state.n_periods} giá trị, mỗi giá trị 1 dòng")
        
        wind_text = st.text_area(
            "Nhập tốc độ gió:",
            height=300,
            placeholder=f"Ví dụ cho {st.session_state.n_periods} chu kỳ:\n2.5\n3.1\n4.2\n2.8\n3.5\n...",
            key="wind_input"
        )
        
        wind_file = st.file_uploader(
            "Hoặc upload file TXT (mỗi số 1 dòng)", 
            type=['txt'], 
            key="wind_file"
        )
        if wind_file:
            content = wind_file.read().decode('utf-8')
            wind_text = content
            st.success(f"✅ Đã đọc file")
    
    with col2:
        st.markdown("**🌡️ Nhiệt độ (°C)**")
        st.caption(f"Nhập đúng {st.session_state.n_periods} giá trị, mỗi giá trị 1 dòng")
        
        temp_text = st.text_area(
            "Nhập nhiệt độ:",
            height=300,
            placeholder=f"Ví dụ cho {st.session_state.n_periods} chu kỳ:\n20.5\n21.0\n22.3\n21.8\n20.9\n...",
            key="temp_input"
        )
        
        temp_file = st.file_uploader(
            "Hoặc upload file TXT (mỗi số 1 dòng)", 
            type=['txt'], 
            key="temp_file"
        )
        if temp_file:
            content = temp_file.read().decode('utf-8')
            temp_text = content
            st.success(f"✅ Đã đọc file")
    
    # ================================
    # 11.3 HÀM XỬ LÝ DỮ LIỆU NHẬP
    # ================================
    def parse_input_data(text, expected_count, data_name):
        if not text or not text.strip():
            return None, f"❌ Chưa nhập dữ liệu {data_name}"
        
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        
        if len(lines) == 0:
            return None, f"❌ Không có dữ liệu {data_name}"
        
        if len(lines) != expected_count:
            return None, f"❌ {data_name}: Cần {expected_count} giá trị, bạn đã nhập {len(lines)} giá trị"
        
        values = []
        invalid_lines = []
        
        for i, line in enumerate(lines, 1):
            try:
                val = float(line)
                values.append(val)
            except ValueError:
                invalid_lines.append(i)
        
        if invalid_lines:
            return None, f"❌ {data_name}: Dòng {', '.join(map(str, invalid_lines))} không phải số hợp lệ"
        
        return values, None
    
    # ================================
    # 11.4 XỬ LÝ VÀ DỰ BÁO
    # ================================
    if wind_text and temp_text:
        wind_values, wind_error = parse_input_data(wind_text, st.session_state.n_periods, "Tốc độ gió")
        temp_values, temp_error = parse_input_data(temp_text, st.session_state.n_periods, "Nhiệt độ")
        
        if wind_error:
            st.error(wind_error)
        elif temp_error:
            st.error(temp_error)
        else:
            st.success(f"✅ Đã nhập đúng {st.session_state.n_periods} giá trị cho cả hai cột!")
            
            with st.expander("📋 Xem trước dữ liệu đã nhập (10 giá trị đầu)"):
                preview_data = []
                for i in range(min(10, st.session_state.n_periods)):
                    preview_data.append({
                        'STT': i + 1,
                        'Tốc độ gió (m/s)': f"{wind_values[i]:.2f}",
                        'Nhiệt độ (°C)': f"{temp_values[i]:.2f}"
                    })
                preview_df = pd.DataFrame(preview_data)
                st.dataframe(preview_df, use_container_width=True)
                
                col1_stats, col2_stats = st.columns(2)
                with col1_stats:
                    st.write(f"**🌬️ Gió:** Min={min(wind_values):.2f}, Max={max(wind_values):.2f}, TB={np.mean(wind_values):.2f}")
                with col2_stats:
                    st.write(f"**🌡️ Nhiệt:** Min={min(temp_values):.2f}, Max={max(temp_values):.2f}, TB={np.mean(temp_values):.2f}")
            
            if st.button("🚀 DỰ BÁO NGAY", type="primary", use_container_width=True):
                with st.spinner(f"Đang dự báo {st.session_state.n_periods} chu kỳ..."):
                    forecast_input = np.array(list(zip(wind_values, temp_values)))
                    forecast_input_scaled = st.session_state.scaler.transform(forecast_input)
                    forecast_power = st.session_state.model.predict(forecast_input_scaled)
                    
                    forecast_power[forecast_power < 0] = 0
                    forecast_power[np.array(wind_values) > 25] = 0
                    forecast_power[np.array(wind_values) <= 3] = 0
                
                st.success("✅ Dự báo hoàn tất!")
                
                # Thống kê nhanh
                col1_metric, col2_metric, col3_metric, col4_metric = st.columns(4)
                with col1_metric:
                    st.metric("📊 Công suất TB", f"{np.mean(forecast_power):.2f} kW")
                with col2_metric:
                    st.metric("📈 Công suất max", f"{np.max(forecast_power):.2f} kW")
                with col3_metric:
                    st.metric("📉 Công suất min", f"{np.min(forecast_power):.2f} kW")
                
                # Tính tổng sản lượng
                if st.session_state.n_periods == 24:
                    time_step = 1.0
                elif st.session_state.n_periods == 48:
                    time_step = 0.5
                else:
                    time_step = 0.25
                
                total_energy = np.sum(forecast_power) * time_step
                
                with col4_metric:
                    st.metric("Tổng sản lượng 24h", f"{total_energy:.2f} kWh")
                
                # Vẽ biểu đồ
                st.subheader("📊 Biểu đồ dự báo")
                
                fig, axes = plt.subplots(3, 1, figsize=(14, 10))
                
                axes[0].plot(range(1, st.session_state.n_periods+1), forecast_power, 
                            marker='o', linestyle='-', color='b', markersize=3, linewidth=1)
                axes[0].set_title(f"Dự báo công suất {st.session_state.n_periods} chu kỳ", 
                                fontsize=12, fontweight='bold')
                axes[0].set_xlabel("Chu kỳ")
                axes[0].set_ylabel("Công suất (kW)")
                axes[0].grid(True, alpha=0.3)
                axes[0].fill_between(range(1, st.session_state.n_periods+1), forecast_power, alpha=0.2, color='b')
                
                axes[1].plot(range(1, st.session_state.n_periods+1), wind_values, 
                            marker='s', linestyle='-', color='g', markersize=3, linewidth=1)
                axes[1].set_title("Tốc độ gió đầu vào", fontsize=12, fontweight='bold')
                axes[1].set_xlabel("Chu kỳ")
                axes[1].set_ylabel("Tốc độ gió (m/s)")
                axes[1].grid(True, alpha=0.3)
                axes[1].fill_between(range(1, st.session_state.n_periods+1), wind_values, alpha=0.2, color='g')
                
                axes[2].plot(range(1, st.session_state.n_periods+1), temp_values, 
                            marker='^', linestyle='-', color='r', markersize=3, linewidth=1)
                axes[2].set_title("Nhiệt độ đầu vào", fontsize=12, fontweight='bold')
                axes[2].set_xlabel("Chu kỳ")
                axes[2].set_ylabel("Nhiệt độ (°C)")
                axes[2].grid(True, alpha=0.3)
                axes[2].fill_between(range(1, st.session_state.n_periods+1), temp_values, alpha=0.2, color='r')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Bảng kết quả
                st.subheader("📋 Bảng kết quả chi tiết")
                
                result_data = []
                for i in range(st.session_state.n_periods):
                    result_data.append({
                        'Period': i + 1,
                        'WindSpeed': f"{wind_values[i]:.2f}",
                        'Temperature': f"{temp_values[i]:.2f}",
                        'Power': f"{forecast_power[i]:.2f}"
                    })
                result_df = pd.DataFrame(result_data)
                
                st.dataframe(result_df, height=400, use_container_width=True)
                
                csv = result_df.to_csv(index=False)
                
                if st.session_state.n_periods == 24:
                    filename = f"forecast_{st.session_state.n_periods}_periods.csv"
                elif st.session_state.n_periods == 48:
                    filename = f"forecast_{st.session_state.n_periods}_periods.csv"
                else:
                    filename = f"forecast_{st.session_state.n_periods}_periods.csv"
                
                st.download_button(
                    label="📥 Tải xuống file CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Thống kê theo giờ cho 96 chu kỳ
                if st.session_state.n_periods == 96:
                    st.subheader("📊 Thống kê công suất phát theo giờ")
                    
                    hourly_power = []
                    for i in range(0, 96, 4):
                        hourly_power.append(np.sum(forecast_power[i:i+4]) * 0.25)
                    
                    hours = range(1, 25)
                    fig_hourly, ax = plt.subplots(figsize=(12, 5))
                    ax.bar(hours, hourly_power, color='skyblue', alpha=0.7, edgecolor='navy')
                    ax.set_xlabel("Giờ trong ngày")
                    ax.set_ylabel("Công suất phát (kWh)")
                    ax.set_title("Công suất phát theo giờ")
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    for i, (h, val) in enumerate(zip(hours, hourly_power)):
                        if val > 10:
                            ax.text(h, val + 5, f'{val:.0f}', ha='center', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    st.pyplot(fig_hourly)
                    
                    hourly_data = []
                    cumulative = 0
                    for i in range(24):
                        cumulative += hourly_power[i]
                        hourly_data.append({
                            'Giờ': f"{i+1:02d}:00",
                            'Sản lượng (kWh)': f"{hourly_power[i]:.2f}",
                            'Lũy kế (kWh)': f"{cumulative:.2f}"
                        })
                    hourly_df = pd.DataFrame(hourly_data)
                    st.dataframe(hourly_df, use_container_width=True)
                    
                    max_hour = np.argmax(hourly_power) + 1
                    st.info(f"💡 **Giờ phát điện cao điểm:** {max_hour}:00 với sản lượng {hourly_power[max_hour-1]:.2f} kWh")
    
    else:
        st.info(f"📝 Vui lòng nhập {st.session_state.n_periods} giá trị tốc độ gió và {st.session_state.n_periods} giá trị nhiệt độ (mỗi dòng 1 số)")

# ================================
# 12. Footer
# ================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🌬️ Hệ thống dự báo công suất tuabin gió | XGBoost | Hỗ trợ 24/48/96 chu kỳ"
    "</div>",
    unsafe_allow_html=True
)

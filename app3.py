import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --- 1. Load mô hình ---
try:
    model = joblib.load("model_knn.pkl")  # Đảm bảo đúng tên file mô hình
except:
    st.error("Không tìm thấy file model_knn.pkl. Hãy chắc rằng nó nằm trong cùng thư mục với app.py")
    st.stop()

try:
    # Tải các ranh giới bin đã lưu từ file huấn luyện
    bin_edges_dict = joblib.load("bin_edges.pkl")
except FileNotFoundError:
    st.error("Không tìm thấy file bin_edges.pkl. Không thể thực hiện binning cho các cột số. Hãy chắc rằng nó nằm trong cùng thư mục với app.py")
    st.stop()
except Exception as e:
    st.error(f"Đã xảy ra lỗi khi tải ranh giới bin: {e}")
    st.stop()

# --- 2. Cài đặt tiêu đề ---
st.set_page_config(
    page_title="Dự đoán mức độ hài lòng hành khách VietJet",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <style>
    /* Toàn bộ nền trang */
    .main {
        background-color: #FFFFFF;
    }

    /* Tiêu đề chính */
    .main-header {
        font-size: 3.5em;
        font-weight: bold;
        color: #ED1B24; /* Đỏ VietJet */
        text-align: center;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    /* Tiêu đề phụ */
    .subheader {
        font-size: 1.8em;
        font-weight: bold;
        color: #FFD200; /* Vàng VietJet */
        margin-top: 1.5em;
        margin-bottom: 1em;
        border-bottom: 2px solid #F2F2F2;
        padding-bottom: 0.5em;
    }

    /* Thanh trượt (slider) */
    .stSlider > div > div > div[data-baseweb="slider"] {
        background: #ED1B24;
    }

    /* Nút bấm */
    .stButton > button {
        background-color: #ED1B24;  /* Đỏ VietJet */
        color: white;
        font-size: 1.2em;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.8em 2em;
        margin-top: 1em;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }

    .stButton > button:hover {
        background-color: #FFD200;  /* Hover: chuyển sang vàng VietJet */
        color: black;
        box-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }

    /* Nhãn Radio và Input */
    .stRadio > label, .stNumberInput > label {
        font-weight: bold;
        color: #333333;
    }

    /* Thông báo hoặc Alert box */
    .stAlert {
        border-radius: 10px;
        padding: 1em;
        margin-top: 1em;
        background-color: #FFF3CD;
        border: 1px solid #FFD200;
        color: #856404;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🎯 Dự đoán mức độ hài lòng hành khách khi sử dụng dịch vụ bay của VietJet")
st.write("Ứng dụng này dùng Machine Learning để dự đoán mức độ hài lòng của hành khách dựa trên thông tin chuyến bay.")

st.write("---")

# --- 3. Nhập liệu ---
st.subheader("🔍 Nhập thông tin hành khách:")

# Các thuộc tính đầu vào
gender = st.radio("Giới tính", ["Nam", "Nữ"])
customer_type = st.radio("Loại hành khách", ["Khách hàng vãng lai", "Khách hàng trung thành"])
type_of_travel = st.radio("Loại chuyến bay", ["Công tác", "Cá nhân"])
travel_class = st.radio("Hạng ghế", ["Business", "Eco plus", "Eco"])
age = st.slider("Tuổi", 7, 85, 40)
flight_distance = st.number_input("Khoảng cách chuyến bay (km)", 31, 4983, 1189)

# Các dịch vụ đánh giá theo thang điểm 0–5
wifi = st.slider("Dịch vụ wifi trên máy bay", 0, 5, 3)
dep_arr_time = st.slider("Thời gian cất/hạ cánh thuận tiện", 0, 5, 3)
booking = st.slider("Dễ dàng đặt vé online", 0, 5, 3)
gate = st.slider("Vị trí cổng ra máy bay", 0, 5, 3)
food = st.slider("Đồ ăn và thức uống", 0, 5, 3)
boarding = st.slider("Thời gian lên máy bay", 0, 5, 3)
seat = st.slider("Sự thoải mái của ghế", 0, 5, 3)
entertain = st.slider("Giải trí trên máy bay", 0, 5, 3)
onboard_service = st.slider("Dịch vụ trên máy bay", 0, 5, 3)
legroom = st.slider("Không gian để chân", 0, 5, 3)
baggage = st.slider("Xử lý hành lý", 0, 5, 3)
checkin = st.slider("Dịch vụ checkin", 0, 5, 3)
inflight_service = st.slider("Dịch vụ khi đang bay", 0, 5, 3)
clean = st.slider("Độ sạch sẽ", 0, 5, 3)

#⏰ Thời gian trễ chuyến
dep_delay = st.number_input("Thời gian trễ khởi hành (phút)", 0, 1592, 0)
arr_delay = st.number_input("Thời gian trễ đến (phút)", 0, 1584, 0)

# --- 4. Mã hóa dữ liệu ---
# Các biến đã được mã hóa: Gender (Nam=1), Customer Type (Khách hàng trung thành=0), Type of Travel (Công tác=0), Class (Business=0, Eco plus=2, Eco=1)
gender_encoded = 1 if gender == "Nam" else 0
customer_encoded = 0 if customer_type == "Khách hàng trung thành" else 1
travel_encoded = 1 if type_of_travel == "Cá nhân" else 0
class_encoded = {"Business": 0, "Eco plus": 2, "Eco": 1}[travel_class]

# --- 5. Dự đoán ---
if st.button("📊 Dự đoán mức độ hài lòng"):
    input_data = pd.DataFrame([[
        gender_encoded, customer_encoded, age, travel_encoded, class_encoded, flight_distance,
        wifi, dep_arr_time, booking, gate, food, boarding, seat, entertain,
        onboard_service, legroom, baggage, checkin, inflight_service, clean,
        dep_delay, arr_delay
    ]], columns=[
        'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
        'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service',
        'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
        'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][int(prediction)]

    if prediction == 1:
        st.success(f"Hành khách hài lòng với dịch vụ bay của VietJet (Xác suất: {proba:.2%})")
        st.markdown(
            """
            <div style="background-color:#e6ffe6; padding:15px; border-radius:10px; border-left: 5px solid #4CAF50; margin-top:1em;">
                <h4 style="color:#28A745;">Cảm ơn quý khách đã tin tưởng và lựa chọn VietJet!</h4>
                <p>Sự hài lòng của quý khách là động lực để chúng tôi không ngừng cải thiện và mang đến những trải nghiệm bay tốt nhất.</p>
                <p>Mong rằng quý khách sẽ tiếp tục đồng hành cùng VietJet trong những chuyến đi sắp tới!</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error(f"Hành khách có mức độ hài lòng trung lập hoặc không hài lòng với dịch vụ bay của VietJet (Xác suất: {proba:.2%})")
        st.markdown(
            """
            <div style="background-color:#ffe6e6; padding:15px; border-radius:10px; border-left: 5px solid #DC3545; margin-top:1em;">
                <h4 style="color:#DC3545;">VietJet chân thành xin lỗi quý khách!</h4>
                <p>Chúng tôi rất tiếc khi trải nghiệm của quý khách chưa đạt được sự mong đợi. Phản hồi của quý khách vô cùng quan trọng để chúng tôi cải thiện dịch vụ.</p>
                <p>Thông tin này đã được gửi đến Đội Quản lý Chất lượng Dịch vụ của VietJet để xem xét và đưa ra các biện pháp khắc phục kịp thời. Chúng tôi cam kết sẽ nỗ lực hơn nữa để mang đến dịch vụ tốt nhất cho quý khách.</p>
                <p>Rất mong quý khách sẽ cho VietJet cơ hội phục vụ tốt hơn trong tương lai!</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- 6. Thông tin thêm ---
# --- 6. Thông tin thêm (Footer) ---
st.write("---")
st.markdown(
    """
    <div style="text-align: center; margin-top: 2em; color: #777;">
        <p>Ứng dụng được phát triển bởi Nhóm 16 - 2025</p>
        <p>Sử dụng dữ liệu hành khách hàng không để huấn luyện mô hình Machine Learning.</p>
    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os

# تعيين إعدادات الصفحة (تم تغيير الأيقونة إلى أيقونة نار)
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🔥",
    layout="wide"
)

# --- Custom CSS for Modern/Fiery Look ---
st.markdown("""
<style>
/* Custom font and general styling */
.stApp {
    background-color: #0E1117; /* خلفية داكنة جداً */
    color: #FAFAFA;
}

/* Fiery Glow for Title */
h1 {
    text-shadow: 0 0 5px #FF4B4B, 0 0 10px #FF4B4B, 0 0 15px #FF4B4B;
    color: #FF4B4B !important;
    font-size: 3em;
    text-align: center;
    padding-bottom: 20px;
}

/* Custom success/error boxes for prediction */
.stSuccess > div {
    border-left: 6px solid #FF4B4B; /* حدود نارية للنجاح */
    background-color: #1A1A1A;
    color: #FAFAFA;
    box-shadow: 0 0 10px rgba(255, 75, 75, 0.5); /* ظل ناري خفيف */
}

.stError > div {
    border-left: 6px solid #4B4BFF; /* حدود زرقاء داكنة للفشل (برودة الموت) */
    background-color: #1A1A1A;
    color: #FAFAFA;
    box-shadow: 0 0 10px rgba(75, 75, 255, 0.5); /* ظل بارد خفيف */
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #1F2430;
}

/* Input elements styling */
.stSlider > div > div > div:nth-child(2) {
    background-color: #FF4B4B; /* لون شريط التمرير */
}

</style>
""", unsafe_allow_html=True)

# تحميل النموذج
try:
    # تحديد المسار الصحيح للملف
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ لم يتم العثور على ملف النموذج. تأكد من وجود best_model.pkl في مجلد models.")
    st.stop()

# --- الواجهة الرئيسية ---
st.title("🔥 تايتانيك: محاكي البقاء 🔥")

# محاذاة الصورة في المنتصف
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        img_path = os.path.join(os.path.dirname(__file__), 'assets', 'ship.png')
        img = Image.open(img_path)
        st.image(img, use_column_width=True)
    except FileNotFoundError:
        st.warning("⚠️ لم يتم العثور على صورة السفينة.")

st.markdown("---")

# --- شريط الإدخال الجانبي ---
st.sidebar.header("بيانات الراكب")

# حقول الإدخال
pclass = st.sidebar.selectbox("درجة التذكرة (PClass)", [1, 2, 3], help="1st = عليا، 2nd = متوسطة، 3rd = دنيا")
sex = st.sidebar.selectbox("الجنس", ["ذكر", "أنثى"])
age = st.sidebar.slider("العمر", 1, 80, 25)
fare = st.sidebar.slider("سعر التذكرة", 0.0, 500.0, 50.0)
sibsp = st.sidebar.slider("عدد الإخوة/الزوجات على متن السفينة", 0, 8, 0)
parch = st.sidebar.slider("عدد الآباء/الأطفال على متن السفينة", 0, 6, 0)

# تحويل الجنس إلى قيمة رقمية
sex_encoded = 1 if sex == "ذكر" else 0

# تجهيز بيانات الإدخال
input_data = np.array([[pclass, sex_encoded, age, fare, sibsp, parch]])

# --- منطقة التنبؤ ---
st.header("نتيجة التنبؤ")

if st.button("توقع البقاء", help="اضغط لمعرفة ما إذا كان الراكب سينجو أم لا"):
    
    # إجراء التنبؤ
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][prediction]

    # عرض النتيجة
    if prediction == 1:
        st.success(f"🎉 نجاة مؤكدة! 🔥")
        st.markdown(f"**احتمالية النجاة:** **{prob*100:.2f}%**")
        st.balloons()
    else:
        st.error(f"💔 للأسف، لم ينجُ الراكب. 🧊")
        st.markdown(f"**احتمالية عدم النجاة:** **{prob*100:.2f}%**")

st.markdown("---")
st.info("تم تطوير هذا المحاكي باستخدام Streamlit ونموذج تعلم آلي للتنبؤ بالبقاء على متن سفينة تايتانيك.")

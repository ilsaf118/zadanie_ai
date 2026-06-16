import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Настройка страницы
st.set_page_config(page_title="Прогноз энергопотребления", layout="wide")
st.title("🏠 Прогноз энергопотребления бытовых приборов")
st.markdown("Введите параметры окружающей среды и получите прогноз потребления электроэнергии (Вт·ч) за следующие 10 минут.")

# Загрузка модели и списка признаков
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/best_energy_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, feature_names

model, feature_names = load_artifacts()

# Создаём форму ввода
with st.form("input_form"):
    st.subheader("📊 Входные данные")
    
    # Разбиваем на три колонки для удобства
    col1, col2, col3 = st.columns(3)
    
    # Словарь для хранения значений
    input_values = {}
    
    with col1:
        st.markdown("**💡 Освещение**")
        lights = st.number_input("Энергопотребление освещения (lights), Вт·ч", value=10, min_value=0)
        input_values['lights'] = lights
        
        st.markdown("**🏠 Температура в комнатах (°C)**")
        input_values['T1'] = st.number_input("T1 (кухня)", value=20.0)
        input_values['T2'] = st.number_input("T2 (гостиная)", value=21.0)
        input_values['T3'] = st.number_input("T3 (прачечная)", value=19.0)
        input_values['T4'] = st.number_input("T4 (офис)", value=20.0)
        input_values['T5'] = st.number_input("T5 (ванная)", value=21.0)
        input_values['T6'] = st.number_input("T6 (южная комната)", value=20.0)
        input_values['T7'] = st.number_input("T7 (детская)", value=20.0)
        input_values['T8'] = st.number_input("T8 (коридор)", value=19.0)
        input_values['T9'] = st.number_input("T9 (подвал)", value=18.0)
    
    with col2:
        st.markdown("**💧 Влажность в комнатах (%)**")
        input_values['RH_1'] = st.number_input("RH_1 (кухня)", value=50.0)
        input_values['RH_2'] = st.number_input("RH_2 (гостиная)", value=50.0)
        input_values['RH_3'] = st.number_input("RH_3 (прачечная)", value=50.0)
        input_values['RH_4'] = st.number_input("RH_4 (офис)", value=50.0)
        input_values['RH_5'] = st.number_input("RH_5 (ванная)", value=55.0)
        input_values['RH_6'] = st.number_input("RH_6 (южная комната)", value=50.0)
        input_values['RH_7'] = st.number_input("RH_7 (детская)", value=50.0)
        input_values['RH_8'] = st.number_input("RH_8 (коридор)", value=50.0)
        input_values['RH_9'] = st.number_input("RH_9 (подвал)", value=60.0)
    
    with col3:
        st.markdown("**🌤️ Уличные условия**")
        input_values['T_out'] = st.number_input("Температура на улице (T_out), °C", value=15.0)
        input_values['RH_out'] = st.number_input("Влажность на улице (RH_out), %", value=70.0)
        input_values['Press_mm_hg'] = st.number_input("Атмосферное давление (Press_mm_hg), мм рт. ст.", value=760.0)
        input_values['Windspeed'] = st.number_input("Скорость ветра (Windspeed), м/с", value=2.0)
        input_values['Visibility'] = st.number_input("Видимость (Visibility), км", value=10.0)
        input_values['Tdewpoint'] = st.number_input("Точка росы (Tdewpoint), °C", value=10.0)
    
    submitted = st.form_submit_button("🔮 Предсказать энергопотребление")

if submitted:
    # Собираем вектор в том порядке, в котором модель обучена
    try:
        input_vector = np.array([input_values[f] for f in feature_names]).reshape(1, -1)
        prediction = model.predict(input_vector)[0]
        st.success(f"### ⚡ Прогнозируемое потребление приборов: **{prediction:.1f} Вт·ч**")
        st.info("Это суммарное энергопотребление (стиральная машина, посудомоечная, сушилка, холодильник и т.д.) за следующие 10 минут.")
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")
        st.write("Ожидаемые признаки:", feature_names)
        st.write("Переданные значения:", list(input_values.keys()))
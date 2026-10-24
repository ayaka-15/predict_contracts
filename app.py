import streamlit as st
import joblib
import pandas as pd

# モデルの読み込み
model = joblib.load("models/logisticRegression.pkl")

st.title("銀行商品加入予測アプリ")
st.write("顧客情報を入力して、判定ボタンを押してください。")

# ユーザー入力

# 年齢
age = st.number_input(
    "年齢", min_value=18, max_value=100, value=35, step=1
)

# 職業
job = st.selectbox(
    "職業",
    [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ],
    index=9
)

# 婚姻
marital = st.selectbox(
    "婚姻",
    ["divorced", "married", "single"],
    index=2
)

# 学歴
education = st.selectbox(
    "学歴",
    ["primary", "secondary", "tertiary", "unknown"],
    index=3
)

# 住宅ローン
housing = st.selectbox(
    "住宅ローン", ["yes", "no"], index=0
)

# 個人ローン
loan = st.selectbox(
    "個人ローン", ["yes", "no"], index=1
)

# 過去のキャンペーン結果
poutcome = st.selectbox(
    "過去のキャンペーン結果",
    ["failure", "other", "success", "unknown"],
    index=0
)

# 判定
# 判定
if st.button("判定する"):
    input_data = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "housing": housing,
        "loan": loan,
        "poutcome": poutcome
    }

    # 1行のDataFrameに変換
    X = pd.DataFrame([input_data])

    # 推論と結果表示
    try:
        pred = model.predict(X)[0]

        st.subheader("予測結果")

        if pred == "yes":
            st.success("顧客が定期預金を契約するか：契約する")
        else:
            st.info("顧客が定期預金を契約するか：契約しない")

    except Exception as e:
        st.error(f"予測中にエラーが発生しました：{e}")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")

# ---------------- SESSION INIT ----------------
if "page" not in st.session_state:
    st.session_state.page = "upload"

# ---------------- PAGE 1: UPLOAD ----------------
if st.session_state.page == "upload":

    st.markdown(
        """
        <h1 style='font-size:48px; font-weight:700;'>
        🛍️ Advanced<br>Customer<br>Segmentation
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Upload your dataset (CSV)")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    st.info("👋 Please upload a CSV file to begin analysis.")

    if file is not None:
        try:
            df = pd.read_csv(file)

            if df.empty:
                st.error("Uploaded file is empty!")
            else:
                st.session_state.df = df
                st.session_state.page = "dashboard"
                st.success("✅ File uploaded successfully!")
                st.rerun()

        except Exception as e:
            st.error(f"Error reading file: {e}")

# ---------------- PAGE 2: DASHBOARD ----------------
elif st.session_state.page == "dashboard":

    df = st.session_state.df

    st.title("📊 Customer Segmentation Dashboard")

    # Select numeric columns automatically
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("❌ Need at least 2 numeric columns for analysis")
    else:
        # Feature selection (more flexible)
        selected_cols = st.multiselect(
            "Select 2 features for analysis",
            numeric_cols,
            default=numeric_cols[:2]
        )

        if len(selected_cols) == 2:

            step = st.radio(
                "Select Analysis Step:",
                [
                    "1. Dataset Overview",
                    "2. Visual Analysis (EDA)",
                    "3. K-Means Clustering",
                    "4. Marketing Insights"
                ]
            )

            st.markdown("---")

            X = df[selected_cols]

            # -------- STEP 1 --------
            if step == "1. Dataset Overview":
                st.header("📊 Data Summary")

                col1, col2 = st.columns(2)

                col1.metric("Total Records", len(df))
                col2.metric("Columns", len(df.columns))

                st.write("Selected Features:", selected_cols)
                st.dataframe(df.head())

            # -------- STEP 2 --------
            elif step == "2. Visual Analysis (EDA)":
                st.header("📈 Scatter Plot")

                fig, ax = plt.subplots()
                ax.scatter(X.iloc[:, 0], X.iloc[:, 1])
                ax.set_xlabel(selected_cols[0])
                ax.set_ylabel(selected_cols[1])
                st.pyplot(fig)

            # -------- STEP 3 --------
            elif step == "3. K-Means Clustering":
                st.header("🤖 Clustering")

                k = st.slider("Select number of clusters", 2, 10, 3)

                kmeans = KMeans(n_clusters=k, random_state=42)
                df['Cluster'] = kmeans.fit_predict(X)

                st.session_state.df = df

                fig, ax = plt.subplots()
                ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['Cluster'])
                ax.set_xlabel(selected_cols[0])
                ax.set_ylabel(selected_cols[1])
                st.pyplot(fig)

            # -------- STEP 4 --------
            elif step == "4. Marketing Insights":
                st.header("🧠 Insights")

                if 'Cluster' in df.columns:
                    st.dataframe(df.groupby('Cluster')[selected_cols].mean())
                else:
                    st.warning("Run clustering first!")

        else:
            st.warning("⚠️ Please select exactly 2 features")

    # Reset button
    if st.button("🔄 Upload New File"):
        st.session_state.clear()
        st.rerun()
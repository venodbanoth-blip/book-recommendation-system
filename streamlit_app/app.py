
import streamlit as st
import requests

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://book-recommendation-system-2c8p.onrender.com"

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}

.book-card {
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-bottom: 20px;
}

.book-title {
    font-size: 20px;
    font-weight: 600;
}

.book-author {
    color: #666;
}

.score {
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 Book Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Content-Based Machine Learning + FastAPI + GenAI'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Recommendation Settings")

user_id = st.sidebar.number_input(
    "User ID",
    min_value=1,
    value=8,
    step=1
)

n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

st.sidebar.markdown("---")

st.sidebar.info(
    "The recommendation engine is powered by "
    "your Content-Based ML model running through FastAPI."
)


# ============================================================
# HEALTH CHECK
# ============================================================

try:

    health = requests.get(
        f"{API_URL}/health",
        timeout=30
    )

    if health.status_code == 200:
        st.sidebar.success("🟢 API Online")
    else:
        st.sidebar.warning("🟡 API responded")

except Exception:
    st.sidebar.error("🔴 API Offline")


# ============================================================
# RECOMMENDATION BUTTON
# ============================================================

if st.button(
    "🔍 Get Recommendations",
    type="primary",
    use_container_width=True
):

    with st.spinner("Finding books for you..."):

        try:

            response = requests.post(
                f"{API_URL}/recommend",
                json={
                    "user_id": int(user_id),
                    "n": int(n)
                },
                timeout=120
            )

            if response.status_code != 200:

                st.error(
                    f"Recommendation API error: "
                    f"{response.status_code}"
                )

                st.code(response.text)

            else:

                data = response.json()

                recommendations = data.get(
                    "recommendations",
                    []
                )

                st.session_state["recommendations"] = recommendations
                st.session_state["user_id"] = int(user_id)
                st.session_state["n"] = int(n)

        except requests.exceptions.Timeout:

            st.error(
                "The API request timed out. "
                "Render may be waking up."
            )

        except Exception as e:

            st.error(
                f"Connection error: {e}"
            )


# ============================================================
# DISPLAY RECOMMENDATIONS
# ============================================================

if "recommendations" in st.session_state:

    recommendations = st.session_state["recommendations"]

    st.subheader(
        f"📖 Recommended Books for User "
        f"{st.session_state['user_id']}"
    )

    if not recommendations:

        st.warning(
            "No recommendations were returned."
        )

    else:

        for i, book in enumerate(
            recommendations,
            1
        ):

            title = book.get(
                "title",
                "Unknown Title"
            )

            author = book.get(
                "author",
                "Unknown Author"
            )

            publisher = book.get(
                "publisher",
                "Unknown Publisher"
            )

            image_url = book.get(
                "image_url_m"
            ) or book.get(
                "image_url_s"
            ) or book.get(
                "image_url_l"
            )

            score = book.get(
                "score",
                0
            )

            col1, col2 = st.columns(
                [1, 4]
            )

            with col1:

                if image_url:

                    try:
                        st.image(
                            image_url,
                            use_container_width=True
                        )
                    except Exception:

                        st.write("📕")

                else:

                    st.write("📕")

            with col2:

                st.markdown(
                    f"### {i}. {title}"
                )

                st.write(
                    f"**Author:** {author}"
                )

                st.write(
                    f"**Publisher:** {publisher}"
                )

                st.write(
                    f"**Similarity Score:** "
                    f"{float(score):.4f}"
                )

            st.divider()


    # ========================================================
    # GENAI EXPLANATION
    # ========================================================

    st.subheader("🤖 Why these books?")

    if st.button(
        "✨ Generate AI Explanation",
        use_container_width=True
    ):

        with st.spinner(
            "Generating explanation..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/explain",
                    json={
                        "user_id": int(
                            st.session_state["user_id"]
                        ),
                        "n": int(
                            st.session_state["n"]
                        )
                    },
                    timeout=240
                )

                if response.status_code == 200:

                    data = response.json()

                    explanation = data.get(
                        "explanation",
                        ""
                    )

                    genai_status = data.get(
                        "genai_status",
                        "unknown"
                    )

                    if explanation:

                        st.info(
                            explanation
                        )

                        st.caption(
                            f"GenAI status: "
                            f"{genai_status}"
                        )

                    else:

                        st.warning(
                            "No AI explanation returned."
                        )

                else:

                    st.error(
                        f"GenAI API error: "
                        f"{response.status_code}"
                    )

                    st.code(
                        response.text
                    )

            except requests.exceptions.Timeout:

                st.error(
                    "The AI request timed out."
                )

            except Exception as e:

                st.error(
                    f"GenAI connection error: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Book Recommendation System • "
    "Content-Based ML • FastAPI • Render • Hugging Face GenAI"
)

# 🚀 Deployment Guide: AURORA AI / HR Smart Assistant

This project is now prepared for deployment. Below are the steps and considerations for different platforms.

## 1. Option A: Streamlit Community Cloud (Recommended)
This is the **easiest** and most reliable way to host Streamlit.
- **Why?** It's designed specifically for Streamlit, supports persistence, and handles Python environments perfectly.
- **Steps:**
  1. Push this project to a **GitHub repository**.
  2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
  3. Click "New app" and select your repository.
  4. Use `app.py` as the main file path.

## 2. Option B: Vercel (Per your request)
Vercel is great for static/serverless, but Streamlit is a server-based app, so it has limitations:
- **Limitations:**
  - **Persistence:** The `history.db` (SQLite) is ephmeral on Vercel. Every time the app redeploys or sleeps, the history will be lost. To fix this, you should connect to a remote database like **Supabase** or **Neon**.
  - **Cold Starts:** Serverless functions can take a few seconds to wake up.
- **Steps:**
  1. Ensure you have the `vercel.json` already created in this folder.
  2. Use the Vercel CLI: `vercel --prod` or link to GitHub.

## 3. Option C: Render / Heroku
Good middle ground between Vercel and Streamlit Cloud.
- Supports a `Procfile`:
  ```text
  web: streamlit run app.py --server.port $PORT
  ```

---

### 📦 Current File Structure Checklist:
- ✅ `requirements.txt`: Cleaned up to only include necessary libraries.
- ✅ `vercel.json`: Initial configuration for Vercel's Python runtime.
- ✅ Models: All `.pkl` files (less than 1MB total) are included.
- ✅ Assets: `salary_banner.jpg` is linked correctly.

### 🛑 Important Note on Database:
The app currently uses a local `history.db`. For professional deployment, you should update the `DATABASE_URL` in `app.py` to use a cloud-hosted Postgres instance.

# -User-Friendly-Library-Book-Lens-
A smart book discovery platform built with Streamlit, OCR, ML recommender, and web scraping. Features include data cleaning, database storage, login with profile, favorites, and intelligent search. Developed by Tawfiq Al-Tahineh, Rama Al Jada, and Sondos Abu Mardiya.


---

# 📚 User Friendly Library (BookLens)

**User Friendly Library (BookLens)** is an intelligent and interactive book discovery platform. It combines **web scraping**, **data storage**, **data cleaning**, **OCR**, and **machine learning recommenders** with a modern **Streamlit UI**. The project was developed collaboratively by **Tawfiq Al-Tahineh**, **Rama Al Jada**, and **Sondos Abu Mardiya**.

## README.md (Full, Descriptive)

### 1) Overview

**User Friendly Library (BookLens)** is a smart book discovery app built with **Streamlit**. It integrates **OCR (Tesseract)** to read titles from cover images, **fuzzy matching** (RapidFuzz) to link covers to database records, and a clean **SQLite** workflow for storing books, profiles, favorites, and browsing history. The project was developed by **Tawfiq Al‑Tahineh**, **Rama Al Jada**, and **Sondos Abu Mardiya**.

### 2) Key Features

* 🔎 **Web Scraping**: Collects book data from google books (https://www.googleapis.com/books/v1/volumes)
* 🗄️ **Database Storage**: Uses SQLite (default) with SQLAlchemy, supports MySQL/PostgreSQL.
* 🧹 **Data Cleaning**: Normalizes text, removes duplicates, standardizes formatting.
* 🖼️ **OCR (Tesseract)**: Extracts text from book covers and fuzzy-matches with stored titles.
* 🤖 **ML Recommender System**: Hybrid recommendations (content-based + similarity).
* 🎨 **Modern UI**: Built with Streamlit, responsive and user-friendly , support Dark and Light mode.
* 👤 **Authentication & Profiles**: User registration, login, favorites, and history.
* 🔍 **Smart Search**: Keyword search with fuzzy matching and suggestions.
* 🔍 **Inteactive Chatbot** : Similar like Elizza.
* ⚡  **Performance Optimizations**: Local caching, parallel scraping, batch processing.
* 🔐 **Security**: Password hashing, environment-based secrets.
* 🧪 **Testing**: Comprehensive tests with `pytest` and `coverage`.
* 🤖 **TTS (gTTS)**: Optional "Listen" button inside the book details to play title/summary audio.

### 3) Tech Stack

* **Frontend**: Streamlit
* **OCR**: pytesseract + pillow (optional preprocessing)
* **Search/Similarity**: rapidfuzz
* **HTTP**: requests (fetch remote resources/covers when needed)
* **DB**: sqlite3 (built‑in), files managed locally (no server required)

### 4) Project Structure (suggested)

```
user-friendly-library/
├─ app.py                 # Streamlit entry point
├─ Code_DB (3).ipynb
├─ requirements.txt
├─ README.md
├─ LICENSE
```

### 5) Setup & Installation

#### Prerequisites

* Python 3.10+
* **Tesseract OCR** installed on your OS

  * Windows sample path: `C:\\Users\\<name>\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe`
  * Linux/macOS: `tesseract` available on PATH

#### Create env and install

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate    # Windows
pip install -r requirements.txt
```

#### Environment variables

Copy `.env.example` → `.env` and set:

```
DB_PATH=./books.sqlite
TESSERACT_CMD=C:\\Users\\<name>\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe
APP_SECRET=change-me
```

(If `TESSERACT_CMD` is omitted and tesseract is on PATH, it will be autodetected.)

### 6) Running

```bash
streamlit run app.py
```

Open: [http://localhost:8501](http://localhost:8501)

### 7) Usage Guide

* **Browse**: Explore books in a grid (8 per page). Click **Details** to see full metadata and the **Open Link** button.
* **Search**: Type a query; fuzzy matching helps even with typos (e.g., `harry poter`).
* **OCR**: Upload or drag a cover image; the app reads the title and tries to match it.
* **Favorites**: While viewing a book, click **Add to Favorites**. View them on the **Favorites** page.
* **Profile**: See session info and history. The **Listen** button (if enabled) plays an audio snippet via gTTS.

### 8) Database Schema (SQLite)

Minimal tables typically include:

* **books** (id,title,authors,publishedDate,industry_identifiers,selfLink,description,reading_text,reading_image,pageCount,printType,categories,thumbnail,language,infoLink,saleability,isEbook,pdf_available,)
* **users**(id, username, password\_hash, created\_at)
* **favorites**(user\_id, book\_id, created\_at)
* **history**(user\_id, book\_id, action, ts)

> Passwords are hashed with `bcrypt/passlib`. Avoid storing plaintext.

### 9) OCR Tips

* Prefer high‑resolution covers.
* Preprocess (grayscale, threshold) to improve OCR.
* Keep Tesseract language packs updated if you read non‑English text.

### 10) Testing

```bash
pytest -q
pytest --cov=src
```

### 11) Roadmap

* Advanced hybrid recommendations (content + collaborative)
* Admin dashboard (moderate/edit metadata)
* Export/import datasets; incremental scraping
* i18n (Arabic/English UI)



### 12) Security

* Never commit `.env` or database dumps.
* Rotate `APP_SECRET` regularly.
* Report vulnerabilities privately to the maintainers.

### 13) License

MIT License © 2025 Tawfiq Al‑Tahineh, Rama Al Jada, Sondos Abu Mardiya

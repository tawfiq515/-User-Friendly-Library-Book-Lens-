# -*- coding: utf-8 -*-
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract, re, sqlite3, pandas as pd, os, hashlib, secrets, tempfile, math, base64, html, random, json, io
from gtts import gTTS
from datetime import datetime
from rapidfuzz import fuzz, process  # للتطابق التشابهي

# ===================== Windows only Tesseract =====================
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\sondos\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# ===================== Config =====================
DB_FILE        = r"/home/tawfiq/Desktop/new_books_crawl.sqlite"
BOOKS_TABLE    = "books"
COVER_FALLBACK = r"/home/tawfiq/Desktop/coverbokkproject.jpeg"

PAGE_SIZE           = 8
PAGINATION_WINDOW   = 10
CARD_IFRAME_HEIGHT  = 560

st.set_page_config(page_title="📚 User Friendly Library", layout="wide")

# ===================== Theme (page-level) =====================
def inject_theme(dark: bool):
    if dark:
        bg, box, btn, text, primary, primary_text = "#0f172a", "#111827", "#374151", "#e5e7eb", "#2563eb", "#ffffff"
        muted = "#9ca3af"
        sidebar_head = "#0b1220"
    else:
        bg, box, btn, text, primary, primary_text = "#f5e6c8", "#fff3e0", "#d9b08c", "#000000", "#1f2937", "#ffffff"
        muted = "#444444"
        sidebar_head = "#ead7b2"

    st.markdown(f"""
    <style>
    :root {{
      --bg:{bg}; --box:{box}; --btn:{btn}; --text:{text};
      --primary:{primary}; --primary-text:{primary_text}; --muted:{muted};
    }}
    .stApp {{ background-color: var(--bg); color: var(--text); }}
    .stTextArea textarea, .stTextInput>div>input, .stPassword>div>input, div[data-baseweb="select"] > div {{
      background-color: var(--box)!important; color: var(--text)!important; border-radius:10px; border:none;
    }}
    .stButton > button[kind="primary"] {{ background: var(--primary); color: var(--primary-text); border:none; border-radius:10px; }}
    .stButton > button[kind="secondary"] {{ background: var(--btn); color: var(--text); border:none; border-radius:10px; }}
    label, .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: var(--text)!important; }}

    .helper-bubble {{
      font-size: 1.05rem; font-weight:700; line-height:1.4;
      background: #ffe9c5; color: #111; padding: 10px 12px; border-radius: 10px;
      border: 1px dashed #d1a577; margin: 6px 0 14px 0;
    }}

    .details {{
      direction: rtl; background: var(--box); border-radius: 16px; padding: 14px 18px; border: 1px solid #00000020;
    }}
    .details .muted {{ color: var(--muted); font-size: .95rem; }}
    .details .desc {{ margin-top: 10px; line-height: 1.6; }}
    .cover-box img {{ width: 160px; height: auto; border-radius: 10px; object-fit: cover; }}

    /* ==== Sidebar history table: نفس لون الموقع لكن أغمق وبولدر ==== */
    section[data-testid="stSidebar"] [data-testid="stDataFrame"] {{
      background: var(--box)!important;
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 12px;
      padding: 6px;
    }}
    section[data-testid="stSidebar"] [data-testid="stDataFrame"] thead tr th {{
      background: {sidebar_head}!important;
      color: var(--text)!important;
      font-weight: 800!important;
    }}
    section[data-testid="stSidebar"] [data-testid="stDataFrame"] tbody tr td {{
      background: var(--box)!important;
      color: var(--text)!important;
      font-weight: 600!important; /* بُولدَر */
      border-bottom: 1px solid rgba(255,255,255,0.06)!important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ===================== Sanitizers =====================
def strip_html_tags(s: str) -> str:
    if not s: return ""
    s = re.sub(r"(?is)<(script|style).*?>.*?(</\\1>)", "", s)
    s = re.sub(r"(?s)<[^>]+>", "", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", s).strip()

def safe_text(s: str) -> str:
    return html.escape(strip_html_tags(str(s or "")))

# ===================== ELIZA =====================
ELIZA_RULES = [
    (r".*\b(دور|ابحث|فتش|بدور|بحث)\b.*", "title: أكتب كلمة مفتاحية أو عنوان الكتاب"),
    (r".*\b(عنوان|title)\b.*", "title: تمام، شو العنوان اللي بدك إياه؟"),
    (r".*\b(أشهر|الأكثر قراءة|الترند|trend|popular)\b.*", "title: أجيب الأكثر شيوعًا ضمن فئتك المفضلة؟"),
    (r".*\b(popular|trending|hot)\b.*", "title: Want the trending picks within your favorite genre?"),
    (r".*\b(مساعدة|كيف استعمل|help)\b.*", "title: اكتب: ‘بحث + كلمة’ أو ‘فئة + اسم الفئة’ أو ارفع صورة."),
    (r".*\b(تعليمات الصوت|tts help)\b.*", "title: اضغط 🔊 لسماع الوصف."),
    (r".*\b(how to|guide|instructions)\b.*", "title: Type a keyword, pick a category, or upload a cover image."),
    (r".*", "title: مرحبا اكتب كلمة بحث أو ارفع صورة.")
]
def eliza_reply(text: str) -> str:
    t = (text or "").strip()
    for pat, resp in ELIZA_RULES:
        if re.search(pat, t, flags=re.I | re.U):
            return resp
    return "title: مرحبا اكتب كلمة بحث أو ارفع صورة."

# ===================== DB =====================
def db_conn(): return sqlite3.connect(DB_FILE, check_same_thread=False)

def column_exists(conn, table, column) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())

def create_auth_tables():
    with db_conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS auth_users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            pass_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL,
            favorites_json TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS auth_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            query TEXT,
            book_id TEXT,
            ts TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES auth_users(id)
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_auth_users_username ON auth_users(username)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_auth_history_user ON auth_history(user_id, ts)")
        c.execute("UPDATE auth_users SET favorites_json='[]' WHERE favorites_json IS NULL")
        c.commit()

def hash_pw(password, salt=None):
    salt = salt or secrets.token_hex(16)
    return hashlib.sha256((salt + password).encode('utf-8')).hexdigest(), salt

def register_user(username, password):
    try:
        with db_conn() as c:
            h, s = hash_pw(password)
            c.execute("INSERT INTO auth_users(username, pass_hash, salt, created_at, favorites_json) VALUES(?,?,?,?,?)",
                      (username, h, s, datetime.utcnow().isoformat(), "[]"))
            c.commit()
            return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "Username already exists."

def verify_login(username, password):
    with db_conn() as c:
        row = c.execute("SELECT id, pass_hash, salt FROM auth_users WHERE username=?", (username,)).fetchone()
    if not row: return None
    uid, ph, s = row
    h, _ = hash_pw(password, s)
    return uid if h == ph else None

def add_history(user_id, query=None, book_id=None, is_guest=False):
    if not user_id or is_guest: return
    with db_conn() as c:
        c.execute("INSERT INTO auth_history(user_id, query, book_id, ts) VALUES(?,?,?,?)",
                  (int(user_id), query, book_id, datetime.utcnow().isoformat()))
        c.commit()

def get_history(user_id, limit=50):
    if not user_id: return pd.DataFrame(columns=["query","book_id","ts"])
    with db_conn() as c:
        df = pd.read_sql("SELECT query, book_id, ts FROM auth_history WHERE user_id=? ORDER BY ts DESC LIMIT ?",
                         c, params=(int(user_id), int(limit)))
    return df

# ======== Favorites JSON =========
def load_user_favs(user_id: int) -> list:
    if user_id is None:
        return []
    try:
        uid = int(user_id)
    except Exception:
        return []
    with db_conn() as c:
        row = c.execute("SELECT favorites_json FROM auth_users WHERE id=?", (uid,)).fetchone()
    if not row or not row[0]:
        return []
    try:
        favs = json.loads(row[0])
        return [str(x) for x in favs] if isinstance(favs, list) else []
    except Exception:
        return []

def save_user_favs(user_id: int, favs: list):
    if user_id is None:
        return
    try:
        uid = int(user_id)
    except Exception:
        return
    data = json.dumps(list(dict.fromkeys([str(x) for x in favs])))
    with db_conn() as c:
        c.execute("UPDATE auth_users SET favorites_json=? WHERE id=?", (data, uid))
        c.commit()

def add_favorite(user_id: int, book_id: str):
    if user_id is None:
        return
    favs = load_user_favs(user_id)
    bid = str(book_id)
    if bid not in favs:
        favs.insert(0, bid)
        save_user_favs(user_id, favs)

def remove_favorite(user_id: int, book_id: str):
    if user_id is None:
        return
    favs = load_user_favs(user_id)
    bid = str(book_id)
    if bid in favs:
        favs = [x for x in favs if x != bid]
        save_user_favs(user_id, favs)

def is_favorite(user_id: int, book_id: str) -> bool:
    if user_id is None:
        return False
    favs = load_user_favs(user_id)
    return str(book_id) in favs

def list_favorites(user_id: int) -> pd.DataFrame:
    favs = load_user_favs(user_id)
    return pd.DataFrame({"book_id": favs})

# ===================== Books / OCR / Search / TTS =====================
def load_books():
    if not os.path.exists(DB_FILE): return pd.DataFrame()
    with db_conn() as c:
        return pd.read_sql(f"SELECT * FROM {BOOKS_TABLE}", c)

def ocr_text(file_like) -> str:
    if hasattr(file_like, "read"):
        try: file_like.seek(0)
        except Exception: pass
        data = file_like.read()
    else:
        data = file_like
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("L")
    w, h = img.size
    if max(w, h) < 1200:
        scale = 1200 / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.MedianFilter(3))
    cfg = r"--oem 3 --psm 6"
    try:
        text = pytesseract.image_to_string(img, lang="ara+eng", config=cfg)
    except Exception:
        text = pytesseract.image_to_string(img, lang="eng", config=cfg)
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text

def clean_text(text):
    text = re.sub(r"[^\w\s]", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def search_books(df, query=None, category=None):
    if df.empty: return pd.DataFrame()
    res = df.copy()
    if category:
        res = res[res['categories'].str.contains(category, case=False, na=False)]
    if query:
        q = clean_text(query)
        mask = (
            res['title'].str.lower().str.contains(q, na=False) |
            res['authors'].str.lower().str.contains(q, na=False) |
            res['description'].str.lower().str.contains(q, na=False)
        )
        res = res[mask].copy()
        res['score'] = res['title'].apply(lambda t: fuzz.token_sort_ratio(q, str(t)))
        res = res.sort_values(by='score', ascending=False)
    return res

# ========= OCR matching against TITLES only =========
def _norm(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def ocr_match_titles(df_books: pd.DataFrame, ocr_txt: str, top_k: int = 32) -> pd.DataFrame:
    if df_books.empty or not ocr_txt:
        return pd.DataFrame()
    titles = df_books.get("title")
    if titles is None:
        return pd.DataFrame()

    candidates = [(i, _norm(t)) for i, t in titles.items()]
    queries = [c for c in candidates if c[1]]
    idxs = [i for i, _ in queries]
    vals = [t for _, t in queries]

    matches = process.extract(
        _norm(ocr_txt),
        vals,
        scorer=fuzz.token_set_ratio,
        limit=min(top_k, len(vals))
    )
    rows = []
    for matched_val, score, pos in matches:
        df_idx = idxs[pos]
        row = df_books.iloc[df_idx].to_dict()
        row["ocr_score"] = int(score)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("ocr_score", ascending=False)
    return out

def play_text(text):
    tts = gTTS(text=text or "لا يوجد وصف.", lang="ar")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        return f.name

# --- NEW: TTS for American English title ---
def play_title_us_female(title_text: str):
    txt = (title_text or "").strip() or "No title"
    tts = gTTS(text=txt, lang="en", tld="com")  # American accent (default, typically female)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        return f.name

# ===================== Images =====================
def path_to_data_uri(path: str) -> str:
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""

def pick_image_src(thumb, reading_image) -> str:
    if reading_image and thumb:
        s = str(thumb)
        if s.startswith(("http://","https://")):
            return s
        if os.path.exists(s):
            uri = path_to_data_uri(s)
            if uri: return uri
    return path_to_data_uri(COVER_FALLBACK) if os.path.exists(COVER_FALLBACK) else ""

# ===================== Card HTML =====================
CARD_CSS = """
<style>
  :root { --box:#111827; --muted:#9ca3af; --text:#e5e7eb; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:var(--text);}
  a { text-decoration:none; color:inherit; }
  .card {
    display:flex; flex-direction:column; justify-content:space-between;
    background: var(--box); border-radius:14px; padding:12px; text-align:center;
    height: 520px; border: 1px solid #00000022; transition: transform .2s; cursor:pointer;
  }
  .card:hover { transform: translateY(-2px); }
  .card img { border-radius:10px; width:100%; aspect-ratio: 3/4; object-fit: cover; background:#eee; }
  .title { font-weight:800; font-size:1.02rem; margin-top:8px; line-height:1.35; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; min-height:2.7em; }
  .subtitle { font-style: italic; color:#93c5fd; display:-webkit-box; -webkit-line-clamp:1; -webkit-box-orient:vertical; overflow:hidden; min-height:1.4em; }
  .author { color:#10b981; font-weight:700; margin-top:4px; display:-webkit-box; -webkit-line-clamp:1; -webkit-box-orient:vertical; overflow:hidden; min-height:1.4em; }
  .meta { color:var(--muted); font-size:.9rem; min-height:1.3em; }
</style>
"""
def build_card_html(row) -> str:
    title    = safe_text(getattr(row, 'title', 'غير معروف'))
    title_en = safe_text(getattr(row, 'title_en', ''))
    authors  = safe_text(getattr(row, 'authors', 'غير معروف'))
    cats     = safe_text(getattr(row, 'categories', 'غير محدد'))

    thumb = getattr(row, 'thumbnail', None)
    reading_image = getattr(row, 'reading_image', 0)
    info_link = str(getattr(row, 'infoLink', '') or '')
    src = pick_image_src(thumb, reading_image)

    title_en_html = f'<div class="subtitle">{title_en}</div>' if title_en else ''
    pdf_status = getattr(row, 'pdf_available', 0)
    pdf_html = "📄 متوفر" if pdf_status else "❌ لا توجد نسخة PDF"

    body = f"""
      {CARD_CSS}
      <div class="card">
        <div><img src="{src}" alt="cover" /></div>
        <div class="title">{title}</div>
        {title_en_html}
        <div class="author">{authors}</div>
        <div class="meta">الفئة: {cats}</div>
        <div class="meta">{pdf_html}</div>
      </div>
    """
    if info_link.startswith(("http://","https://")):
        return f'<a href="{html.escape(info_link, quote=True)}" target="_blank">{body}</a>'
    return body

def safe_key_from_title(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    return s[:20] if s else "x"

def render_details_panel(book: dict):
    if not book: return
    thumb = book.get("thumbnail", "")
    reading_image = str(book.get("reading_image", "0")) in ("1", "True", "true")
    cover_src = pick_image_src(thumb, reading_image)

    with st.container():
        st.markdown('<div class="details">', unsafe_allow_html=True)
        c_desc, c_side, c_img = st.columns([7,3,2], gap="large")

        with c_img:
            st.markdown('<div class="cover-box">', unsafe_allow_html=True)
            st.image(cover_src, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_side:
            title = safe_text(book.get("title") or "بدون عنوان")
            title_en = safe_text(book.get("title_en") or "")
            st.markdown(f"### **{title}**")
            authors = safe_text(book.get("authors") or "غير معروف")
            pages = safe_text(book.get("pageCount") or "")
            pub = safe_text(book.get("publishedDate") or "")
            publisher = safe_text(book.get("publisher") or "")
            st.markdown(f'<div class="muted">{publisher} • {pub} • {pages} صفحة</div>', unsafe_allow_html=True)
            st.markdown(f"{authors}")
            info = book.get("infoLink") or ""
            if isinstance(info, str) and info.startswith("http"):
                st.link_button("🌐 فتح الصفحة الرسمية", info, type="primary")

            # 🔊 الأزرار
            desc_for_tts = strip_html_tags(book.get("description") or "")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                if st.button("🔊 استمع للوصف", key=f"listen_{book.get('id','')}", type="secondary"):
                    st.audio(play_text(desc_for_tts), format="audio/mp3")
            with col_t2:
                # يقرأ عنوان الكتاب بالإنجليزية الأمريكية (إن وُجد عنوان إنجليزي نستخدمه)
                title_to_read = title_en if title_en else title
                if st.button("🔊 استمع للعنوان (EN – American)", key=f"listen_title_{book.get('id','')}", type="secondary"):
                    st.audio(play_title_us_female(title_to_read), format="audio/mp3")

            if st.session_state.get("is_guest", True):
                st.caption("⭐ سجّل الدخول لإضافة هذا الكتاب إلى المفضلة.")
            else:
                bid = str(book.get("id", ""))
                fav_now = is_favorite(st.session_state.user_id, bid)
                if st.button(("🗑️ إزالة من المفضلة" if fav_now else "⭐ إضافة إلى المفضلة"),
                             key=f"fav_details_{bid}",
                             type=("secondary" if fav_now else "primary")):
                    if fav_now:
                        remove_favorite(st.session_state.user_id, bid)
                    else:
                        add_favorite(st.session_state.user_id, bid)
                    st.rerun()

        with c_desc:
            desc = strip_html_tags(book.get("description") or "")
            st.markdown(f'<div class="desc">{html.escape(desc) if desc else "لا يوجد وصف."}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ===================== Pagination helpers =====================
def reset_page_if_query_changed(q_key: str, cat_key: str):
    last_q = st.session_state.get("last_query_value")
    last_c = st.session_state.get("last_category_value")
    cur_q  = st.session_state.get(q_key, "")
    cur_c  = st.session_state.get(cat_key, "")
    if last_q != cur_q or last_c != cur_c:
        st.session_state.current_page = 1
    st.session_state["last_query_value"] = cur_q
    st.session_state["last_category_value"] = cur_c

def get_total_pages(total_count: int) -> int:
    return max(1, math.ceil(total_count / PAGE_SIZE))

def slice_df_for_page(df: pd.DataFrame, page: int) -> pd.DataFrame:
    start = (page - 1) * PAGE_SIZE
    end   = start + PAGE_SIZE
    return df.iloc[start:end]

def pagination_bar(total_pages: int, bar_id: str):
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    cp = int(st.session_state.current_page)

    st.caption(f"أنت الآن في الصفحة **{cp}** من **{total_pages}**")

    left, _, right = st.columns([1, 6, 1])
    with left:
        if st.button("⬅️ السابق", key=f"{bar_id}_prev", use_container_width=True, disabled=cp <= 1):
            st.session_state.current_page = cp - 1
            st.rerun()
    with right:
        if st.button("التالي ➡️", key=f"{bar_id}_next", use_container_width=True, disabled=cp >= total_pages):
            st.session_state.current_page = cp + 1
            st.rerun()

    window = PAGINATION_WINDOW
    start = max(1, cp - window // 2)
    end   = min(total_pages, start + window - 1)
    start = max(1, end - window + 1)

    cols = st.columns(end - start + 1)
    for i, p in enumerate(range(start, end + 1)):
        btn_type = "primary" if p == cp else "secondary"
        if cols[i].button(f"{p}", key=f"{bar_id}_pg_{p}", type=btn_type):
            st.session_state.current_page = p
            st.rerun()

# ===================== Session =====================
def init_session():
    defaults = dict(stage="auth", user_id=None, username=None, auth_mode="login",
                    is_guest=False, current_page=1, last_query_value="", last_category_value="",
                    dark_mode=False, selected_book=None,
                    profile_tab="history",
                    dice_value=None,
                    dice_spin_seed=0)
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

create_auth_tables()
init_session()
inject_theme(st.session_state.dark_mode)

# ===================== 3D Dice (template-safe HTML) =====================
def build_dice_html(final_value: int, seed: int = 0, size: int = 380) -> str:
    html_tpl = """
    <html>
    <head>
      <meta charset="utf-8"/>
      <script src="https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.min.js"></script>
    </head>
    <body style="margin:0;background:transparent;">
      <canvas id="diceCanvas" width="__SIZE__" height="__SIZE__"></canvas>
      <script>
        const FINAL_VALUE = __FV__;
        const SEED = __SEED__;
        const canvas = document.getElementById("diceCanvas");
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({canvas: canvas, antialias: true, alpha: true});
        renderer.setSize(__SIZE__, __SIZE__);
        renderer.setClearColor(0x000000, 0);

        function mulberry32(a){return function(){var t=a+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296}}
        const rng = mulberry32(SEED || (Date.now() % 1e9));

        const dl = new THREE.DirectionalLight(0xffffff, 1.1);
        dl.position.set(4,6,8);
        scene.add(dl);
        scene.add(new THREE.AmbientLight(0x808080, 0.6));

        function faceMat(num){
          const s=256, r=22;
          const c = document.createElement('canvas');
          c.width=s; c.height=s;
          const ctx=c.getContext('2d');
          ctx.fillStyle='#ffffff';
          ctx.fillRect(0,0,s,s);
          ctx.fillStyle='#000000';
          ctx.strokeStyle='#111111';
          ctx.lineWidth=4;
          ctx.strokeRect(4,4,s-8,s-8);

          const posMap = {
            1:[[0.5,0.5]],
            2:[[0.25,0.25],[0.75,0.75]],
            3:[[0.25,0.25],[0.5,0.5],[0.75,0.75]],
            4:[[0.25,0.25],[0.25,0.75],[0.75,0.25],[0.75,0.75]],
            5:[[0.25,0.25],[0.25,0.75],[0.75,0.25],[0.75,0.75],[0.5,0.5]],
            6:[[0.25,0.25],[0.25,0.5],[0.25,0.75],[0.75,0.25],[0.75,0.5],[0.75,0.75]]
          }[num];

          posMap.forEach(p=>{
            ctx.beginPath();
            ctx.arc(p[0]*s, p[1]*s, r, 0, Math.PI*2);
            ctx.fill();
          });
          const tex = new THREE.CanvasTexture(c);
          return new THREE.MeshStandardMaterial({map: tex});
        }

        const materials = [1,2,3,4,5,6].map(faceMat);
        const geo = new THREE.BoxGeometry(1,1,1);
        const dice = new THREE.Mesh(geo, materials);
        dice.castShadow = true; dice.receiveShadow = true;
        scene.add(dice);
        camera.position.z = 3;

        const targetMap = {
          1: {rx: 0,           ry: -Math.PI/2},
          2: {rx: 0,           ry:  Math.PI/2},
          3: {rx:  Math.PI/2,  ry:  0},
          4: {rx: -Math.PI/2,  ry:  0},
          5: {rx: 0,           ry:  0},
          6: {rx:  Math.PI,    ry:  0}
        };

        function rollTo(value){
          const t = targetMap[value] || targetMap[6];
          const kx = (2 + Math.floor(rng()*3)) * 2 * Math.PI;
          const ky = (2 + Math.floor(rng()*3)) * 2 * Math.PI;
          const kz = (1 + Math.floor(rng()*2)) * 2 * Math.PI;

          const rx = t.rx + kx;
          const ry = t.ry + ky;
          const rz = 0    + kz;

          const duration = 900 + Math.floor(rng()*500);
          const start = performance.now();

          function anim(ts){
            const p = Math.min((ts - start)/duration, 1);
            const e = 1 - Math.pow(1-p, 3);
            dice.rotation.x = rx * e;
            dice.rotation.y = ry * e;
            dice.rotation.z = rz * e;
            renderer.render(scene, camera);
            if(p < 1) requestAnimationFrame(anim);
          }
          requestAnimationFrame(anim);
        }

        renderer.render(scene, camera);
        rollTo(FINAL_VALUE);
      </script>
    </body>
    </html>
    """
    html_tpl = html_tpl.replace("__FV__", str(int(final_value if final_value in [1,2,3,4,5,6] else 6)))
    html_tpl = html_tpl.replace("__SEED__", str(int(seed)))
    html_tpl = html_tpl.replace("__SIZE__", str(int(size)))
    return html_tpl

# ===================== ML-lite Recommendations =====================
def recommend_from_history(df_books: pd.DataFrame, hist_df: pd.DataFrame, top_n=24):
    if df_books.empty or hist_df.empty: return pd.DataFrame()
    hist_book_ids = [x for x in hist_df['book_id'].dropna().tolist() if str(x).strip() != ""]
    df_seen = df_books[df_books['id'].astype(str).isin([str(bid) for bid in hist_book_ids])] if 'id' in df_books.columns else pd.DataFrame()
    seed_titles = set(df_seen['title'].dropna().astype(str).tolist())
    seed_authors = set(sum([str(a).split(",") for a in df_seen['authors'].fillna("").astype(str)], []))
    seed_cats = set(sum([str(c).split("/") for c in df_seen['categories'].fillna("").astype(str)], []))

    hist_queries = [q for q in hist_df['query'].dropna().tolist() if str(q).strip() != ""]
    for q in hist_queries[:10]:
        seed_titles.add(q)

    def score_row(r):
        s = 0
        t = str(r.get('title','') or '')
        a = str(r.get('authors','') or '')
        c = str(r.get('categories','') or '')
        if any(x.strip() and x.strip().lower() in a.lower() for x in seed_authors): s += 30
        if any(x.strip() and x.strip().lower() in c.lower() for x in seed_cats): s += 25
        s += max([fuzz.token_set_ratio(t, seed) for seed in seed_titles] or [0]) * 0.4
        return s

    df = df_books.copy()
    df['rec_score'] = df.apply(score_row, axis=1)
    if not df_seen.empty:
        df = df[~df['id'].astype(str).isin(df_seen['id'].astype(str))]
    df = df.sort_values('rec_score', ascending=False)
    return df.head(top_n)

# ===================== UI helpers =====================
def render_books_grid(df_page):
    if df_page.empty:
        st.warning("⚠ لا توجد نتائج مطابقة.")
        return
    cols = st.columns(4)
    for i, row in enumerate(df_page.itertuples(index=False)):
        with cols[i % 4]:
            components.html(build_card_html(row), height=CARD_IFRAME_HEIGHT, scrolling=False)

            bid = str(getattr(row, 'id', i))
            fav_now = False
            if not st.session_state.get("is_guest", True):
                fav_now = is_favorite(st.session_state.user_id, bid)

            fav_label = ("🗑️ إزالة من المفضلة" if fav_now else "⭐ إضافة إلى المفضلة")
            fav_disabled = st.session_state.get("is_guest", True)

            if st.button(fav_label,
                         key=f"fav_{safe_key_from_title(getattr(row,'title',''))}_{bid}",
                         type=("secondary" if fav_now else "primary"),
                         disabled=fav_disabled):
                if fav_now:
                    remove_favorite(st.session_state.user_id, bid)
                else:
                    add_favorite(st.session_state.user_id, bid)
                st.rerun()

            if st.button("📑 تفاصيل",
                         key=f"details_{safe_key_from_title(getattr(row,'title',''))}_{bid}",
                         type="secondary"):
                try:
                    row_dict = row._asdict()
                except Exception:
                    row_dict = {col: getattr(row, col, None) for col in df_page.columns}
                st.session_state.selected_book = {k: ("" if v is None else v) for k, v in row_dict.items()}
                if not st.session_state.is_guest:
                    add_history(st.session_state.user_id, book_id=str(bid), is_guest=False)
                st.rerun()

def profile_sidebar(df_books):
    with st.sidebar:
        st.markdown("## 👤 الملف الشخصي")
        initials = (st.session_state.username or "U")[:2].upper()
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;">
          <div style="width:40px;height:40px;border-radius:50%;background:#4b5563;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800;">
            {initials}
          </div>
          <div><b>{st.session_state.username}</b></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        tab = st.radio("القائمة", options=["📜 السجل","🤖 التوصيات","⭐ المفضلة"],
                       index={"history":0,"recs":1,"favorites":2}[
                           st.session_state.profile_tab if st.session_state.profile_tab in ["history","recs","favorites"] else "history"
                       ])
        st.session_state.profile_tab = ("history" if tab=="📜 السجل" else ("recs" if tab=="🤖 التوصيات" else "favorites"))

        hist_df = get_history(st.session_state.user_id, limit=50)

        if st.session_state.profile_tab == "history":
            if hist_df.empty:
                st.info("لا يوجد سجل بعد.")
            else:
                st.write("**أحدث النشاط**")
                view = hist_df.loc[:, ["query", "ts"]].copy()
                view.rename(columns={"query": "الطلب", "ts": "الوقت"}, inplace=True)
                try:
                    view["الوقت"] = pd.to_datetime(view["الوقت"]).dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass
                st.dataframe(view, use_container_width=True, height=260)

        elif st.session_state.profile_tab == "recs":
            if df_books.empty:
                st.warning("لا توجد بيانات كتب.")
            else:
                if hist_df.empty:
                    st.info("ابدأ بالبحث أو فتح تفاصيل كتاب للحصول على توصيات.")
                else:
                    df_recs = recommend_from_history(df_books, hist_df, top_n=8)
                    st.write("**اقتراحات لأجلك**")
                    if df_recs.empty:
                        st.info("لا توجد توصيات كافية حتى الآن.")
                    else:
                        for _, r in df_recs.head(8).iterrows():
                            t = str(r.get('title','غير معروف'))
                            a = str(r.get('authors',''))
                            st.markdown(f"• **{t[:50]}** — {a[:40]}")

        else:
            if st.session_state.get("is_guest", True):
                st.info("سجّل الدخول لعرض مفضلتك.")
            else:
                fav_df = list_favorites(st.session_state.user_id)
                if fav_df.empty:
                    st.info("ما في كتب مضافة للمفضلة لسه.")
                else:
                    st.write(f"**عدد المفضلة:** {len(fav_df)}")
                    book_ids = set(str(x) for x in fav_df['book_id'].tolist())
                    fav_books = df_books[df_books['id'].astype(str).isin(book_ids)].copy()
                    for _, r in fav_books.iterrows():
                        t = str(r.get('title','غير معروف'))
                        a = str(r.get('authors',''))
                        bid = str(r.get('id',''))
                        cols = st.columns([6,2,2])
                        with cols[0]:
                            st.markdown(f"**{t}** — {a}")
                        with cols[1]:
                            if st.button("📑 تفاصيل", key=f"fav_details_btn_{bid}", type="secondary"):
                                st.session_state.selected_book = {k: ("" if pd.isna(v) else v) for k, v in r.to_dict().items()}
                                st.rerun()
                        with cols[2]:
                            if st.button("🗑️ إزالة", key=f"fav_remove_btn_{bid}", type="primary"):
                                remove_favorite(st.session_state.user_id, bid)
                                st.rerun()

# ===================== Pages =====================
def page_auth():
    st.title("📚 User Friendly Library — 🔐 Login")
    _, toggle_col = st.columns([3,1])
    with toggle_col:
        st.session_state.dark_mode = st.toggle("🌙 الوضع الداكن", value=st.session_state.dark_mode, key="toggle_dark_login")
        inject_theme(st.session_state.dark_mode)

    colA, colB, colC = st.columns([1,1,1])
    with colA:
        if st.button("🔑 Login", use_container_width=True): st.session_state.auth_mode = "login"
    with colB:
        if st.button("🆕 Sign up", use_container_width=True): st.session_state.auth_mode = "signup"
    with colC:
        if st.button("🚪 Continue as Guest", use_container_width=True):
            st.session_state.stage = "app"; st.session_state.username = "Guest"
            st.session_state.user_id = None; st.session_state.is_guest = True
            st.success("تم الدخول كضيف. لن يتم حفظ أي معلومات.")

    st.markdown("---")
    if st.session_state.auth_mode == "login":
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        if st.button("Sign in", use_container_width=True, key="do_login"):
            uid = verify_login(u, p)
            if uid:
                st.session_state.user_id = uid; st.session_state.username = u
                st.session_state.is_guest = False; st.session_state.stage = "app"
                st.success(f"Welcome {u} 🌟")
            else:
                st.error("Invalid credentials.")
    else:
        u = st.text_input("New username", key="signup_user")
        p = st.text_input("New password", type="password", key="signup_pass")
        if st.button("Create account", use_container_width=True, key="do_signup"):
            ok, msg = register_user(u, p); (st.success if ok else st.error)(msg)

def page_app():
    st.title("📚 User Friendly Library")

    left, mid, right = st.columns([1,2,1])
    with right:
        st.session_state.dark_mode = st.toggle("🌙 الوضع الداكن", value=st.session_state.dark_mode, key="toggle_dark_app")
        inject_theme(st.session_state.dark_mode)

    df_books = load_books()
    if not st.session_state.get("is_guest", True) and st.session_state.get("username"):
        profile_sidebar(df_books)

    # ======== زر النرد + واجهة 3D ========
    col0, col_dice, col2 = st.columns([1.2,1.4,1.2])

    with col0:
        if st.button("🎲 ارمِ النرد", key="roll_btn"):
            st.session_state.dice_value = random.randint(1, 6)
            st.session_state.dice_spin_seed += 1
            st.info(f"نتيجة النرد: {st.session_state.dice_value}")
            st.session_state.current_page = 1

        st.caption("💡 مو عارف شو تدور؟ جرّب النرد لاكتشاف كتب عشوائية. "
                   "إذا اخترت فئة، النرد يعرض عيّنة **من نفس الفئة** فقط؛ "
                   "وإذا ما اخترت أي شيء يعرض من **كل الفئات**.")
        with st.expander("لماذا أستخدم 🎲 النرد؟"):
            st.markdown("""
- للاستكشاف العشوائي لما ما عندك كتاب معيّن ببالك.
- يحترم الفئة المختارة (مثلاً **Science**).
- كل رمية تعرض لغاية **قيمة النرد × حجم الصفحة** (حجم الصفحة = 8).
- مفيد للاقتراحات السريعة وفتح الشهية للقراءة.
""")

    with col_dice:
        dice_val = st.session_state.get("dice_value") or 6
        dice_html = build_dice_html(dice_val, seed=st.session_state.get("dice_spin_seed", 0), size=360)
        components.html(dice_html, height=400, scrolling=False)

    # ======== شريط البحث ========
    query = st.text_input("🔍 اكتب اسم كتاب / مؤلف / كلمة في الوصف", key="query_input")

    # ======== ELIZA ========
    st.markdown(f"""<div class="helper-bubble"><b>🧭 User Friendly Library: {eliza_reply(query)}</b></div>""",
                unsafe_allow_html=True)

    # ======== رفع الصورة ========
    uploaded = st.file_uploader("📷 ارفع صورة الغلاف (اختياري)", type=["png","jpg","jpeg"])

    used_source = None
    df_res = pd.DataFrame()
    ocr_txt = ""

    if uploaded:
        st.image(uploaded, caption="📷 Uploaded cover", width=260)
        try:
            ocr_txt = ocr_text(uploaded)
            if ocr_txt:
                st.caption(f"📝 OCR: {ocr_txt[:120]}{'...' if len(ocr_txt)>120 else ''}")
                df_res = ocr_match_titles(df_books, ocr_txt, top_k=PAGE_SIZE*3)
                used_source = "ocr"
                if df_res.empty:
                    st.info("لم يتم العثور على تطابقات واضحة من الصورة — جرّب كتابة اسم تقريبـي في البحث.")
            else:
                st.warning("لم أتمكن من قراءة نص واضح من الصورة.")
        except Exception as e:
            st.warning(f"OCR failed: {e}")

    if used_source is None:
        cats = df_books['categories'].dropna().unique().tolist() if not df_books.empty else []
        cat = st.selectbox("📂 اختر فئة (اختياري)", [""] + cats, key="category_select") if cats else ""
        st.session_state.setdefault("query_input", ""); st.session_state.setdefault("category_select", "")
        reset_page_if_query_changed("query_input", "category_select")

        used_query = (query or "").strip()
        df_res = search_books(df_books, query=(used_query or None), category=(cat or None))
        used_source = "text"

    if used_source in ("ocr", "text") and not st.session_state.get("is_guest", True):
        add_history(st.session_state.user_id, query=(ocr_txt if used_source=="ocr" else (query or "")), is_guest=False)

    total_count = len(df_res)
    total_pages = get_total_pages(total_count)
    st.caption(f"عدد النتائج: **{total_count}** — عدد الصفحات: **{total_pages}** (كل صفحة {PAGE_SIZE} عنصر)")

    if st.session_state.get("selected_book"):
        render_details_panel(st.session_state.selected_book)
        st.markdown("")

    pagination_bar(total_pages, bar_id="top")
    cp = st.session_state.current_page
    df_page = slice_df_for_page(df_res.reset_index(drop=True), cp)
    render_books_grid(df_page)
    st.divider()
    pagination_bar(total_pages, bar_id="bottom")

# ===================== Run =====================
create_auth_tables()
if st.session_state.get("stage", "auth") == "auth":
    page_auth()
else:
    page_app()

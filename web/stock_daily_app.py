import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib import request, error

import pandas as pd
import streamlit as st


RESULTS_ROOT = Path(__file__).resolve().parents[1] / "stock_results_daily"
DB_PATH = Path(__file__).resolve().parent / "chat_history.sqlite"
FALLBACK_DB_PATH = Path.home() / ".cache" / "tslib" / "chat_history.sqlite"

DATE_FILE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})\.csv$")

DEFAULT_BASE_URL = os.getenv("AI_BASE_URL", "https://api.openai.com")
DEFAULT_API_KEY = os.getenv("AI_API_KEY", "")
DEFAULT_MODEL = os.getenv("AI_DEFAULT_MODEL", "gpt-4o-mini")
DEFAULT_MODEL_LIST = os.getenv("AI_MODELS", DEFAULT_MODEL)


@dataclass
class ChatSession:
    session_id: str
    title: str
    model: str
    provider: str
    created_at: str


def _resolve_db_path() -> Path:
    if "chat_db_path" in st.session_state:
        return Path(st.session_state.chat_db_path)
    candidates = [DB_PATH, FALLBACK_DB_PATH]
    last_error = None
    for path in candidates:
        try:
            os.makedirs(path.parent, exist_ok=True)
            conn = sqlite3.connect(path)
            conn.close()
            st.session_state.chat_db_path = str(path)
            return path
        except sqlite3.OperationalError as exc:
            last_error = exc
            continue
    raise last_error  # type: ignore[misc]


def _db_connect():
    path = _resolve_db_path()
    return sqlite3.connect(path)


def _init_db():
    conn = _db_connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            model TEXT,
            provider TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _list_sessions() -> List[ChatSession]:
    conn = _db_connect()
    rows = conn.execute(
        "SELECT id, title, model, provider, created_at FROM sessions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [ChatSession(*row) for row in rows]


def _create_session(model: str, provider: str) -> ChatSession:
    session_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat(timespec="seconds")
    title = "新对话"
    conn = _db_connect()
    conn.execute(
        "INSERT INTO sessions (id, title, model, provider, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, title, model, provider, created_at),
    )
    conn.commit()
    conn.close()
    return ChatSession(session_id, title, model, provider, created_at)


def _update_session_title(session_id: str, title: str):
    conn = _db_connect()
    conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))
    conn.commit()
    conn.close()


def _delete_session(session_id: str):
    conn = _db_connect()
    conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()
    conn.close()


def _load_messages(session_id: str) -> List[Dict[str, str]]:
    conn = _db_connect()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"role": role, "content": content, "created_at": created_at} for role, content, created_at in rows]


def _append_message(session_id: str, role: str, content: str):
    conn = _db_connect()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.utcnow().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def _load_setting(key: str, default: str) -> str:
    conn = _db_connect()
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row and row[0] is not None else default


def _save_setting(key: str, value: str):
    conn = _db_connect()
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()
    conn.close()


def _ensure_chat_state() -> List[ChatSession]:
    if "chat_provider" not in st.session_state:
        st.session_state.chat_provider = _load_setting("chat_provider", "openai-compatible")
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = _load_setting("chat_model", DEFAULT_MODEL)
    if "chat_base_url" not in st.session_state:
        st.session_state.chat_base_url = _load_setting("chat_base_url", DEFAULT_BASE_URL)
    if "chat_api_key" not in st.session_state:
        st.session_state.chat_api_key = _load_setting("chat_api_key", DEFAULT_API_KEY)
    if "chat_model_list" not in st.session_state:
        stored_models = _load_setting("chat_model_list", DEFAULT_MODEL_LIST)
        st.session_state.chat_model_list = [m.strip() for m in stored_models.split(',') if m.strip()]

    sessions = _list_sessions()
    if "chat_session_id" not in st.session_state:
        if sessions:
            st.session_state.chat_session_id = sessions[0].session_id
        else:
            new_session = _create_session(st.session_state.chat_model, st.session_state.chat_provider)
            st.session_state.chat_session_id = new_session.session_id
            sessions = _list_sessions()
    else:
        if sessions and st.session_state.chat_session_id not in {s.session_id for s in sessions}:
            st.session_state.chat_session_id = sessions[0].session_id
    return sessions


def _call_openai_compatible(messages: List[Dict[str, str]], base_url: str, api_key: str,
                            model: str, temperature: float = 0.2, timeout: int = 60) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    url = base_url + "/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"API error: {exc.code} {detail}") from exc

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("empty response from API")
    return choices[0]["message"]["content"].strip()


def _list_prediction_files(results_root: Path) -> Dict[str, Path]:
    if not results_root.exists():
        return {}
    candidates = {}
    for path in results_root.glob("*.csv"):
        match = DATE_FILE_PATTERN.match(path.name)
        if match:
            candidates[match.group(1)] = path
    return dict(sorted(candidates.items(), key=lambda x: x[0], reverse=True))


def _load_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "pred_date" in df.columns:
        df["pred_date"] = pd.to_datetime(df["pred_date"], errors="coerce")
    return df


def _render_chat_panel():
    _ensure_chat_state()
    st.markdown("### AI助手")

    messages = _load_messages(st.session_state.chat_session_id)
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    status_placeholder = st.empty()
    prompt = st.chat_input("输入你的问题...")
    if prompt:
        _append_message(st.session_state.chat_session_id, "user", prompt)
        if len(prompt.strip()) > 0:
            status_placeholder.info("思考中…")
            if st.session_state.chat_api_key:
                try:
                    with st.spinner("思考中…"):
                        response = _call_openai_compatible(
                            messages=[{"role": m["role"], "content": m["content"]} for m in messages]
                            + [{"role": "user", "content": prompt}],
                            base_url=st.session_state.chat_base_url,
                            api_key=st.session_state.chat_api_key,
                            model=st.session_state.chat_model,
                        )
                except Exception as exc:
                    response = f"调用失败: {exc}"
            else:
                response = "请先在设置中填写 API Key。"

            _append_message(st.session_state.chat_session_id, "assistant", response)
            if len(messages) == 0:
                _update_session_title(st.session_state.chat_session_id, prompt[:16])
            st.rerun()


def main():
    st.set_page_config(page_title="Daily Rank-Sum", layout="wide")
    st.title("每日预测 Rank-Sum")
    _init_db()

    with st.sidebar:
        st.markdown("### 数据源")
        results_root = st.text_input("预测结果目录", str(RESULTS_ROOT))
        results_root = Path(results_root)
        files = _list_prediction_files(results_root)

        if not files:
            st.warning("未找到预测结果文件，请先运行 daily 推理脚本。")
            selected_date = None
            selected_path = None
        else:
            selected_date = st.selectbox("选择预测日期", list(files.keys()))
            selected_path = files[selected_date]

        st.markdown("---")
        st.markdown("### AI助手")
        sessions = _ensure_chat_state()
        session_options = {
            f"{s.title} ({s.model}) · {s.session_id[:8]}": s.session_id for s in sessions
        }
        current_label = None
        for label, sid in session_options.items():
            if sid == st.session_state.chat_session_id:
                current_label = label
                break
        if current_label is None and session_options:
            current_label = list(session_options.keys())[0]

        if session_options:
            selected_label = st.selectbox(
                "会话",
                list(session_options.keys()),
                index=list(session_options.keys()).index(current_label) if current_label else 0,
            )
            st.session_state.chat_session_id = session_options[selected_label]
        else:
            st.write("暂无会话")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("新建"):
                new_session = _create_session(st.session_state.chat_model, st.session_state.chat_provider)
                st.session_state.chat_session_id = new_session.session_id
                st.rerun()
        with col_b:
            if sessions and st.button("删除"):
                _delete_session(st.session_state.chat_session_id)
                sessions = _list_sessions()
                if sessions:
                    st.session_state.chat_session_id = sessions[0].session_id
                else:
                    new_session = _create_session(st.session_state.chat_model, st.session_state.chat_provider)
                    st.session_state.chat_session_id = new_session.session_id
                st.rerun()

        with st.expander("设置", expanded=False):
            base_url = st.text_input("Base URL", st.session_state.chat_base_url)
            st.caption("Base URL 可带或不带 /v1，系统会自动修正。")
            normalized_base_url = base_url.rstrip("/")
            if normalized_base_url.endswith("/v1"):
                normalized_base_url = normalized_base_url[:-3]
            if base_url != st.session_state.chat_base_url:
                st.session_state.chat_base_url = normalized_base_url
                _save_setting("chat_base_url", normalized_base_url)

            api_key = st.text_input("API Key", st.session_state.chat_api_key, type="password")
            if api_key != st.session_state.chat_api_key:
                st.session_state.chat_api_key = api_key
                _save_setting("chat_api_key", api_key)

            model_list = st.text_input("模型列表(逗号分隔)", ",".join(st.session_state.chat_model_list))
            new_models = [m.strip() for m in model_list.split(',') if m.strip()]
            if new_models and new_models != st.session_state.chat_model_list:
                st.session_state.chat_model_list = new_models
                _save_setting("chat_model_list", ",".join(new_models))

            model = st.selectbox("模型", st.session_state.chat_model_list)
            if model != st.session_state.chat_model:
                st.session_state.chat_model = model
                _save_setting("chat_model", model)

    if selected_path is not None:
        df = _load_prediction(selected_path)
        st.caption(f"当前文件: {selected_path}")

        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            code_filter = st.text_input("代码筛选", "")
        with col2:
            top_n = st.number_input("显示前 N 行", min_value=10, max_value=1000, value=200, step=10)
        with col3:
            st.download_button(
                "下载 CSV",
                data=selected_path.read_bytes(),
                file_name=selected_path.name,
                mime="text/csv",
            )

        if code_filter:
            df = df[df["code"].astype(str).str.contains(code_filter.strip(), case=False)]

        if "rank_sum" in df.columns:
            df = df.sort_values("rank_sum", ascending=True)

        st.dataframe(df.head(int(top_n)), use_container_width=True)

    st.markdown("---")
    _render_chat_panel()


if __name__ == "__main__":
    main()

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

def get_llm() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY が見つかりません。.env か Streamlit Secrets を設定してください。")
        st.stop()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

SYSTEM_PROMPTS = {
    "エンジニア": "あなたは熟練のソフトウェアエンジニアです。コード例や設計のトレードオフも説明してください。",
    "マーケター": "あなたはデータドリブンなマーケターです。ターゲット、訴求軸、KPI、検証方法を提案してください。",
    "歴史学者": "あなたは一次資料を重視する歴史学者です。背景と因果関係をわかりやすく解説してください。",
    "料理研究家": "あなたは家庭で再現可能なレシピに長けた料理研究家です。分量・手順・コツを具体的に示してください。"
}

def ask_llm(input_text: str, expert_kind: str) -> str:
    llm = get_llm()
    msgs = [SystemMessage(content=SYSTEM_PROMPTS.get(expert_kind, "あなたは有能なアシスタントです。")),
            HumanMessage(content=input_text)]
    res = llm.invoke(msgs)
    return res.content

st.set_page_config(page_title="LLM 専門家チャット", page_icon="💬")
st.title("💬 LLM 専門家チャット（Streamlit + LangChain）")
st.markdown("1) 専門家を選ぶ → 2) 入力 → 3) 送信 で回答が出ます。")

expert = st.radio("専門家を選択：", list(SYSTEM_PROMPTS.keys()))
user_input = st.text_area("質問・お題：", height=140, placeholder="例）小規模チームのテスト戦略を提案して")

if st.button("送信"):
    if not user_input.strip():
        st.warning("入力テキストを入れてください。")
    else:
        with st.spinner("考え中..."):
            st.write(ask_llm(user_input.strip(), expert))


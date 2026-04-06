"""
app.py — 퀀트 에이전트 Web UI (Flask + SSE)
모드: auto(자동추천) | ticker(종목지정분석) | recommend(경제상황종목추천)
"""
from __future__ import annotations

import json
import os
import queue
import threading
from datetime import date
from typing import Any

from flask import Flask, Response, render_template, request, stream_with_context

import anthropic
from macro_agent import fetch_fred_data, classify_economic_phase, get_investment_strategy, get_full_recommendations
from price_agent import fetch_price_snapshot, snapshot_to_dict
from strategy_engine import build_trade_setup, setup_to_dict
from trading_agent import SYSTEM_PROMPT

app = Flask(__name__)
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────
# Tools 정의 (모드별 확장)
# ──────────────────────────────────────────────────

BASE_TOOLS: list[dict] = [
    {
        "name": "fetch_macro_data",
        "description": (
            "FRED API로 미국 거시경제 지표 4종을 조회합니다: "
            "T10Y2Y(장단기 금리차), UNRATE(실업률), VIXCLS(VIX 공포지수), DEXKOUS(원달러 환율). "
            "경제 국면(확장기/수축기/침체기)을 판별하고 기본 종목을 결정합니다. "
            "항상 가장 먼저 호출하세요."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "fetch_price_data",
        "description": (
            "yfinance로 종목의 일봉 OHLCV 데이터를 가져와 지지선·저항선을 탐지합니다. "
            "ticker 파라미터는 yfinance 형식 사용 (예: TQQQ, QQQ, AAPL, 005930.KS)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "분석할 종목 티커"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "evaluate_trade_setup",
        "description": (
            "메모 전략 수치화: 지지선 기반 손절가, 1차/2차 익절가, RRR, 분할 비율, 진입 신호. "
            "fetch_price_data 결과를 받은 후 호출하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "current_price": {"type": "number"},
                "nearest_support": {"type": "number"},
                "nearest_resistance": {"type": "number"},
                "trend": {"type": "string"},
                "distance_to_support_pct": {"type": "number"},
            },
            "required": ["ticker", "current_price", "nearest_support",
                         "nearest_resistance", "trend", "distance_to_support_pct"],
        },
    },
    {
        "name": "execute_mock_order",
        "description": (
            "모의 매수 주문 실행. entry_signal이 '강력 진입' 또는 '진입 고려'일 때만 호출. "
            "'대기'이면 호출하지 말 것."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "entry_signal": {"type": "string"},
                "split_ratio": {"type": "string"},
                "split_first_pct": {"type": "integer"},
                "stop_loss": {"type": "number"},
                "target_1": {"type": "number"},
                "target_2": {"type": "number"},
                "rrr": {"type": "number"},
            },
            "required": ["ticker", "entry_signal", "split_ratio", "split_first_pct",
                         "stop_loss", "target_1", "target_2", "rrr"],
        },
    },
    {
        "name": "save_report",
        "description": "분석 결과를 마크다운 보고서로 저장. 모든 분석이 끝난 마지막 단계에서 호출.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "마크다운 형식 보고서 전문"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "get_stock_recommendations",
        "description": (
            "현재 경제 국면에 맞는 다중 종목 추천 리스트를 조회합니다. "
            "각 종목별 투자 근거, 위험도, 목표 수익률을 포함합니다. "
            "fetch_macro_data 이후 호출하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "economic_phase": {"type": "string", "description": "classify_economic_phase 결과값"},
                "vix": {"type": "number", "description": "현재 VIX 값 (옵션)"},
                "usdkrw": {"type": "number", "description": "현재 원달러 환율 (옵션)"},
            },
            "required": ["economic_phase"],
        },
    },
]


# ──────────────────────────────────────────────────
# Tool 실행
# ──────────────────────────────────────────────────

def execute_tool(name: str, inp: dict[str, Any], fred_key: str, q: queue.Queue) -> str:
    if name == "fetch_macro_data":
        q.put(("log", "FRED API에서 거시경제 지표 조회 중..."))
        t10y2y, unrrate, vix, usdkrw = fetch_fred_data(fred_key)
        phase = classify_economic_phase(t10y2y, unrrate)
        strategy = get_investment_strategy(phase)
        kill_switch = []
        if vix and vix > 30:
            kill_switch.append(f"VIX {vix:.1f} > 30 → 극단적 공포장 (매수 중단)")
        if usdkrw and usdkrw > 1400:
            kill_switch.append(f"원달러 {usdkrw:.0f}원 > 1,400 → 환차손 경고")
        result = {
            "t10y2y": t10y2y, "unrrate": unrrate,
            "vix": vix, "usdkrw": usdkrw,
            "economic_phase": phase,
            "recommended_ticker": strategy["ticker"],
            "recommended_splits": strategy["splits"],
            "target_return": strategy["target_return"],
            "kill_switch_alerts": kill_switch,
        }
        q.put(("macro", result))
        return json.dumps(result, ensure_ascii=False, indent=2)

    if name == "fetch_price_data":
        ticker = inp["ticker"]
        q.put(("log", f"{ticker} 가격 데이터 & 지지/저항선 분석 중..."))
        snap = fetch_price_snapshot(ticker)
        d = snapshot_to_dict(snap)
        q.put(("price", d))
        return json.dumps(d, ensure_ascii=False, indent=2)

    if name == "evaluate_trade_setup":
        q.put(("log", f"{inp['ticker']} 매매 전략 수치화 중 (RRR / 분할 / 손절)..."))
        setup = build_trade_setup(
            ticker=inp["ticker"],
            current_price=inp["current_price"],
            nearest_support=inp.get("nearest_support"),
            nearest_resistance=inp.get("nearest_resistance"),
            trend=inp.get("trend", "횡보"),
            distance_to_support_pct=inp.get("distance_to_support_pct", 5.0),
        )
        d = setup_to_dict(setup)
        q.put(("setup", d))
        return json.dumps(d, ensure_ascii=False, indent=2)

    if name == "execute_mock_order":
        signal = inp.get("entry_signal", "대기")
        if signal == "대기":
            msg = "RRR 미충족 — 주문 미실행"
            q.put(("order", {"status": "skipped", "message": msg}))
            return json.dumps({"status": "skipped", "message": msg}, ensure_ascii=False)
        result_msg = (
            f"{inp['ticker']} | {signal} | 분할 {inp['split_ratio']} "
            f"(1차 {inp['split_first_pct']}%) | 손절 {inp['stop_loss']} / "
            f"1차익절 {inp['target_1']} / 2차익절 {inp['target_2']} | RRR 1:{inp['rrr']}"
        )
        q.put(("order", {"status": "executed", "message": result_msg, "data": inp}))
        return json.dumps({"status": "executed", "message": result_msg}, ensure_ascii=False)

    if name == "save_report":
        content = inp.get("content", "")
        filename = f"{date.today().isoformat()}_투자보고서.md"
        filepath = os.path.join(REPORT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        q.put(("report", {"filepath": filepath, "content": content}))
        return json.dumps({"status": "saved", "filepath": filepath}, ensure_ascii=False)

    if name == "get_stock_recommendations":
        phase = inp.get("economic_phase", "")
        vix = inp.get("vix")
        usdkrw = inp.get("usdkrw")
        recs = get_full_recommendations(phase)
        # Kill switch 적용: VIX>30이면 고위험 종목 경고 추가
        if vix and vix > 30:
            for r in recs:
                if r["risk"] == "고위험":
                    r["warning"] = f"VIX {vix:.1f} > 30 — 고위험 종목 진입 주의"
        if usdkrw and usdkrw > 1400:
            for r in recs:
                r["fx_warning"] = f"원달러 {usdkrw:.0f}원 > 1,400 — 환헤지 고려"
        result = {"economic_phase": phase, "recommendations": recs}
        q.put(("recommendations", result))
        return json.dumps(result, ensure_ascii=False, indent=2)

    return json.dumps({"error": f"알 수 없는 도구: {name}"})


# ──────────────────────────────────────────────────
# 에이전트 실행
# ──────────────────────────────────────────────────

def _build_user_prompt(mode: str, tickers: list[str]) -> str:
    if mode == "ticker" and tickers:
        tlist = ", ".join(tickers)
        return (
            f"다음 종목들을 분석해주세요: {tlist}\n"
            "먼저 거시 지표를 조회하고, 각 종목에 대해 순서대로 가격 분석과 "
            "메모 전략(RRR, 분할, 손절/익절)을 적용하여 투자 판단을 내려주세요."
        )
    if mode == "recommend":
        return (
            "현재 거시경제 지표를 조회하고, 경제 국면에 맞는 종목 추천 리스트를 가져와서 "
            "각 종목별 투자 근거와 우선순위를 상세히 설명해주세요. "
            "최종적으로 지금 가장 주목해야 할 TOP 3 종목과 그 이유를 명확히 제시하세요."
        )
    # auto (기본)
    return (
        "오늘의 거시 지표를 조회하고, 경제 국면에 맞는 종목의 "
        "지지선·저항선을 분석한 뒤, 메모 전략(RRR, 분할, 손절/익절)을 적용하여 "
        "오늘의 최종 투자 판단을 내려주세요."
    )


def run_agent(fred_key: str, anthropic_key: str, q: queue.Queue,
              mode: str = "auto", tickers: list[str] | None = None):
    try:
        client = anthropic.Anthropic(api_key=anthropic_key)
        messages: list[dict] = [
            {"role": "user", "content": _build_user_prompt(mode, tickers or [])}
        ]

        for step in range(15):
            q.put(("log", f"[Step {step + 1}] Claude에게 요청 중..."))
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=BASE_TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})

            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    q.put(("think", block.text.strip()))

            if resp.stop_reason == "end_turn":
                texts = [b.text for b in resp.content if b.type == "text"]
                q.put(("final", "\n".join(texts)))
                break

            if resp.stop_reason == "tool_use":
                results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        q.put(("tool_call", block.name))
                        tool_result = execute_tool(block.name, block.input, fred_key, q)
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_result,
                        })
                messages.append({"role": "user", "content": results})

        q.put(("done", None))
    except Exception as e:
        q.put(("error", str(e)))
        q.put(("done", None))


# ──────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["GET"])
def run_stream():
    fred_key      = request.args.get("fred_key", "")
    anthropic_key = request.args.get("anthropic_key", "")
    mode          = request.args.get("mode", "auto")          # auto | ticker | recommend
    tickers_raw   = request.args.get("tickers", "")           # "AAPL,NVDA,TSLA"

    if not fred_key or not anthropic_key:
        return Response("API 키가 누락되었습니다.", status=400)

    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    q: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=run_agent,
        args=(fred_key, anthropic_key, q, mode, tickers),
        daemon=True,
    )
    thread.start()

    def generate():
        while True:
            try:
                event_type, data = q.get(timeout=180)
                payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                if event_type == "done":
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'error', 'data': '타임아웃'})}\n\n"
                break

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)

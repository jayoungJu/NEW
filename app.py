"""
app.py — 퀀트 에이전트 Web UI (Flask + SSE)
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
from macro_agent import fetch_fred_data, classify_economic_phase, get_investment_strategy
from price_agent import fetch_price_snapshot, snapshot_to_dict
from strategy_engine import build_trade_setup, setup_to_dict
from trading_agent import TOOLS, SYSTEM_PROMPT

app = Flask(__name__)
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────
# Tool 실행 (스트리밍 큐 포함)
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
        q.put(("log", "매매 전략 수치화 중 (RRR / 분할 / 손절)..."))
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

    return json.dumps({"error": f"알 수 없는 도구: {name}"})


# ──────────────────────────────────────────────────
# 에이전트 실행 (백그라운드 스레드)
# ──────────────────────────────────────────────────

def run_agent(fred_key: str, anthropic_key: str, q: queue.Queue):
    try:
        client = anthropic.Anthropic(api_key=anthropic_key)
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    "오늘의 거시 지표를 조회하고, 경제 국면에 맞는 종목의 "
                    "지지선·저항선을 분석한 뒤, 메모 전략(RRR, 분할, 손절/익절)을 적용하여 "
                    "오늘의 최종 투자 판단을 내려주세요."
                ),
            }
        ]

        for step in range(10):
            q.put(("log", f"[Step {step + 1}] Claude에게 요청 중..."))
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
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
    fred_key = request.args.get("fred_key", "")
    anthropic_key = request.args.get("anthropic_key", "")

    if not fred_key or not anthropic_key:
        return Response("API 키가 누락되었습니다.", status=400)

    q: queue.Queue = queue.Queue()
    thread = threading.Thread(target=run_agent, args=(fred_key, anthropic_key, q), daemon=True)
    thread.start()

    def generate():
        while True:
            try:
                event_type, data = q.get(timeout=120)
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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)

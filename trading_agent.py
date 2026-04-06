"""
trading_agent.py
Claude Tool Use 기반 주식 투자 에이전트 (메모 전략 통합판)

실행 흐름:
  1. fetch_macro_data    — FRED 거시 지표 (금리차, 실업률, VIX, 환율)
  2. fetch_price_data    — TradingView 방식 지지/저항선 탐지 (yfinance)
  3. evaluate_trade_setup — RRR / 손절 / 분할 비율 자동 계산
  4. execute_mock_order  — 모의 주문 실행
  5. save_report         — 마크다운 보고서 저장
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

import anthropic

from macro_agent import fetch_fred_data, classify_economic_phase, get_investment_strategy
from price_agent import fetch_price_snapshot, snapshot_to_dict
from strategy_engine import build_trade_setup, setup_to_dict


# ────────────────────────────────────────────────
# Tool 정의
# ────────────────────────────────────────────────

TOOLS: list[dict] = [
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
            "yfinance로 종목의 일봉 OHLCV 데이터를 가져와 TradingView 방식으로 "
            "지지선·저항선을 탐지합니다. "
            "macro_data로 종목이 결정된 후 호출하세요. "
            "ticker 파라미터는 yfinance 형식 사용 (예: TQQQ, QQQ, 005930.KS)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "분석할 종목 티커 (예: TQQQ, QQQ, TMF)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "evaluate_trade_setup",
        "description": (
            "메모 전략을 수치로 변환합니다: "
            "① 지지선 기반 손절가, ② 1차/2차 익절가, ③ RRR(≥2.0 진입 원칙), "
            "④ 분할 비율(1:1/1:2/2:1) 자동 결정, ⑤ 진입 신호(강력진입/진입고려/대기). "
            "fetch_price_data 결과를 받은 후 호출하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "current_price": {"type": "number"},
                "nearest_support": {"type": "number", "description": "가장 가까운 지지선 가격"},
                "nearest_resistance": {"type": "number", "description": "가장 가까운 저항선 가격"},
                "trend": {"type": "string", "description": "상승|횡보|하락"},
                "distance_to_support_pct": {"type": "number", "description": "지지선까지 거리(%)"},
            },
            "required": [
                "ticker", "current_price", "nearest_support",
                "nearest_resistance", "trend", "distance_to_support_pct",
            ],
        },
    },
    {
        "name": "execute_mock_order",
        "description": (
            "모의 매수 주문을 실행합니다. "
            "entry_signal이 '강력 진입' 또는 '진입 고려'일 때만 호출하세요. "
            "'대기'이면 호출하지 말고 이유를 설명하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "entry_signal": {"type": "string", "description": "강력 진입|진입 고려|대기"},
                "split_ratio": {"type": "string", "description": "1:1|1:2|2:1"},
                "split_first_pct": {"type": "integer", "description": "1차 매수 비율(%)"},
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
        "description": (
            "오늘의 분석 결과를 마크다운 보고서로 저장합니다. "
            "모든 분석이 끝난 마지막 단계에서 반드시 호출하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "마크다운 형식 보고서 전문"},
            },
            "required": ["content"],
        },
    },
]


# ────────────────────────────────────────────────
# 시스템 프롬프트
# ────────────────────────────────────────────────

SYSTEM_PROMPT = """너는 다음 메모 전략을 정확히 따르는 퀀트 투자 에이전트다.

## 핵심 원칙 (메모에서 도출)

1. **지지선 기반 진입**: TradingView 일봉 기준 지지선 근처에서만 진입.
   현재가가 지지선에서 2% 이상 떨어져 있으면 눌림을 기다려라.

2. **RRR 1:2 이상 강제**: evaluate_trade_setup 결과에서 RRR < 2.0이면
   entry_signal이 '대기'로 나온다. 이 경우 execute_mock_order를 절대 호출하지 마라.

3. **분할 매매 (1:1 / 1:2 / 2:1)**: evaluate_trade_setup이 상황에 맞는 비율을 자동 결정.
   - 2:1: 지지선 근접 + 상승 추세 + RRR≥3 → 초기 집중
   - 1:1: 중간 확신 → 균등
   - 1:2: 지지선 멀거나 불확실 → 눌림목 대기

4. **손절 칼같이**: 지지선 -0.5% 또는 고정 -3~5% 중 유리한 쪽.
   설정된 손절가는 절대 임의로 늘리지 않는다.

5. **포지션 다이어트 (익절)**: 1차 목표 도달 시 50% 매도, 2차 목표에서 나머지 전량.

## Kill Switch 조건
- VIX > 30: 모든 매수 중단 → 관망(현금)
- 원달러 > 1,400: 분할 횟수 1.5배 증가 (보수적 접근)

## 실행 순서
fetch_macro_data → fetch_price_data → evaluate_trade_setup
→ (신호에 따라) execute_mock_order → save_report

추론 과정을 한국어로 단계별로 명확하게 서술하라."""


# ────────────────────────────────────────────────
# Tool 실행 핸들러
# ────────────────────────────────────────────────

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))


def _execute_tool(name: str, inp: dict[str, Any], fred_key: str) -> str:
    if name == "fetch_macro_data":
        t10y2y, unrrate, vix, usdkrw = fetch_fred_data(fred_key)
        phase = classify_economic_phase(t10y2y, unrrate)
        strategy = get_investment_strategy(phase)
        kill_switch = []
        if vix and vix > 30:
            kill_switch.append(f"VIX {vix:.1f} > 30 → 극단적 공포장 (매수 중단)")
        if usdkrw and usdkrw > 1400:
            kill_switch.append(f"원달러 {usdkrw:.0f}원 > 1,400 → 환차손 경고 (분할 1.5배 확대)")
        result = {
            "t10y2y": t10y2y, "unrrate": unrrate,
            "vix": vix, "usdkrw": usdkrw,
            "economic_phase": phase,
            "recommended_ticker": strategy["ticker"],
            "recommended_splits": strategy["splits"],
            "target_return": strategy["target_return"],
            "kill_switch_alerts": kill_switch,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    if name == "fetch_price_data":
        ticker = inp["ticker"]
        snap = fetch_price_snapshot(ticker)
        return json.dumps(snapshot_to_dict(snap), ensure_ascii=False, indent=2)

    if name == "evaluate_trade_setup":
        setup = build_trade_setup(
            ticker=inp["ticker"],
            current_price=inp["current_price"],
            nearest_support=inp.get("nearest_support"),
            nearest_resistance=inp.get("nearest_resistance"),
            trend=inp.get("trend", "횡보"),
            distance_to_support_pct=inp.get("distance_to_support_pct", 5.0),
        )
        return json.dumps(setup_to_dict(setup), ensure_ascii=False, indent=2)

    if name == "execute_mock_order":
        signal = inp.get("entry_signal", "대기")
        if signal == "대기":
            return json.dumps({"status": "skipped", "message": "RRR 미충족 또는 지지선 근접 아님 — 주문 미실행"}, ensure_ascii=False)
        ticker = inp["ticker"]
        split = inp["split_ratio"]
        first_pct = inp["split_first_pct"]
        result_msg = (
            f"[모의주문] {ticker} | {signal} | "
            f"분할 {split} (1차 {first_pct}%) | "
            f"손절 {inp['stop_loss']} / 1차익절 {inp['target_1']} / 2차익절 {inp['target_2']} | "
            f"RRR 1:{inp['rrr']}"
        )
        print(f"\n  ✅ {result_msg}")
        return json.dumps({"status": "executed", "message": result_msg}, ensure_ascii=False)

    if name == "save_report":
        content = inp.get("content", "")
        filename = f"{date.today().isoformat()}_투자보고서.md"
        filepath = os.path.join(REPORT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n  📄 보고서 저장: {filepath}")
        return json.dumps({"status": "saved", "filepath": filepath}, ensure_ascii=False)

    return json.dumps({"error": f"알 수 없는 도구: {name}"})


# ────────────────────────────────────────────────
# 에이전트 실행
# ────────────────────────────────────────────────

def run_trading_agent(fred_api_key: str) -> str:
    client = anthropic.Anthropic()
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
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        # 중간 추론 출력
        for block in resp.content:
            if block.type == "text" and block.text.strip():
                print(f"\n  [Step {step + 1}] {block.text.strip()[:300]}{'...' if len(block.text) > 300 else ''}")

        if resp.stop_reason == "end_turn":
            texts = [b.text for b in resp.content if b.type == "text"]
            return "\n".join(texts)

        if resp.stop_reason == "tool_use":
            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    print(f"\n  🔧 [{block.name}] 호출 중...")
                    tool_result = _execute_tool(block.name, block.input, fred_api_key)
                    # 결과 미리보기 (처음 200자)
                    preview = tool_result[:200].replace("\n", " ")
                    print(f"     → {preview}{'...' if len(tool_result) > 200 else ''}")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result,
                    })
            messages.append({"role": "user", "content": results})

    return "[최대 반복 횟수 도달]"


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────

def main():
    fred_key = os.environ.get("FRED_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not fred_key:
        print("❌  FRED_API_KEY 환경변수를 설정하세요.")
        print("    발급: https://fred.stlouisfed.org/docs/api/api_key.html")
        return
    if not anthropic_key:
        print("❌  ANTHROPIC_API_KEY 환경변수를 설정하세요.")
        return

    print()
    print("=" * 62)
    print("   📈  메모 전략 통합 퀀트 에이전트  (Claude Sonnet)")
    print("=" * 62)
    print("   전략 원칙: 지지선 진입 | RRR ≥ 1:2 | 분할 매매 | 칼손절")
    print("=" * 62)

    final = run_trading_agent(fred_key)

    print()
    print("=" * 62)
    print("  📋  최종 투자 판단")
    print("=" * 62)
    print(final)
    print("=" * 62)


if __name__ == "__main__":
    main()

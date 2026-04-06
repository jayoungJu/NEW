"""
strategy_engine.py
메모 전략 핵심 로직 모듈

- RRR(리스크 대비 보상 비율) 계산
- 분할 매매 비율 자동 결정 (1:1 / 1:2 / 2:1)
- 손절가 / 익절가 자동 산출
- 지지선 기반 진입 적정성 판단
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ────────────────────────────────────────────────
# 데이터 클래스
# ────────────────────────────────────────────────

@dataclass
class TradeSetup:
    """한 번의 진입에 필요한 모든 수치"""
    ticker: str
    entry_price: float
    stop_loss: float          # 손절가
    target_1: float           # 1차 익절가 (50% 매도)
    target_2: float           # 2차 익절가 (나머지 전량)
    rrr: float                # 리스크 대비 보상 비율 (목표1 기준)
    split_ratio: str          # "1:1" | "1:2" | "2:1"
    split_first_pct: int      # 1차 매수 비율(%)
    split_second_pct: int     # 2차 매수 비율(%)
    entry_signal: str         # "강력 진입" | "진입 고려" | "대기"
    reason: str               # 판단 근거 설명
    risk_pct: float           # 진입가 대비 손실 위험(%)
    reward_pct: float         # 진입가 대비 수익 기대(%)


# ────────────────────────────────────────────────
# 핵심 계산 함수
# ────────────────────────────────────────────────

def calculate_rrr(entry: float, stop: float, target: float) -> float:
    """
    RRR = (목표가 - 진입가) / (진입가 - 손절가)
    메모 원칙: 최소 1:2 이상이어야 진입
    """
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk == 0:
        return 0.0
    return round(reward / risk, 2)


def determine_split_ratio(
    distance_to_support_pct: float,
    trend: str,
    rrr: float,
) -> tuple[str, int, int]:
    """
    메모의 3가지 분할 전략 자동 선택:

    2:1 (초기 집중) — 강한 모멘텀, 지지선에 매우 근접, RRR이 높을 때
    1:1 (균등)     — 방향성 불명확, 중간 상황
    1:2 (눌림목)   — 지지선에서 멀거나 변동성 높을 때, 추가 눌림 대기

    반환: (비율 문자열, 1차%, 2차%)
    """
    at_support = distance_to_support_pct <= 1.5   # 지지선 1.5% 이내
    strong_trend = trend == "상승"
    high_rrr = rrr >= 3.0

    if at_support and strong_trend and high_rrr:
        return "2:1", 67, 33   # 확신 높음 → 초기 집중
    elif at_support or (strong_trend and rrr >= 2.0):
        return "1:1", 50, 50   # 중간 확신 → 균등
    else:
        return "1:2", 33, 67   # 눌림목 추가 대기


def build_trade_setup(
    ticker: str,
    current_price: float,
    nearest_support: Optional[float],
    nearest_resistance: Optional[float],
    trend: str,
    distance_to_support_pct: float,
    stop_loss_pct: float = 3.0,    # 메모 기준: -3~5% 손절
    target_multiplier: float = 2.0,  # RRR 최소 1:2 → 수익폭 = 손실폭 × 2
) -> TradeSetup:
    """
    메모 전략을 수치로 변환해 TradeSetup을 생성합니다.

    손절 우선순위:
      1. 지지선 기반 손절 (지지선 -0.5%)
      2. 고정 비율 손절 (-stop_loss_pct%)
      → 둘 중 더 유리한(현재가에 가까운) 값 사용
    """
    # ── 손절가 계산 ──
    fixed_stop = round(current_price * (1 - stop_loss_pct / 100), 2)
    if nearest_support and nearest_support < current_price:
        support_stop = round(nearest_support * 0.995, 2)  # 지지선 -0.5%
        stop_loss = max(fixed_stop, support_stop)          # 더 가까운(높은) 손절 채택
    else:
        stop_loss = fixed_stop

    # ── 익절가 계산 ──
    risk_amount = current_price - stop_loss
    target_1 = round(current_price + risk_amount * target_multiplier, 2)     # 1:2 기준
    target_2 = round(current_price + risk_amount * target_multiplier * 1.5, 2)  # 1:3 기준

    # 저항선이 목표 1보다 낮으면 저항선을 1차 목표로 조정
    if nearest_resistance and nearest_resistance < target_1:
        target_1 = round(nearest_resistance * 0.995, 2)

    rrr = calculate_rrr(current_price, stop_loss, target_1)

    # ── 분할 비율 결정 ──
    split_ratio, split_first, split_second = determine_split_ratio(
        distance_to_support_pct, trend, rrr
    )

    # ── 진입 신호 판단 ──
    risk_pct = round((current_price - stop_loss) / current_price * 100, 2)
    reward_pct = round((target_1 - current_price) / current_price * 100, 2)

    if rrr >= 2.0 and distance_to_support_pct <= 2.0 and trend in ("상승", "횡보"):
        entry_signal = "강력 진입"
        reason = (
            f"RRR {rrr} (≥2.0 충족) + "
            f"지지선 근접 ({distance_to_support_pct}% 이내) + "
            f"추세 {trend} → 진입 조건 모두 충족"
        )
    elif rrr >= 2.0:
        entry_signal = "진입 고려"
        reason = (
            f"RRR {rrr} 충족. "
            f"단, 지지선까지 {distance_to_support_pct}% 거리 있음. "
            f"추세 {trend} — 1차만 소량 진입 후 눌림 대기 권장"
        )
    else:
        entry_signal = "대기"
        reason = (
            f"RRR {rrr} (< 2.0 미충족) — 메모 원칙에 따라 진입 보류. "
            f"지지선({nearest_support}) 근처로 가격 하락 시 재검토"
        )

    return TradeSetup(
        ticker=ticker,
        entry_price=current_price,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        rrr=rrr,
        split_ratio=split_ratio,
        split_first_pct=split_first,
        split_second_pct=split_second,
        entry_signal=entry_signal,
        reason=reason,
        risk_pct=risk_pct,
        reward_pct=reward_pct,
    )


def setup_to_dict(setup: TradeSetup) -> dict:
    """TradeSetup → JSON 직렬화 가능한 dict"""
    return {
        "ticker": setup.ticker,
        "entry_price": setup.entry_price,
        "stop_loss": setup.stop_loss,
        "target_1": setup.target_1,
        "target_2": setup.target_2,
        "rrr": setup.rrr,
        "split_ratio": setup.split_ratio,
        "split_first_pct": setup.split_first_pct,
        "split_second_pct": setup.split_second_pct,
        "entry_signal": setup.entry_signal,
        "risk_pct": setup.risk_pct,
        "reward_pct": setup.reward_pct,
        "reason": setup.reason,
        "summary": (
            f"[{setup.entry_signal}] {setup.ticker} | "
            f"진입 {setup.entry_price} → 손절 {setup.stop_loss} / "
            f"1차익절 {setup.target_1} / 2차익절 {setup.target_2} | "
            f"RRR 1:{setup.rrr} | 분할 {setup.split_ratio} "
            f"({setup.split_first_pct}% + {setup.split_second_pct}%)"
        ),
    }


# ────────────────────────────────────────────────
# 단독 실행 테스트
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # 예시: 현재가 240, 지지선 236 (메모 기준), 저항선 255
    setup = build_trade_setup(
        ticker="QQQ",
        current_price=240.0,
        nearest_support=236.0,
        nearest_resistance=255.0,
        trend="상승",
        distance_to_support_pct=1.67,
    )
    print(json.dumps(setup_to_dict(setup), ensure_ascii=False, indent=2))

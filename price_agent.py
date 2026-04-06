"""
price_agent.py
TradingView 일봉 기준 지지/저항선 탐지 모듈

yfinance로 OHLCV 데이터를 내려받고,
피벗 포인트(로컬 저점/고점)를 자동 탐지합니다.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import yfinance as yf
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ────────────────────────────────────────────────
# 데이터 클래스
# ────────────────────────────────────────────────

@dataclass
class PriceSnapshot:
    """종목의 현재 가격 + 핵심 지지/저항 정보"""
    ticker: str
    current_price: float
    support_levels: list[float]          # 가까운 지지선 (오름차순)
    resistance_levels: list[float]       # 가까운 저항선 (오름차순)
    nearest_support: Optional[float]     # 현재가 바로 아래 지지선
    nearest_resistance: Optional[float]  # 현재가 바로 위 저항선
    daily_range_pct: float               # 최근 5일 평균 일중 변동폭(%)
    trend: str                           # "상승" | "횡보" | "하락"
    distance_to_support_pct: float       # 현재가 → 지지선 거리(%)
    raw_data: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


# ────────────────────────────────────────────────
# 핵심 함수
# ────────────────────────────────────────────────

def _detect_pivots(series: pd.Series, window: int = 5) -> tuple[list[float], list[float]]:
    """
    로컬 저점(지지)과 고점(저항)을 탐지합니다.
    window: 좌우 n개 캔들보다 낮은/높은 값을 피벗으로 인식
    """
    lows, highs = [], []
    arr = series.values
    for i in range(window, len(arr) - window):
        window_slice = arr[i - window: i + window + 1]
        if arr[i] == window_slice.min():
            lows.append(float(arr[i]))
        if arr[i] == window_slice.max():
            highs.append(float(arr[i]))
    return lows, highs


def _cluster_levels(levels: list[float], tolerance_pct: float = 0.5) -> list[float]:
    """
    비슷한 가격대의 지지/저항선을 클러스터링하여 중복 제거.
    tolerance_pct: 이 퍼센트 이내면 같은 레벨로 묶음
    """
    if not levels:
        return []
    sorted_lvls = sorted(set(levels))
    clusters: list[list[float]] = [[sorted_lvls[0]]]
    for lvl in sorted_lvls[1:]:
        if abs(lvl - clusters[-1][-1]) / clusters[-1][-1] * 100 <= tolerance_pct:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])
    return [round(np.mean(c), 2) for c in clusters]


def _determine_trend(close: pd.Series, period: int = 20) -> str:
    """20일 이동평균과 현재가 비교로 추세 판단"""
    if len(close) < period:
        return "횡보"
    ma = close.rolling(period).mean().iloc[-1]
    current = close.iloc[-1]
    if current > ma * 1.02:
        return "상승"
    if current < ma * 0.98:
        return "하락"
    return "횡보"


def fetch_price_snapshot(
    ticker: str,
    period: str = "6mo",
    pivot_window: int = 5,
    max_levels: int = 3,
) -> PriceSnapshot:
    """
    종목 티커를 받아 PriceSnapshot을 반환합니다.

    Args:
        ticker    : yfinance 티커 (예: "QQQ", "TQQQ", "005930.KS")
        period    : 데이터 기간 (기본 6개월)
        pivot_window : 피벗 탐지 좌우 캔들 수
        max_levels   : 반환할 지지/저항 레벨 최대 개수
    """
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"'{ticker}' 데이터를 가져올 수 없습니다.")

    # 컬럼이 MultiIndex인 경우 평탄화
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    low = df["Low"]
    high = df["High"]

    current_price = float(close.iloc[-1])

    # 지지/저항 탐지
    raw_lows, raw_highs = _detect_pivots(low, window=pivot_window)
    support_all = _cluster_levels(raw_lows)
    resistance_all = _cluster_levels(raw_highs)

    # 현재가 기준 필터링
    supports = sorted([s for s in support_all if s < current_price * 0.999], reverse=True)[:max_levels]
    supports = sorted(supports)
    resistances = sorted([r for r in resistance_all if r > current_price * 1.001])[:max_levels]

    nearest_support = supports[-1] if supports else None
    nearest_resistance = resistances[0] if resistances else None

    # 지지선까지 거리
    dist_pct = 0.0
    if nearest_support:
        dist_pct = round((current_price - nearest_support) / current_price * 100, 2)

    # 일중 변동폭
    recent_range = ((high - low) / close * 100).iloc[-5:].mean()
    daily_range_pct = round(float(recent_range), 2)

    trend = _determine_trend(close)

    return PriceSnapshot(
        ticker=ticker,
        current_price=round(current_price, 2),
        support_levels=supports,
        resistance_levels=resistances,
        nearest_support=round(nearest_support, 2) if nearest_support else None,
        nearest_resistance=round(nearest_resistance, 2) if nearest_resistance else None,
        daily_range_pct=daily_range_pct,
        trend=trend,
        distance_to_support_pct=dist_pct,
        raw_data=df,
    )


def snapshot_to_dict(snap: PriceSnapshot) -> dict:
    """PriceSnapshot을 JSON 직렬화 가능한 dict로 변환"""
    return {
        "ticker": snap.ticker,
        "current_price": snap.current_price,
        "trend": snap.trend,
        "support_levels": snap.support_levels,
        "resistance_levels": snap.resistance_levels,
        "nearest_support": snap.nearest_support,
        "nearest_resistance": snap.nearest_resistance,
        "distance_to_support_pct": snap.distance_to_support_pct,
        "daily_range_pct": snap.daily_range_pct,
        "description": (
            f"현재가 {snap.current_price}는 가장 가까운 지지선 {snap.nearest_support}에서 "
            f"{snap.distance_to_support_pct}% 위에 위치. 추세: {snap.trend}. "
            f"최근 일중 변동폭: ±{snap.daily_range_pct}%"
        ),
    }


# ────────────────────────────────────────────────
# 단독 실행 테스트
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    for t in ["QQQ", "TQQQ"]:
        print(f"\n{'='*50}")
        snap = fetch_price_snapshot(t)
        print(json.dumps(snapshot_to_dict(snap), ensure_ascii=False, indent=2))

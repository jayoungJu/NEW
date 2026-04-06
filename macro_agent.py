"""
macro_agent.py — trading_agent 프로젝트용 복사본
(원본 파일에서 vix/usdkrw 반환 포함하도록 유지)
"""

from fredapi import Fred
from typing import Optional, Tuple, Dict, Any


def get_investment_strategy(phase: str) -> Dict[str, Any]:
    if "확장기" in phase:
        return {"ticker": "TQQQ", "splits": "40분할", "target_return": "+10%"}
    if "수축기" in phase:
        return {"ticker": "QQQ", "splits": "60분할", "target_return": "+5%"}
    if "침체기" in phase or "저점" in phase:
        return {"ticker": "TMF", "splits": "40분할", "target_return": "+10%"}
    return {"ticker": "관망(현금)", "splits": "0분할", "target_return": "0%"}


def _latest_value(series) -> Optional[float]:
    cleaned = series.dropna()
    return float(cleaned.iloc[-1]) if len(cleaned) > 0 else None


def fetch_fred_data(api_key: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    fred = Fred(api_key=api_key)
    try:
        t10y2y = _latest_value(fred.get_series("T10Y2Y"))
        unrrate = _latest_value(fred.get_series("UNRATE"))
        vix     = _latest_value(fred.get_series("VIXCLS"))
        usdkrw  = _latest_value(fred.get_series("DEXKOUS"))
    except Exception as e:
        print(f"FRED API 오류: {e}")
        return None, None, None, None
    return t10y2y, unrrate, vix, usdkrw


def classify_economic_phase(t10y2y: Optional[float], unrrate: Optional[float]) -> str:
    if t10y2y is None or unrrate is None:
        return "판별 불가(데이터 부족)"
    if t10y2y < 0:
        return "수축기(침체 경고)"
    elif unrrate > 4.5:
        return "침체기/저점"
    else:
        return "확장기"

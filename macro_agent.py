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


# 경제 국면별 다중 종목 추천 테이블
_RECOMMENDATIONS: Dict[str, list] = {
    "확장기": [
        {"ticker": "TQQQ", "name": "ProShares UltraPro QQQ", "type": "레버리지ETF", "risk": "고위험",
         "reason": "확장기 나스닥 3배 레버리지. 금리차 양전환+저실업률 구간 최적 수익", "target": "+15~30%", "splits": "40분할"},
        {"ticker": "QQQ",  "name": "Invesco QQQ Trust",      "type": "ETF",      "risk": "중위험",
         "reason": "나스닥100 추적. 성장주 중심 확장기 안정적 수익", "target": "+8~15%", "splits": "30분할"},
        {"ticker": "NVDA", "name": "NVIDIA",                 "type": "개별주",   "risk": "고위험",
         "reason": "AI·데이터센터 수혜주. 확장기 성장 모멘텀 최강", "target": "+20~40%", "splits": "50분할"},
        {"ticker": "MSFT", "name": "Microsoft",              "type": "개별주",   "risk": "중위험",
         "reason": "클라우드·AI 복합 성장. 확장기 방어적 성장주", "target": "+10~20%", "splits": "30분할"},
        {"ticker": "SOXL", "name": "Direxion Semi Bull 3x",  "type": "레버리지ETF", "risk": "고위험",
         "reason": "반도체 3배 레버리지. AI 사이클 확장기 공격적 베팅", "target": "+20~50%", "splits": "60분할"},
    ],
    "수축기": [
        {"ticker": "QQQ",  "name": "Invesco QQQ Trust",      "type": "ETF",      "risk": "중위험",
         "reason": "레버리지 축소, 수축기 방어적 나스닥 추적", "target": "+5~10%", "splits": "60분할"},
        {"ticker": "SPY",  "name": "SPDR S&P 500 ETF",       "type": "ETF",      "risk": "중저위험",
         "reason": "분산된 S&P500. 수축기 안정적 포지션", "target": "+3~8%",  "splits": "50분할"},
        {"ticker": "TLT",  "name": "iShares 20Y Treasury",   "type": "채권ETF",  "risk": "중위험",
         "reason": "장기국채. 수축기 금리 하락 기대 수혜", "target": "+5~12%", "splits": "40분할"},
        {"ticker": "GLD",  "name": "SPDR Gold Shares",       "type": "원자재ETF","risk": "중위험",
         "reason": "금. 수축기 안전자산 수요 증가", "target": "+5~10%", "splits": "40분할"},
        {"ticker": "AAPL", "name": "Apple",                  "type": "개별주",   "risk": "중위험",
         "reason": "방어적 빅테크. 현금흐름 우수, 수축기 방어력 강함", "target": "+5~15%", "splits": "40분할"},
    ],
    "침체기/저점": [
        {"ticker": "TMF",  "name": "Direxion 20Y Bull 3x",   "type": "레버리지ETF", "risk": "고위험",
         "reason": "장기국채 3배 레버리지. 침체기 금리 급락 최대 수혜", "target": "+20~50%", "splits": "40분할"},
        {"ticker": "TLT",  "name": "iShares 20Y Treasury",   "type": "채권ETF",  "risk": "중위험",
         "reason": "침체기 금리 인하 사이클 수혜 안전판", "target": "+10~25%", "splits": "30분할"},
        {"ticker": "GLD",  "name": "SPDR Gold Shares",       "type": "원자재ETF","risk": "중위험",
         "reason": "침체·불확실성 구간 금 수요 급증", "target": "+10~20%", "splits": "30분할"},
        {"ticker": "BIL",  "name": "SPDR 1-3M T-Bill ETF",   "type": "단기채권", "risk": "저위험",
         "reason": "초단기 국채. 침체 불확실성 최고조 시 현금성 자산", "target": "+2~5%",  "splits": "현금유지"},
        {"ticker": "SQQQ", "name": "ProShares UltraPro Short QQQ", "type": "인버스ETF", "risk": "고위험",
         "reason": "나스닥 3배 인버스. 침체기 하락 베팅 (단기 헤지)", "target": "+15~40%", "splits": "60분할"},
    ],
    "판별 불가": [
        {"ticker": "BIL",  "name": "SPDR 1-3M T-Bill ETF",   "type": "단기채권", "risk": "저위험",
         "reason": "데이터 부족 시 현금성 자산 대기", "target": "+2~5%", "splits": "현금유지"},
        {"ticker": "GLD",  "name": "SPDR Gold Shares",       "type": "원자재ETF","risk": "중위험",
         "reason": "불확실성 구간 안전자산 헤지", "target": "+5~10%", "splits": "30분할"},
    ],
}


def get_full_recommendations(phase: str) -> list:
    """경제 국면에 맞는 다중 종목 추천 리스트 반환"""
    for key in _RECOMMENDATIONS:
        if key in phase:
            return _RECOMMENDATIONS[key]
    return _RECOMMENDATIONS["판별 불가"]


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

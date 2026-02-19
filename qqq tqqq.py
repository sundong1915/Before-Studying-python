import yfinance as yf
from fredapi import Fred
import pandas as pd
import numpy as np
import itertools
import warnings
import time

warnings.filterwarnings("ignore")

# ==========================================
# 0. 설정 (FRED API 키 입력 필수)
# ==========================================

FRED_API_KEY = 'b00d0e909d7e2e38815b8fbf62461695'
# ==========================================
# 1. 데이터 수집 (QQQ + FRED 하이일드 결합)
# ==========================================
def get_combined_data():
    print("⏳ 데이터 수집 중... (QQQ + 하이일드 스프레드)")
    
    # 1. 주식 데이터 (QQQ, SHY)
    tickers = ['QQQ', 'SHY']
    df = yf.download(tickers, start="2006-01-01", progress=False)
    
    # 멀티인덱스 컬럼 처리
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df['Close']
        except:
            df.columns = df.columns.get_level_values(0)
    
    # 2. FRED 데이터 (하이일드 스프레드)
    try:
        fred = Fred(api_key=FRED_API_KEY)
        # BAMLH0A0HYM2: 하이일드 스프레드
        spread = fred.get_series('BAMLH0A0HYM2', observation_start="2006-01-01")
        spread.name = 'HighYield_Spread'
        
        # 인덱스 시간대 제거 (YFinance와 병합 위해)
        df.index = df.index.tz_localize(None)
        spread.index = spread.index.tz_localize(None)
        
        # 데이터 병합 (주식 거래일 기준)
        df = df.join(spread, how='inner')
        
        # 결측치 채우기 (휴일 등으로 빈 FRED 데이터는 전일 데이터로 채움)
        df['HighYield_Spread'] = df['HighYield_Spread'].fillna(method='ffill')
        
    except Exception as e:
        print(f"⚠️ FRED 데이터 로드 실패: {e}")
        print("API 키를 확인하거나 인터넷 연결을 확인하세요.")
        return pd.DataFrame() # 빈 데이터프레임 반환으로 중단

    df = df.dropna()

    # [데이터 가공]
    # 1. 변동률 계산
    df['QQQ_Pct'] = df['QQQ'].pct_change()
    df['Sim_TQQQ_3X'] = df['QQQ_Pct'] * 3.0  # TQQQ 시뮬레이션
    df['Sim_Cash'] = df['SHY'].pct_change()  # 현금(단기채)

    # 2. RSI 계산 (QQQ 1배수 기준)
    delta = df['QQQ'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ==========================================
    # [핵심] FRED 중기 필터 로직 생성
    # ==========================================
    # 스프레드 20일 이동평균
    df['Spread_MA20'] = df['HighYield_Spread'].rolling(window=20).mean()
    
    # Look-ahead Bias 방지: 어제까지의 지표로 오늘 매매
    # shift(1)을 하여 '어제 스프레드'와 '어제 MA'를 비교
    prev_spread = df['HighYield_Spread'].shift(1)
    prev_ma20 = df['Spread_MA20'].shift(1)

    # 필터 조건: 스프레드가 이평선보다 5% 이상 튀어 오르면 '위험(Risk Off)'
    # (스프레드 > MA20 * 1.05) -> True면 매매 금지
    df['Macro_Risk_Off'] = prev_spread > (prev_ma20 * 1.05)

    return df.dropna()

# ==========================================
# 2. 백테스팅 엔진 (FRED 필터 + 피라미딩)
# ==========================================
def run_tqqq_strategy(df, ma_period, rsi_limit, sell_buffer):
    # QQQ 이동평균선
    ma = df['QQQ'].ewm(span=ma_period, adjust=False).mean()
    
    price_arr = df['QQQ'].values
    ma_arr = ma.values
    rsi_arr = df['RSI'].values
    macro_risk_arr = df['Macro_Risk_Off'].values # FRED 필터 배열
    
    position_size = np.zeros(len(df))
    current_pos = 0.0
    
    for i in range(1, len(df)):
        # 1. [거시 경제 필터] FRED가 위험 신호를 보내면 무조건 현금화
        if macro_risk_arr[i] == True:
            current_pos = 0.0
        
        # 2. [기술적 필터] 거시 경제가 안전할 때만 차트 분석
        else:
            price = price_arr[i]
            ma_val = ma_arr[i]
            sell_threshold = ma_val * (1 - sell_buffer)
            
            # 매도 로직 (기술적 손절)
            if price < sell_threshold:
                current_pos = 0.0
            else:
                # 매수 및 불타기 로직
                if price > ma_val:
                    if current_pos == 0.0: current_pos = 0.3      # 정찰병
                    elif current_pos == 0.3: current_pos = 0.7    # 불타기
                    elif current_pos == 0.7:
                        if rsi_arr[i] < rsi_limit: current_pos = 1.0 # 풀매수
                        else: current_pos = 0.7 
                    elif current_pos == 1.0:
                        if rsi_arr[i] > rsi_limit: current_pos = 0.7 # 과열 시 축소
                        else: current_pos = 1.0
                else:
                    pass # 버퍼존 유지

        position_size[i] = current_pos

    # 수익률 계산
    pos = pd.Series(position_size, index=df.index).shift(1).fillna(0)
    strategy_ret = (df['Sim_TQQQ_3X'] * pos) + (df['Sim_Cash'] * (1 - pos))
    df['Strategy_Ret'] = strategy_ret.fillna(0)
    
    # 전략 포지션 저장 (분석용)
    df['Strategy_Pos'] = position_size
    
    return (1 + strategy_ret).prod(), df

# ==========================================
# 3. 결과 분석
# ==========================================
def analyze_today(df, ma_period, rsi_limit, sell_buffer, total_score):
    last_row = df.iloc[-1]
    
    # 현재 상태 데이터
    price = last_row['QQQ']
    ma_val = df['QQQ'].ewm(span=ma_period, adjust=False).mean().iloc[-1]
    
    # FRED 상태
    current_spread = last_row['HighYield_Spread']
    spread_ma = last_row['Spread_MA20']
    is_macro_risk = last_row['Macro_Risk_Off']
    
    # 성과 분석
    start_date = df.index[0]
    end_date = df.index[-1]
    years = (end_date - start_date).days / 365.25
    cagr = (total_score ** (1 / years)) - 1
    cum_ret = (1 + df['Strategy_Ret']).cumprod()
    mdd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
    
    print("\n" + "="*60)
    print(f"📊 [FRED 하이일드 필터 + TQQQ 전략 결과]")
    print(f"   • 분석 기간 : {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({years:.1f}년)")
    print(f"   • 누적 수익률 : {(total_score - 1) * 100:,.0f}% ({total_score:.1f}배)")
    print(f"   • 연평균 수익률 (CAGR) : {cagr * 100:.2f}%")
    print(f"   • 최대 낙폭 (MDD) : {mdd * 100:.2f}%")
    print("-" * 60)
    
    print(f"🌍 [거시 경제(FRED) 상태 진단]")
    print(f"   • 현재 하이일드 스프레드 : {current_spread:.2f}%")
    print(f"   • 스프레드 20일 이평선 : {spread_ma:.2f}%")
    
    macro_msg = ""
    if is_macro_risk:
        macro_msg = "🚨 위험 (RISK OFF) - 모든 매매 중단 및 현금화"
    else:
        macro_msg = "✅ 안전 (RISK ON) - 알고리즘 매매 허용"
    print(f"   • 거시 필터 판정 : {macro_msg}")
    print("-" * 60)
    
    # 최종 행동 권고
    strategy_target_pos = last_row['Strategy_Pos']
    action_msg = ""
    
    if strategy_target_pos == 0.0:
        if is_macro_risk: action_msg = "현금 100% (거시 경제 위험)"
        else: action_msg = "현금 100% (기술적 하락 추세)"
    elif strategy_target_pos == 0.3: action_msg = "TQQQ 30% 보유"
    elif strategy_target_pos == 0.7: action_msg = "TQQQ 70% 보유"
    elif strategy_target_pos == 1.0: action_msg = "TQQQ 100% 풀매수"

    print(f"📢 [오늘({end_date.strftime('%Y-%m-%d')})의 최종 행동]")
    print(f"   • 목표 포지션 : {action_msg}")
    print("="*60)

# ==========================================
# 4. 실행부
# ==========================================
if __name__ == "__main__":
    # 데이터 수집 (API 키 필요)
    if FRED_API_KEY == 'YOUR_FRED_API_KEY_HERE':
        print("❌ 오류: 스크립트 상단에 FRED API 키를 입력해주세요!")
    else:
        df_raw = get_combined_data()
        
        if not df_raw.empty:
            # 최적화 범위 (예시)
            ma_range = range(20, 201, 1)    # 굵직한 추세만 확인
            rsi_range = range(70, 90, 1)         # 과열 기준
            buffer_range = [0.0, 0.01, 0.02, 0.03]   # 휩소 방지 버퍼
            
            print(f"\n⚡ FRED 필터 적용 후 최적 파라미터 탐색 중...")
            
            best_score = -999
            best_params = {}
            
            for ma, rsi, buf in itertools.product(ma_range, rsi_range, buffer_range):
                score, _ = run_tqqq_strategy(df_raw.copy(), ma, rsi, buf)
                if score > best_score:
                    best_score = score
                    best_params = {'ma': ma, 'rsi': rsi, 'buf': buf}
            
            # 최적 결과 실행
            final_score, df_final = run_tqqq_strategy(
                df_raw, 
                best_params['ma'], 
                best_params['rsi'], 
                best_params['buf']
            )
            
            analyze_today(
                df_final, 
                best_params['ma'], 
                best_params['rsi'], 
                best_params['buf'],
                final_score
            )
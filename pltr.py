import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import warnings
import time
from fredapi import Fred

warnings.filterwarnings("ignore")

# ==========================================
# 0. 설정 및 API 키
# ==========================================
FRED_API_KEY = 'b00d0e909d7e2e38815b8fbf62461695'
fred = Fred(api_key=FRED_API_KEY)

# ==========================================
# 1. 데이터 수집 (PLTR + FRED 금리차)
# ==========================================
def get_combined_data():
    print("⏳ 데이터 수집 중... (PLTR & FRED 지표)")
    
    # 1. PLTR & SHY 데이터 (Yahoo Finance)
    tickers = ['PLTR', 'SHY']
    df = yf.download(tickers, start="2020-09-30", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    
    df = df.dropna()
    
    # 2. FRED 데이터 (장단기 금리차: T10Y2Y)
    # 장단기 금리차가 역전되거나 급변할 때 리스크 관리용
    try:
        fred_data = fred.get_series('T10Y2Y', observation_start='2020-09-30')
        df['Yield_Curve'] = fred_data
        # 주말 등 데이터 공백 메우기
        df['Yield_Curve'] = df['Yield_Curve'].fillna(method='ffill')
    except Exception as e:
        print(f"⚠️ FRED 데이터를 가져오지 못했습니다: {e}")
        df['Yield_Curve'] = 1.0 # 기본값 (정상 상황 가정)

    # [데이터 가공]
    df['PLTR_Pct'] = df['PLTR'].pct_change()
    df['Sim_PLTR_2X'] = df['PLTR_Pct'] * 2.0  
    df['Sim_Cash'] = df['SHY'].pct_change()

    # RSI 계산
    delta = df['PLTR'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

# ==========================================
# 2. 백테스팅 엔진 (Macro Filter 추가)
# ==========================================
def run_pltr_strategy(df, ma_period, rsi_limit, sell_buffer):
    ma = df['PLTR'].ewm(span=ma_period, adjust=False).mean()
    
    price_arr = df['PLTR'].values
    ma_arr = ma.values
    rsi_arr = df['RSI'].values
    yield_arr = df['Yield_Curve'].values # FRED 지표
    
    position_size = np.zeros(len(df))
    current_pos = 0.0
    
    for i in range(1, len(df)):
        price = price_arr[i]
        ma_val = ma_arr[i]
        yield_val = yield_arr[i]
        sell_threshold = ma_val * (1 - sell_buffer)
        
        # [Macro Filter] 장단기 금리차에 따른 비중 제한 (과도한 레버리지 방지)
        # 금리차가 -0.5 미만으로 극심하게 역전되면 최대 비중을 50%로 제한
        max_alloc = 1.0
        if yield_val < -0.5:
            max_alloc = 0.5
        
        # 1. 매도 로직
        if price < sell_threshold:
            current_pos = 0.0 
        else:
            # 2. 매수 및 피라미딩
            if price > ma_val: 
                if current_pos == 0.0: 
                    current_pos = min(0.3, max_alloc)
                elif current_pos <= 0.3: 
                    current_pos = min(0.7, max_alloc)
                elif current_pos <= 0.7:
                    if rsi_arr[i] < rsi_limit: 
                        current_pos = min(1.0, max_alloc)
                    else: 
                        current_pos = min(0.7, max_alloc)
                elif current_pos > 0.7:
                    if rsi_arr[i] > rsi_limit: 
                        current_pos = 0.7
            else:
                pass 

        position_size[i] = current_pos

    df['Strategy_Pos'] = position_size
    pos = pd.Series(position_size, index=df.index).shift(1).fillna(0)
    strategy_ret = (df['Sim_PLTR_2X'] * pos) + (df['Sim_Cash'] * (1 - pos))
    df['Strategy_Ret'] = strategy_ret.fillna(0)
    
    return (1 + strategy_ret).prod(), df

# ==========================================
# 3. 결과 분석
# ==========================================
def analyze_today(df, ma_period, rsi_limit, sell_buffer, total_score):
    last_row = df.iloc[-1]
    price = last_row['PLTR']
    ma_val = df['PLTR'].ewm(span=ma_period, adjust=False).mean().iloc[-1]
    rsi = last_row['RSI']
    yield_val = last_row['Yield_Curve']
    
    strategy_target_pos = last_row['Strategy_Pos']
    real_cut_line = ma_val * (1 - sell_buffer)
    
    start_date = df.index[0]
    end_date = df.index[-1]
    years = (end_date - start_date).days / 365.25
    cagr = (total_score ** (1 / years)) - 1
    
    cum_ret = (1 + df['Strategy_Ret']).cumprod()
    mdd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
    
    print("\n" + "="*60)
    print(f"📊 [PLTR 2배 + FRED 매크로 필터 분석]")
    print(f" • 분석 기간 : {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print(f" • 누적 수익률 : {(total_score - 1) * 100:,.1f}% | CAGR : {cagr * 100:.2f}%")
    print(f" • 최대 낙폭 (MDD) : {mdd * 100:.2f}%")
    print("-" * 60)
    print(f"🛡️ [매크로 상태] 장단기 금리차(10Y-2Y): {yield_val:.2f}%")
    if yield_val < -0.5:
        print(" ⚠️ 주의: 금리차 역전 심화로 인해 최대 투자 비중이 50%로 제한됩니다.")
    print("-" * 60)
    
    action_msg = "관망"
    if strategy_target_pos >= 1.0: action_msg = "🚀 풀매수 (100%)"
    elif strategy_target_pos >= 0.7: action_msg = "🟢 공격적 매수 (70%)"
    elif strategy_target_pos >= 0.3: action_msg = "🟡 정찰병 진입 (30%)"
    else: action_msg = "🚨 전량 현금화"

    print(f"📢 [오늘의 추천 행동]")
    print(f" • 목표 비중 : {strategy_target_pos*100:.0f}% → {action_msg}")
    print(f" • 손절 기준 라인 : ${real_cut_line:.2f}")
    print("="*60)

# ==========================================
# 4. 실행부
# ==========================================
if __name__ == "__main__":
    df_raw = get_combined_data()
    
    # 파라미터 최적화 범위 (속도를 위해 조정 가능)
    ma_range = range(50, 150, 10) # EMA 범위 축소
    rsi_range = [75, 80, 85]
    buffer_range = [0.02, 0.04]
    
    best_score = -999
    best_params = {}
    
    print(f"⚡ 거시 지표 결합 최적 시나리오 탐색 중...")
    
    for ma, rsi, buf in itertools.product(ma_range, rsi_range, buffer_range):
        score, _ = run_pltr_strategy(df_raw.copy(), ma, rsi, buf)
        if score > best_score:
            best_score = score
            best_params = {'ma': ma, 'rsi': rsi, 'buf': buf}
            
    final_score, df_final = run_pltr_strategy(df_raw, best_params['ma'], best_params['rsi'], best_params['buf'])
    analyze_today(df_final, best_params['ma'], best_params['rsi'], best_params['buf'], final_score)
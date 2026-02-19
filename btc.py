import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import warnings
import time

warnings.filterwarnings("ignore")

# ==========================================
# 1. 데이터 수집
# ==========================================
def get_btc_data():
    print("⏳ 데이터 수집 중... (BTC, BITX 시뮬레이션)")
    tickers = ['BTC-USD', 'SHY'] 
    df = yf.download(tickers, start="2016-01-01", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        try: df = df['Close']
        except: df.columns = df.columns.get_level_values(0)
    
    df = df.dropna()
    
    # [수익률 데이터 생성]
    # 1. 현물 수익률
    df['BTC_Pct'] = df['BTC-USD'].pct_change()
    # 2. 레버리지 수익률 (2배수 추종 가정)
    df['Sim_BITX'] = df['BTC_Pct'] * 2.0  
    # 3. 현금성 자산 수익률
    df['Sim_Cash'] = df['SHY'].pct_change()

    # RSI 계산
    delta = df['BTC-USD'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

# ==========================================
# 2. 백테스팅 엔진 (스위칭 로직 적용)
# ==========================================
def run_switching_strategy(df, ma_period, rsi_limit, sell_buffer):
    ma = df['BTC-USD'].ewm(span=ma_period, adjust=False).mean()
    
    price_arr = df['BTC-USD'].values
    ma_arr = ma.values
    rsi_arr = df['RSI'].values
    
    # 자산 상태 기록 (0: 현금, 1: 현물, 2: 레버리지)
    asset_mode = np.zeros(len(df))
    current_mode = 0.0
    
    for i in range(1, len(df)):
        price = price_arr[i]
        ma_val = ma_arr[i]
        sell_threshold = ma_val * (1 - sell_buffer)
        
        # 1. 매도 로직 (추세 이탈)
        if price < sell_threshold:
            current_mode = 0.0 # 현금화
        else:
            # 2. 매수 및 스위칭 로직 (추세 보유)
            if price > ma_val:
                # RSI가 기준을 넘으면 과열 -> '현물(1배)'로 스위칭
                if rsi_arr[i] > rsi_limit:
                    current_mode = 1.0 
                # RSI가 안정적이면 -> '레버리지(2배)' 유지/진입
                else:
                    current_mode = 2.0
            else:
                # 버퍼존 (이평선과 손절라인 사이) -> 기존 포지션 유지
                pass 

        asset_mode[i] = current_mode

    # [수익률 계산]
    # 전일의 포지션(mode)이 오늘의 수익률을 결정함
    pos = pd.Series(asset_mode, index=df.index).shift(1).fillna(0)
    
    # 조건별 수익률 매칭 (Vectorized operation)
    conditions = [
        (pos == 0), # 현금
        (pos == 1), # 현물 (BTC)
        (pos == 2)  # 레버리지 (BITX)
    ]
    
    choices = [
        df['Sim_Cash'],
        df['BTC_Pct'],
        df['Sim_BITX']
    ]
    
    strategy_ret = np.select(conditions, choices, default=0)
    df['Strategy_Ret'] = strategy_ret
    
    # 마지막 상태 기록용
    df['Mode'] = asset_mode
    
    return (1 + strategy_ret).prod(), df

# ==========================================
# 3. 결과 분석
# ==========================================
def analyze_today(df, ma_period, rsi_limit, sell_buffer, total_score):
    last_row = df.iloc[-1]
    price = last_row['BTC-USD']
    ma_val = df['BTC-USD'].ewm(span=ma_period, adjust=False).mean().iloc[-1]
    rsi = last_row['RSI']
    current_mode = last_row['Mode']
    
    # 날짜 및 기간 계산
    start_date = df.index[0]
    end_date = df.index[-1]
    years = (end_date - start_date).days / 365.25
    
    # CAGR & MDD
    cagr = (total_score ** (1 / years)) - 1
    cum_ret = (1 + df['Strategy_Ret']).cumprod()
    mdd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
    
    # 단순 보유 비교 (레버리지 2배 존버)
    bh_ret = (1 + df['Sim_BITX']).prod()

    print("\n" + "="*60)
    print(f"📊 [BTC 현물↔레버리지 스위칭 전략]")
    print(f"   • 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({years:.1f}년)")
    print(f"   • 누적 수익률: {(total_score - 1) * 100:,.0f}% ({total_score:.2f}배)")
    print(f"   • 단순 보유(2x) 수익률: {(bh_ret - 1) * 100:,.0f}%")
    print(f"   • CAGR (연평균): {cagr * 100:.2f}%")
    print(f"   • MDD (최대낙폭): {mdd * 100:.2f}%")
    print("-" * 60)
    
    print(f"🎯 [최적 파라미터]")
    print(f"   • EMA: {ma_period}일선")
    print(f"   • RSI 기준: {rsi_limit} (초과 시 1배 현물로 전환)")
    print(f"   • 버퍼: {sell_buffer*100:.1f}%")
    print("-" * 60)
    
    status_msg = ""
    if current_mode == 0:
        status_msg = "🚨 전량 현금 (관망)"
    elif current_mode == 1:
        status_msg = "⚠️ 1배 현물 보유 (RSI 과열로 레버리지 해제)"
    elif current_mode == 2:
        status_msg = "🚀 2배 레버리지 풀매수 (추세 강력)"

    print(f"📢 [오늘({end_date.strftime('%Y-%m-%d')})의 추천 포지션]")
    print(f"   • 가격: ${price:,.2f} / RSI: {rsi:.1f}")
    print(f"   • 기준선: ${ma_val:,.2f}")
    print(f"   • 포지션: {status_msg}")
    print("="*60)

# ==========================================
# 4. 메인 실행 (최적화)
# ==========================================
if __name__ == "__main__":
    df_raw = get_btc_data()
    
    # 파라미터 탐색 범위
    ma_range = range(60, 201, 1)   # 이평선
    rsi_range = range(70, 96, 5)   # RSI 기준 (70~95)
    buffer_range = [0.0, 0.03, 0.05] # 휩소 버퍼
    
    print(f"\n⚡ 최적 시나리오 분석 중... (총 {len(ma_range)*len(rsi_range)*len(buffer_range)}개 조합)")
    start_time = time.time()
    
    best_score = -999
    best_params = {}
    
    for ma, rsi, buf in itertools.product(ma_range, rsi_range, buffer_range):
        score, _ = run_switching_strategy(df_raw.copy(), ma, rsi, buf)
        if score > best_score:
            best_score = score
            best_params = {'ma': ma, 'rsi': rsi, 'buf': buf}
            
    print(f"✅ 완료! (소요시간: {time.time() - start_time:.1f}초)")
    
    # 최적 결과 실행
    final_score, df_final = run_switching_strategy(
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
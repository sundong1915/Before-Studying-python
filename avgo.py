import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import warnings
import time

warnings.filterwarnings("ignore")

# ==========================================
# 1. 데이터 수집 (AVGO 기준 & 3배 레버리지 시뮬레이션)
# ==========================================
def get_avgo_data():
    print("⏳ 데이터 수집 중... (AVGO, 2007년~현재)")
    # AVGO: Broadcom Inc.
    # SHY: 단기채 (현금 대용)
    tickers = ['AVGO', 'SHY'] 
    
    # AVGO(구 Avago)가 2009년 상장, 데이터 안정성을 위해 2010년부터 수집
    df = yf.download(tickers, start="2010-01-01", progress=False)
    
    # yfinance 버전 이슈 대응 (MultiIndex 처리)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df['Close']
        except:
            # 최신 버전 대응
            df = df.xs('Close', axis=1, level=0)
    
    df = df.dropna()
    
    # [데이터 가공]
    # 1. AVGO(1배수) 변동률 -> 매매 신호용
    df['AVGO_Pct'] = df['AVGO'].pct_change()
    
    # 2. AVGO 3X 시뮬레이션 (Simulated 3x Leveraged)
    # *주의: 개별주 3배는 변동성 끌림(Volatility Drag) 효과가 매우 큽니다.
    df['Sim_AVGO_3X'] = df['AVGO_Pct'] * 3.0  
    
    # 3. 현금성 자산 (SHY)
    df['Sim_Cash'] = df['SHY'].pct_change()

    # RSI 계산 (AVGO 1배수 차트 기준 -> 노이즈 제거)
    delta = df['AVGO'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

# ==========================================
# 2. 백테스팅 엔진 (피라미딩 전략)
# ==========================================
def run_avgo_strategy(df, ma_period, rsi_limit, sell_buffer):
    # 지수이동평균(EMA) 계산 - AVGO 기준
    ma = df['AVGO'].ewm(span=ma_period, adjust=False).mean()
    
    price_arr = df['AVGO'].values
    ma_arr = ma.values
    rsi_arr = df['RSI'].values
    
    position_size = np.zeros(len(df))
    current_pos = 0.0
    
    for i in range(1, len(df)):
        price = price_arr[i]
        ma_val = ma_arr[i]
        
        # 매도 기준선 (EMA 대비 버퍼만큼 하락 시)
        sell_threshold = ma_val * (1 - sell_buffer)
        
        # 1. 매도 로직 (손절/익절)
        if price < sell_threshold:
            current_pos = 0.0 
        else:
            # 2. 매수 및 비중 조절 (피라미딩: 30% -> 70% -> 100%)
            if price > ma_val: 
                if current_pos == 0.0: current_pos = 0.3      # 1차 진입 (정찰병)
                elif current_pos == 0.3: current_pos = 0.7    # 2차 불타기
                elif current_pos == 0.7:
                    # RSI 과열 체크 (개별주는 RSI 90 이상도 자주 감)
                    if rsi_arr[i] < rsi_limit: current_pos = 1.0 # 풀매수
                    else: current_pos = 0.7 
                elif current_pos == 1.0:
                    # 과열 시 비중 축소
                    if rsi_arr[i] > rsi_limit: current_pos = 0.7 
                    else: current_pos = 1.0
            else:
                # 버퍼존: 포지션 유지
                pass 

        position_size[i] = current_pos

    # 전략 포지션 기록
    df['Strategy_Pos'] = position_size

    # 수익률 계산
    # 포지션만큼은 3배 레버리지(Sim_AVGO_3X), 나머지는 현금(Sim_Cash)
    pos = pd.Series(position_size, index=df.index).shift(1).fillna(0)
    strategy_ret = (df['Sim_AVGO_3X'] * pos) + (df['Sim_Cash'] * (1 - pos))
    df['Strategy_Ret'] = strategy_ret.fillna(0)
    
    # 누적 수익(배수) 반환
    return (1 + strategy_ret).prod(), df

# ==========================================
# 3. 결과 분석
# ==========================================
def analyze_today(df, ma_period, rsi_limit, sell_buffer, total_score):
    last_row = df.iloc[-1]
    price = last_row['AVGO']
    ma_val = df['AVGO'].ewm(span=ma_period, adjust=False).mean().iloc[-1]
    rsi = last_row['RSI']
    
    strategy_target_pos = last_row['Strategy_Pos']
    real_cut_line = ma_val * (1 - sell_buffer)
    
    # 성과 지표
    start_date = df.index[0]
    end_date = df.index[-1]
    years = (end_date - start_date).days / 365.25
    cagr = (total_score ** (1 / years)) - 1 if total_score > 0 else -0.99
    
    cum_ret = (1 + df['Strategy_Ret']).cumprod()
    mdd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
    
    print("\n" + "="*60)
    print(f"📊 [AVGO(Signal) -> AVGO 3X(Simulated) 전략 시뮬레이션]")
    print(f"   • 분석 기간 : {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({years:.1f}년)")
    print(f"   • 누적 수익률 : {(total_score - 1) * 100:,.0f}% ({total_score:.1f}배)")
    print(f"   • 연평균 수익률 (CAGR) : {cagr * 100:.2f}%")
    print(f"   • 최대 낙폭 (MDD) : {mdd * 100:.2f}%")
    print("-" * 60)
    
    print(f"🎯 [최적 파라미터]")
    print(f"   • EMA {ma_period}일 / RSI {rsi_limit} / 버퍼 {sell_buffer*100:.0f}%")
    print("-" * 60)
    
    action_msg = ""
    if strategy_target_pos == 0.0:
        action_msg = "🚨 전량 매도 (관망)"
    elif strategy_target_pos == 0.3:
        action_msg = "🟢 1단계 진입 (정찰병 30%)"
    elif strategy_target_pos == 0.7:
        if rsi > rsi_limit:
            action_msg = "⏸️ 비중 70% 유지 (RSI 과열로 풀매수 보류)"
        else:
            action_msg = "🟢 2단계 불타기 (70%)"
    elif strategy_target_pos == 1.0:
        action_msg = "🚀 풀매수 홀딩 (100%)"

    print(f"📢 [오늘({end_date.strftime('%Y-%m-%d')})의 추천 행동]")
    print(f"   • 기준(AVGO) 주가 : ${price:.2f}")
    print(f"   • 기준(AVGO) 이평선 : ${ma_val:.2f} (이탈시 ${real_cut_line:.2f} 매도)")
    print(f"   • 목표 비중(3X) : {strategy_target_pos*100:.0f}% → {action_msg}")
    print("="*60)

# ==========================================
# 4. 실행부
# ==========================================
if __name__ == "__main__":
    # 데이터 수집 (AVGO 2010년~현재)
    df_raw = get_avgo_data()
    
    # AVGO는 변동성이 커서 탐색 범위를 넓게 잡되, 속도를 위해 step을 조정
    ma_range = range(20, 201, 1)   # 5일 간격으로 탐색 (속도 향상)
    rsi_range = range(70, 96, 2)   # 2단위 탐색
    buffer_range = [0.0, 0.02, 0.04, 0.06] # 개별주는 버퍼를 좀 더 넉넉히
    
    total_combinations = len(ma_range) * len(rsi_range) * len(buffer_range)
    
    print(f"\n⚡ 최적 시나리오 분석 시작...")
    print(f"   - 총 시나리오: {total_combinations:,}개")
    
    start_time = time.time()
    best_score = -999
    best_params = {}
    
    # 최적 파라미터 찾기
    for ma, rsi, buf in itertools.product(ma_range, rsi_range, buffer_range):
        score, _ = run_avgo_strategy(df_raw.copy(), ma, rsi, buf)
        
        if score > best_score:
            best_score = score
            best_params = {'ma': ma, 'rsi': rsi, 'buf': buf}
            
    print(f"\n✅ 완료! (소요시간: {time.time() - start_time:.1f}초)")
    
    # 최적 결과로 최종 실행
    final_score, df_final = run_avgo_strategy(
        df_raw, 
        best_params['ma'], 
        best_params['rsi'], 
        best_params['buf']
    )
    
    # 결과 분석
    analyze_today(
        df_final, 
        best_params['ma'], 
        best_params['rsi'], 
        best_params['buf'],
        final_score
    )
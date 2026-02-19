import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import warnings
import time

warnings.filterwarnings("ignore")

# ==========================================
# 1. 데이터 수집 (MSFT & 2배 레버리지 생성)
# ==========================================
def get_msft_data():
    print("⏳ 데이터 수집 중... (MSFT, 최근 20년+)")
    # MSFT: 마이크로소프트
    # SHY: 단기채 (현금 대용)
    tickers = ['MSFT', 'SHY'] 
    
    # 2008년 금융위기를 포함하기 위해 2004년부터 데이터 수집
    df = yf.download(tickers, start="2004-01-01", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df['Close']
        except:
            df.columns = df.columns.get_level_values(0)
    
    df = df.dropna()
    
    # [데이터 가공]
    # 1. MSFT 변동률
    df['MSFT_Pct'] = df['MSFT'].pct_change()
    
    # 2. MSFT 2배 레버리지 시뮬레이션 (Simulated 2x)
    # 개별주 3배는 장기 보유 시 녹아내릴 위험이 커서 2배로 설정 (MSFU 등 참고)
    df['Sim_MSFT_2X'] = df['MSFT_Pct'] * 2.0  
    
    # 3. 현금성 자산 (SHY)
    df['Sim_Cash'] = df['SHY'].pct_change()

    # RSI 계산 (MSFT 현물 기준)
    delta = df['MSFT'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

# ==========================================
# 2. 백테스팅 엔진 (피라미딩 전략)
# ==========================================
def run_msft_strategy(df, ma_period, rsi_limit, sell_buffer):
    # 지수이동평균(EMA) 계산
    ma = df['MSFT'].ewm(span=ma_period, adjust=False).mean()
    
    price_arr = df['MSFT'].values
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
                    # RSI 과열 체크
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
    # 포지션만큼은 2배 레버리지(Sim_MSFT_2X), 나머지는 현금(Sim_Cash)
    pos = pd.Series(position_size, index=df.index).shift(1).fillna(0)
    strategy_ret = (df['Sim_MSFT_2X'] * pos) + (df['Sim_Cash'] * (1 - pos))
    df['Strategy_Ret'] = strategy_ret.fillna(0)
    
    # 누적 수익(배수) 반환
    return (1 + strategy_ret).prod(), df

# ==========================================
# 3. 결과 분석
# ==========================================
def analyze_today(df, ma_period, rsi_limit, sell_buffer, total_score):
    last_row = df.iloc[-1]
    price = last_row['MSFT']
    ma_val = df['MSFT'].ewm(span=ma_period, adjust=False).mean().iloc[-1]
    rsi = last_row['RSI']
    
    strategy_target_pos = last_row['Strategy_Pos']
    real_cut_line = ma_val * (1 - sell_buffer)
    
    # 성과 지표
    start_date = df.index[0]
    end_date = df.index[-1]
    years = (end_date - start_date).days / 365.25
    cagr = (total_score ** (1 / years)) - 1
    
    cum_ret = (1 + df['Strategy_Ret']).cumprod()
    mdd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
    
    print("\n" + "="*60)
    print(f"📊 [Microsoft(MSFT) 2배 레버리지 시뮬레이션 분석]")
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
    print(f"   • 현재 주가 : ${price:.2f}")
    print(f"   • 기준 라인 : ${ma_val:.2f} (이탈시 ${real_cut_line:.2f} 매도)")
    print(f"   • 목표 비중 : {strategy_target_pos*100:.0f}% → {action_msg}")
    print("="*60)

# ==========================================
# 4. 실행부
# ==========================================
if __name__ == "__main__":
    # 데이터 수집 (20년치)
    df_raw = get_msft_data()
    
    # MSFT는 우상향 성향이 강함 -> 파라미터 탐색 범위 설정
    ma_range = range(20, 201, 1)   
    rsi_range = range(70, 96, 2)   
    buffer_range = [0.0, 0.01, 0.02, 0.03, 0.04] 
    
    total_combinations = len(ma_range) * len(rsi_range) * len(buffer_range)
    
    print(f"\n⚡ 최적 시나리오 분석 시작...")
    print(f"   - 총 시나리오: {total_combinations:,}개")
    
    start_time = time.time()
    best_score = -999
    best_params = {}
    
    # 최적 파라미터 찾기
    for ma, rsi, buf in itertools.product(ma_range, rsi_range, buffer_range):
        score, _ = run_msft_strategy(df_raw.copy(), ma, rsi, buf)
        
        if score > best_score:
            best_score = score
            best_params = {'ma': ma, 'rsi': rsi, 'buf': buf}
            
    print(f"\n✅ 완료! (소요시간: {time.time() - start_time:.1f}초)")
    
    # 최적 결과로 최종 실행
    final_score, df_final = run_msft_strategy(
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
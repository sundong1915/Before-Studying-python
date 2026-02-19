import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import warnings
import time

warnings.filterwarnings("ignore")

def get_data_advanced():
    print("⏳ 데이터 수집 및 가공 중... (3중 필터 준비)")
    tickers = ['QQQ', 'SHY', '^VIX']
    df = yf.download(tickers, start="2010-01-01", progress=False)['Close']
    df.columns = tickers
    df = df.dropna()
    
    # 가상 데이터 생성
    df['QQQ_Pct'] = df['QQQ'].pct_change()
    df['Sim_TQQQ'] = df['QQQ_Pct'] * 3
    df['Sim_SGOV'] = df['SHY'].pct_change()
    
    # VIX 이동평균 (급등 감지용)
    df['VIX_MA50'] = df['^VIX'].ewm(span=50, adjust=False).mean()

    # RSI (14일 기준) 미리 계산
    delta = df['QQQ'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def run_pyramiding_strategy(df, ma_period, rsi_limit, vix_panic_line):
    """
    params:
      - ma_period: EMA 기간
      - rsi_limit: RSI 과열 기준
      - vix_panic_line: VIX 매도(공포) 기준 (최적화 대상!)
    """
    # EMA 계산
    ma = df['QQQ'].ewm(span=ma_period, adjust=False).mean()
    
    position_size = [0.0] * len(df)
    current_pos = 0.0
    
    # Numpy 배열 변환 (속도 최적화)
    price_arr = df['QQQ'].values
    ma_arr = ma.values
    vix_arr = df['^VIX'].values
    vix_ma_arr = df['VIX_MA50'].values
    rsi_arr = df['RSI'].values
    
    for i in range(1, len(df)):
        price = price_arr[i]
        ma_val = ma_arr[i]
        vix = vix_arr[i]
        vix_ma = vix_ma_arr[i]
        rsi = rsi_arr[i]
        
        # 1. 매도(대피) 조건
        # VIX 기준을 vix_panic_line 변수로 변경!
        if price < ma_val or vix > vix_ma * 1.2 or vix > vix_panic_line:
            current_pos = 0.0 
            
        # 2. 매수(진입/유지) 조건
        else:
            if current_pos == 0.0:
                current_pos = 0.3
            elif current_pos == 0.3:
                current_pos = 0.7
            elif current_pos == 0.7:
                # 풀매수 조건: VIX가 안정적이고(20 미만), RSI 과열 아님
                if vix < 20 and rsi < rsi_limit:
                    current_pos = 1.0 
                else:
                    current_pos = 0.7 
            elif current_pos == 1.0:
                # RSI 과열 시 비중 축소
                if rsi > rsi_limit:
                    current_pos = 0.7
                else:
                    current_pos = 1.0

        position_size[i] = current_pos

    df['Pos_Size'] = position_size
    
    pos = df['Pos_Size'].shift(1)
    strategy_ret = (df['Sim_TQQQ'] * pos) + (df['Sim_SGOV'] * (1 - pos))
    
    total_ret = (1 + strategy_ret).prod()
    return total_ret, df

def analyze_today(df, ma_period, rsi_limit, vix_panic_line):
    last_row = df.iloc[-1]
    last_pos = df['Pos_Size'].iloc[-1]
    
    price = last_row['QQQ']
    ma_val = df['QQQ'].ewm(span=ma_period, adjust=False).mean().iloc[-1]
    vix = last_row['^VIX']
    rsi = last_row['RSI']
    
    print("\n" + "="*60)
    print(f"🏆 [최종 승리 조합] 20년 백테스트 결과")
    print(f"1. EMA 이동평균선 : {ma_period}일")
    print(f"2. RSI 과열 기준  : {rsi_limit}")
    print(f"3. VIX 공포 기준  : {vix_panic_line} (이보다 높으면 전량 매도)")
    print("="*60)
    
    print(f"\n[오늘 시장 데이터 ({df.index[-1].date()})]")
    print(f"• 추세 (Price vs EMA) : {price:.2f} vs {ma_val:.2f}", end=" ")
    if price > ma_val: print("🔺 (상승장)")
    else: print("🔻 (하락장)")
        
    print(f"• 과열 (RSI)          : {rsi:.1f} (기준: {rsi_limit})", end=" ")
    if rsi > rsi_limit: print("🔥 (과열)")
    else: print("✨ (적정)")
    
    print(f"• 공포 (VIX)          : {vix:.1f} (기준: {vix_panic_line})", end=" ")
    if vix > vix_panic_line: print("😱 (공포 - 대피!)")
    else: print("😊 (안정)")

    print("-" * 60)
    print(f"🎯 [오늘의 결론] 추천 비중: {last_pos*100:.0f}%")
    
    if last_pos == 0.0:
        print("👉 전량 매도 (SGOV 100%) : 위험 신호가 떴습니다.")
    elif last_pos == 0.3:
        print("👉 정찰병 진입 (TQQQ 30%)")
    elif last_pos == 0.7:
        print("👉 비중 확대 (TQQQ 70%)")
    elif last_pos == 1.0:
        print("👉 풀매수 (TQQQ 100%) : 모든 신호가 완벽합니다.")
    print("="*60)

# --- 실행부 ---
if __name__ == "__main__":
    df_raw = get_data_advanced()
    
    # 🔍 최적화 범위 설정 (너무 오래 걸리지 않게 조정)
    # 1. EMA: 120 ~ 200 (2일 단위)
    ma_range = range(120, 201, 2) 
    
    # 2. RSI: 70 ~ 85 (2일 단위)
    rsi_range = range(70, 86, 2)
    
    # 3. VIX 공포 기준: 28 ~ 40 (1 단위) - 여기가 핵심!
    vix_range = range(28, 41, 1)
    
    best_score = -999
    best_params = {'ma': 150, 'rsi': 75, 'vix': 35}
    
    total_combinations = len(ma_range) * len(rsi_range) * len(vix_range)
    print(f"\n🔍 총 {total_combinations}개의 'EMA + RSI + VIX' 조합을 테스트합니다.")
    print("   (잠시만 기다려주세요, 약 3~5분 소요됩니다...)")
    
    count = 0
    start_time = time.time()
    
    # 3중 루프 Grid Search
    for ma, rsi, vix_cut in itertools.product(ma_range, rsi_range, vix_range):
        score, _ = run_pyramiding_strategy(df_raw.copy(), ma, rsi, vix_cut)
        
        if score > best_score:
            best_score = score
            best_params = {'ma': ma, 'rsi': rsi, 'vix': vix_cut}
            # 중간중간 갱신될 때만 출력
            print(f"   ✨ 발견! EMA {ma} / RSI {rsi} / VIX {vix_cut} -> 수익 {score:.2f}배")
            
        count += 1
        if count % 2000 == 0:
            elapsed = time.time() - start_time
            print(f"   ... {count}/{total_combinations} 진행 중 ({elapsed:.1f}초)")

    print(f"\n✅ 최적화 완료! (총 소요시간: {time.time() - start_time:.1f}초)")
    
    # 찾은 최적 값으로 오늘 분석
    _, df_final = run_pyramiding_strategy(df_raw, best_params['ma'], best_params['rsi'], best_params['vix'])
    analyze_today(df_final, best_params['ma'], best_params['rsi'], best_params['vix'])
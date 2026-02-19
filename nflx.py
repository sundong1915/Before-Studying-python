import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import warnings
import time
from pandas_datareader import data as pdr

warnings.filterwarnings("ignore")

# ==========================================
# 1. 데이터 수집 (NFLX, SHY, 장단기 금리차)
# ==========================================
def get_combined_data():
    print("⏳ 데이터 수집 중... (NFLX, SHY, 10Y-2Y Spread)")
    
    tickers = ['NFLX', 'SHY']
    # 넷플릭스의 성장을 충분히 반영하기 위해 2006년부터 수집합니다.
    df = yf.download(tickers, start="2006-01-01", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    
    import pandas_datareader.data as web
    try:
        spread = web.DataReader('T10Y2Y', 'fred', start='2006-01-01')
    except:
        print("⚠️ FRED 연결 실패. yfinance 데이터로 대체 시도.")
        t10 = yf.download("^TNX", start="2006-01-01", progress=False)['Close']
        t02 = yf.download("^IRX", start="2006-01-01", progress=False)['Close']
        spread = t10 - t02

    df = df.join(spread).fillna(method='ffill')
    df.columns = list(df.columns[:-1]) + ['T10Y2Y']
    df = df.dropna()

    # 수익률 정의
    df['NFLX_Pct'] = df['NFLX'].pct_change()      # 본주 (1배)
    df['Sim_Lev_2X'] = df['NFLX_Pct'] * 2.0      # NFLU 가상 (2배)
    df['Sim_Cash'] = df['SHY'].pct_change()      # 현금 (SHY)

    # RSI 14 계산
    delta = df['NFLX'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

# ==========================================
# 2. 백테스팅 엔진 (스위칭 로직)
# ==========================================
def run_strategy(df, ma_period, rsi_limit, sell_buffer):
    ma = df['NFLX'].ewm(span=ma_period, adjust=False).mean()
    
    price_arr = df['NFLX'].values
    ma_arr = ma.values
    rsi_arr = df['RSI'].values
    spread_arr = df['T10Y2Y'].values
    
    position_size = np.zeros(len(df))
    is_leveraged = np.zeros(len(df)) 
    
    current_pos = 0.0
    
    for i in range(1, len(df)):
        price = price_arr[i]
        ma_val = ma_arr[i]
        rsi_val = rsi_arr[i]
        spread_val = spread_arr[i]
        
        sell_threshold = ma_val * (1 - sell_buffer)
        
        if price < sell_threshold:
            current_pos = 0.0
            is_leveraged[i] = 0
        else:
            if price > ma_val:
                if current_pos == 0.0: current_pos = 0.3
                elif current_pos <= 0.3: current_pos = 0.7
                else: current_pos = 1.0
                
                # RSI 스위칭: 과열 시 본주로 전환하여 변동성 관리
                if rsi_val < rsi_limit:
                    is_leveraged[i] = 1 # NFLU(2배) 모드
                else:
                    is_leveraged[i] = 0 # NFLX(본주) 모드
            
            # 매크로 필터: 금리 역전 시 방어 모드
            if spread_val < 0:
                current_pos = min(current_pos, 0.3)
                is_leveraged[i] = 0

        position_size[i] = current_pos

    df['Strategy_Pos'] = position_size
    df['Is_Leveraged'] = is_leveraged
    
    pos = pd.Series(position_size, index=df.index).shift(1).fillna(0)
    is_lev = pd.Series(is_leveraged, index=df.index).shift(1).fillna(0)
    
    ret_lev = df['Sim_Lev_2X'] * pos * is_lev
    ret_spot = df['NFLX_Pct'] * pos * (1 - is_lev)
    ret_cash = df['Sim_Cash'] * (1 - pos)
    
    df['Strategy_Ret'] = ret_lev + ret_spot + ret_cash
    
    return (1 + df['Strategy_Ret']).prod(), df

# ==========================================
# 3. 결과 분석 및 출력
# ==========================================
def analyze_today(df, ma_period, rsi_limit, sell_buffer, total_score):
    last_row = df.iloc[-1]
    price = last_row['NFLX']
    ma_series = df['NFLX'].ewm(span=ma_period, adjust=False).mean()
    current_ma = ma_series.iloc[-1]
    sell_line = current_ma * (1 - sell_buffer)
    rsi = last_row['RSI']
    current_spread = last_row['T10Y2Y']
    
    target_pos = last_row['Strategy_Pos']
    target_lev = last_row['Is_Leveraged']
    
    start_date = df.index[0]
    end_date = df.index[-1]
    years = (end_date - start_date).days / 365.25
    cagr = (total_score ** (1 / years)) - 1
    cum_ret = (1 + df['Strategy_Ret']).cumprod()
    mdd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
    
    print("\n" + "═"*60)
    print(f"🏆 [최적화 완료: NFLX ↔ NFLU 스위칭 전략]")
    print(f"   • 분석 기간 : {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print(f"   • 누적 수익률 : {(total_score - 1) * 100:,.0f}% ({total_score:.1f}배)")
    print(f"   • 연평균 수익률 (CAGR) : {cagr * 100:.2f}%")
    print(f"   • 최대 낙폭 (MDD) : {mdd * 100:.2f}%")
    
    print("-" * 60)
    print(f"✨ [도출된 최적 파라미터]")
    print(f"   • 최적 EMA 기간 : {ma_period}일")
    print(f"   • 최적 RSI 기준 : {rsi_limit} (과열 시 본주 스위칭)")
    print(f"   • 최적 매도 버퍼 : {sell_buffer * 100:.1f}%")
    
    print("-" * 60)
    dist_to_sell = ((price - sell_line) / price) * 100
    spread_status = "⚠️ 역전 (위험)" if current_spread < 0 else "✅ 정상 (안전)"
    
    print(f"🎯 [현재 넷플릭스 지표]")
    print(f"   • NFLX 현재가 : ${price:,.2f}")
    print(f"   • 현재 EMA({ma_period}) 가격 : ${current_ma:,.2f}")
    print(f"   • 매도 감시선 : ${sell_line:,.2f} (남은 여유: {dist_to_sell:.2f}%)")
    print(f"   • 현재 RSI : {rsi:.1f} / 금리차 : {current_spread:.4f} ({spread_status})")
    
    print("-" * 60)
    mode_msg = "🚀 NFLU(2배) 모드" if target_lev == 1 else "🛡️ NFLX(본주) 스위칭 모드"
    if target_pos == 0: mode_msg = "🚨 전량 현금 대피"

    print(f"📢 [오늘의 추천 행동]")
    print(f"   • 목표 비중 : {target_pos*100:.0f}%")
    print(f"   • 운용 모드 : {mode_msg}")
    print("═"*60)

# ==========================================
# 4. 실행부
# ==========================================
if __name__ == "__main__":
    df_raw = get_combined_data()
    
    if df_raw is not None:
        # 정밀 탐색 범위
        ma_range = range(20, 201, 1) 
        rsi_range = range(70, 96, 1)
        buffer_range = [0.01, 0.02, 0.03, 0.05] 
        
        total_comb = len(ma_range) * len(rsi_range) * len(buffer_range)
        print(f"\n⚡ {total_comb:,}개 조합 정밀 분석 중... 넷플릭스의 20년 역사를 탐색합니다.")
        
        start_time = time.time()
        best_score = -999
        best_params = {}
        
        for ma, rsi, buf in itertools.product(ma_range, rsi_range, buffer_range):
            score, _ = run_strategy(df_raw.copy(), ma, rsi, buf)
            if score > best_score:
                best_score = score
                best_params = {'ma': ma, 'rsi': rsi, 'buf': buf}
                
        print(f"✅ 분석 완료! (소요시간: {time.time() - start_time:.1f}초)")
        
        final_score, df_final = run_strategy(df_raw, best_params['ma'], best_params['rsi'], best_params['buf'])
        analyze_today(df_final, best_params['ma'], best_params['rsi'], best_params['buf'], final_score)
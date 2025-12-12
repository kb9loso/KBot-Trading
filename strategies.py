# strategies.py
import pandas as pd
import numpy as np
from data_fetcher import get_historical_klines


# ==============================================================================
# =================== [INÍCIO] FUNÇÕES DE ESTRATÉGIA (def) ===================
# ==============================================================================
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # ============ MÉDIAS MÓVEIS ============
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['EMA_34'] = df['close'].ewm(span=34, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()

    # ============ RSI ============
    def calc_rsi(series, length=14):
        delta = series.diff(1)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
        avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
        rs = avg_gain / (avg_loss.replace(0, 1e-10))
        return 100 - (100 / (1 + rs))

    df['RSI_14'] = calc_rsi(df['close'], 14)
    df['RSI_9'] = calc_rsi(df['close'], 9)

    # ============ MACD & Histograma ============
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # ============ BANDAS DE BOLLINGER ============
    period_bb = 20
    df['BB_middle'] = df['close'].rolling(window=period_bb).mean()
    df['BB_std'] = df['close'].rolling(window=period_bb).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    df['BBW'] = (df['BB_upper'] - df['BB_lower']) / (df['BB_middle'].replace(0, 1e-10))
    df['BBW_min_120'] = df['BBW'].rolling(window=120).min()

    # ============ ATR & Keltner Channels ============
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    df['KC_upper'] = df['EMA_21'] + (df['ATR'] * 2)
    df['KC_lower'] = df['EMA_21'] - (df['ATR'] * 2)

    # ============ Donchian Channels ============
    df['DC_high'] = df['high'].rolling(window=20).max()
    df['DC_low'] = df['low'].rolling(window=20).min()

    # ============ ADX (Wilder's Smoothed) ============
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    p_dm_s = pd.Series(plus_dm, index=df.index)
    m_dm_s = pd.Series(minus_dm, index=df.index)

    # 1. Suavização de Wilder (RMA) para TR, +DM e -DM
    # Equivalente a ewm(alpha=1/14)
    smooth_tr = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    smooth_p_dm = p_dm_s.ewm(alpha=1 / 14, adjust=False).mean()
    smooth_m_dm = m_dm_s.ewm(alpha=1 / 14, adjust=False).mean()

    # 2. Cálculo dos Indicadores Direcionais (+DI, -DI)
    plus_di = 100 * (smooth_p_dm / smooth_tr.replace(0, 1e-10))
    minus_di = 100 * (smooth_m_dm / smooth_tr.replace(0, 1e-10))

    # 3. Cálculo do DX e ADX (Suavizado)
    dx = (abs(plus_di - minus_di) / ((plus_di + minus_di).replace(0, 1e-10))) * 100
    df['ADX'] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    df['P_DI'] = plus_di
    df['M_DI'] = minus_di

    # ============ ESTOCÁSTICO ============
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['STOCH_K'] = 100 * ((df['close'] - low_14) / ((high_14 - low_14).replace(0, 1e-10)))
    df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()

    # ============ VWAP ============
    # O VWAP tradicional deve resetar a cada sessão (diariamente).
    df['TP_Vol'] = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
    df['Vol'] = df['volume']
    daily_grouper = df.index.date
    vwap_num = df.groupby(daily_grouper)['TP_Vol'].cumsum()
    vwap_den = df.groupby(daily_grouper)['Vol'].cumsum().replace(0, 1e-10)
    df['VWAP'] = vwap_num / vwap_den

    # ============ MOMENTUM EXTRAS ============
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

    hh = df['high'].rolling(14).max()
    ll = df['low'].rolling(14).min()
    df['WILLR'] = -100 * ((hh - df['close']) / (hh - ll).replace(0, 1e-10))

    df['ROC'] = ((df['close'] - df['close'].shift(9)) / df['close'].shift(9)) * 100

    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['volume'] / df['Volume_MA'].replace(0, 1)

    df['Chikou'] = df['close'].shift(26)

    # Aroon Indicator
    df['Aroon_Up'] = df['high'].rolling(window=25).apply(lambda x: (24 - x.argmax()) / 24 * 100, raw=False)
    df['Aroon_Down'] = df['low'].rolling(window=25).apply(lambda x: (24 - x.argmin()) / 24 * 100, raw=False)
    df['Aroon_Osc'] = df['Aroon_Up'] - df['Aroon_Down']

    def calc_tema(series, span):
        ema1 = series.ewm(span=span, adjust=False).mean()
        ema2 = ema1.ewm(span=span, adjust=False).mean()
        ema3 = ema2.ewm(span=span, adjust=False).mean()
        return (3 * ema1) - (3 * ema2) + ema3

    df['TEMA_9'] = calc_tema(df['close'], 9)

    # Price Action (Corpo e Pavios)
    df['body_size'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']

    # ============ ICHIMOKU ============
    # Chikou: Preço atual deslocado para trás (para plot).
    # Para sinal lógico (Preço vs Chikou), comparamos Close atual vs Close atrasado.
    df['Chikou'] = df['close'].shift(26)

    tenkan_window, kijun_window = 20, 60
    senkou_b_window, displacement = 120, 30

    df['Tenkan_sen'] = (df['high'].rolling(tenkan_window).max() + df['low'].rolling(tenkan_window).min()) / 2
    df['Kijun_sen'] = (df['high'].rolling(kijun_window).max() + df['low'].rolling(kijun_window).min()) / 2

    # Cloud (Senkou A e B) projetada para o futuro, mas aqui alinhamos ao candle atual para cálculo de sinal
    senkou_a = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2)
    senkou_b = ((df['high'].rolling(senkou_b_window).max() + df['low'].rolling(senkou_b_window).min()) / 2)

    # Importante: O valor "Atual" da nuvem é o que foi projetado 30 candles atrás
    df['Senkou_Span_A'] = senkou_a.shift(displacement)
    df['Senkou_Span_B'] = senkou_b.shift(displacement)


    # ============ VALORES ANTERIORES ============
    cols_to_shift = [
        'close', 'open', 'high', 'low', 'volume',
        'EMA_9', 'EMA_12', 'EMA_21', 'EMA_34', 'EMA_50', 'EMA_100', 'EMA_200',
        'RSI_9', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist', 'ADX', 'P_DI', 'M_DI',
        'BB_upper', 'BB_lower', 'BBW', 'KC_upper', 'KC_lower',
        'DC_high', 'DC_low', 'CCI', 'WILLR', 'ROC', 'ATR', 'STOCH_K', 'STOCH_D', 'VWAP'
    ]
    for col in cols_to_shift:
        if col in df.columns:
            df[f'{col}_prev'] = df[col].shift(1)

    return df


# ==============================================================================
# ============================ HELPERS DE LÓGICA ===============================
# ==============================================================================

def trend_bull(c): return c['close'] > c['EMA_200']
def trend_bear(c): return c['close'] < c['EMA_200']
def vol_ok(c): return c['Vol_Ratio'] > 1.0
def adx_ok(c): return c['ADX'] > 20


# ==============================================================================
# ==================== DEFINIÇÕES DE FUNÇÕES EXPLÍCITAS ========================
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. TENDÊNCIA (30 ESTRATÉGIAS)
# ------------------------------------------------------------------------------

# --- Grupo 1.01 a 1.05: Trend Cross (Padrão) ---
def s1_01_long(c, p, pp=None): return c['EMA_9'] > c['EMA_21'] and p['EMA_9'] < p['EMA_21'] and trend_bull(c)
def s1_01_short(c, p, pp=None): return c['EMA_9'] < c['EMA_21'] and p['EMA_9'] > p['EMA_21'] and trend_bear(c)
def s1_01_exit(c, p, s, pp=None): return c['EMA_9'] < c['EMA_21'] if s == 'long' else c['EMA_9'] > c['EMA_21']
def s1_02_long(c, p, pp=None): return c['EMA_12'] > c['EMA_34'] and p['EMA_12'] < p['EMA_34'] and trend_bull(c)
def s1_02_short(c, p, pp=None): return c['EMA_12'] < c['EMA_34'] and p['EMA_12'] > p['EMA_34'] and trend_bear(c)
def s1_02_exit(c, p, s, pp=None): return c['EMA_12'] < c['EMA_34'] if s == 'long' else c['EMA_12'] > c['EMA_34']
def s1_03_long(c, p, pp=None): return c['EMA_21'] > c['EMA_50'] and p['EMA_21'] < p['EMA_50'] and trend_bull(c)
def s1_03_short(c, p, pp=None): return c['EMA_21'] < c['EMA_50'] and p['EMA_21'] > p['EMA_50'] and trend_bear(c)
def s1_03_exit(c, p, s, pp=None): return c['EMA_21'] < c['EMA_50'] if s == 'long' else c['EMA_21'] > c['EMA_50']
def s1_04_long(c, p, pp=None): return c['EMA_50'] > c['EMA_100'] and p['EMA_50'] < p['EMA_100'] and trend_bull(c)
def s1_04_short(c, p, pp=None): return c['EMA_50'] < c['EMA_100'] and p['EMA_50'] > p['EMA_100'] and trend_bear(c)
def s1_04_exit(c, p, s, pp=None): return c['EMA_50'] < c['EMA_100'] if s == 'long' else c['EMA_50'] > c['EMA_100']
def s1_05_long(c, p, pp=None): return c['EMA_50'] > c['EMA_200'] and p['EMA_50'] < p['EMA_200']
def s1_05_short(c, p, pp=None): return c['EMA_50'] < c['EMA_200'] and p['EMA_50'] > p['EMA_200']
def s1_05_exit(c, p, s, pp=None): return c['EMA_50'] < c['EMA_200'] if s == 'long' else c['EMA_50'] > c['EMA_200']
# --- Grupo 1.06 a 1.10: Trend Cross + ADX Filter ---
def s1_06_long(c, p, pp=None): return c['EMA_9'] > c['EMA_21'] and p['EMA_9'] < p['EMA_21'] and trend_bull(
    c) and adx_ok(c)
def s1_06_short(c, p, pp=None): return c['EMA_9'] < c['EMA_21'] and p['EMA_9'] > p['EMA_21'] and trend_bear(
    c) and adx_ok(c)
def s1_06_exit(c, p, s, pp=None): return c['EMA_9'] < c['EMA_21'] if s == 'long' else c['EMA_9'] > c['EMA_21']
def s1_07_long(c, p, pp=None): return c['EMA_12'] > c['EMA_34'] and p['EMA_12'] < p['EMA_34'] and trend_bull(
    c) and adx_ok(c)
def s1_07_short(c, p, pp=None): return c['EMA_12'] < c['EMA_34'] and p['EMA_12'] > p['EMA_34'] and trend_bear(
    c) and adx_ok(c)
def s1_07_exit(c, p, s, pp=None): return c['EMA_12'] < c['EMA_34'] if s == 'long' else c['EMA_12'] > c['EMA_34']
def s1_08_long(c, p, pp=None): return c['EMA_21'] > c['EMA_50'] and p['EMA_21'] < p['EMA_50'] and trend_bull(
    c) and adx_ok(c)
def s1_08_short(c, p, pp=None): return c['EMA_21'] < c['EMA_50'] and p['EMA_21'] > p['EMA_50'] and trend_bear(
    c) and adx_ok(c)
def s1_08_exit(c, p, s, pp=None): return c['EMA_21'] < c['EMA_50'] if s == 'long' else c['EMA_21'] > c['EMA_50']
def s1_09_long(c, p, pp=None): return c['EMA_50'] > c['EMA_100'] and p['EMA_50'] < p['EMA_100'] and trend_bull(
    c) and adx_ok(c)
def s1_09_short(c, p, pp=None): return c['EMA_50'] < c['EMA_100'] and p['EMA_50'] > p['EMA_100'] and trend_bear(
    c) and adx_ok(c)
def s1_09_exit(c, p, s, pp=None): return c['EMA_50'] < c['EMA_100'] if s == 'long' else c['EMA_50'] > c['EMA_100']
def s1_10_long(c, p, pp=None): return c['EMA_50'] > c['EMA_200'] and p['EMA_50'] < p['EMA_200'] and adx_ok(c)
def s1_10_short(c, p, pp=None): return c['EMA_50'] < c['EMA_200'] and p['EMA_50'] > p['EMA_200'] and adx_ok(c)
def s1_10_exit(c, p, s, pp=None): return c['EMA_50'] < c['EMA_200'] if s == 'long' else c['EMA_50'] > c['EMA_200']
# --- Grupo 1.11 a 1.15: Pullbacks ---
def s1_11_long(c, p, pp=None): return trend_bull(c) and p['low'] <= p['EMA_21'] and c['close'] > c['EMA_21']
def s1_11_short(c, p, pp=None): return trend_bear(c) and p['high'] >= p['EMA_21'] and c['close'] < c['EMA_21']
def s1_11_exit(c, p, s, pp=None): return c['close'] < c['EMA_21'] if s == 'long' else c['close'] > c['EMA_21']
def s1_12_long(c, p, pp=None): return trend_bull(c) and p['low'] <= p['EMA_34'] and c['close'] > c['EMA_34']
def s1_12_short(c, p, pp=None): return trend_bear(c) and p['high'] >= p['EMA_34'] and c['close'] < c['EMA_34']
def s1_12_exit(c, p, s, pp=None): return c['close'] < c['EMA_34'] if s == 'long' else c['close'] > c['EMA_34']
def s1_13_long(c, p, pp=None): return trend_bull(c) and p['low'] <= p['EMA_50'] and c['close'] > c['EMA_50']
def s1_13_short(c, p, pp=None): return trend_bear(c) and p['high'] >= p['EMA_50'] and c['close'] < c['EMA_50']
def s1_13_exit(c, p, s, pp=None): return c['close'] < c['EMA_50'] if s == 'long' else c['close'] > c['EMA_50']
def s1_14_long(c, p, pp=None): return trend_bull(c) and p['low'] <= p['EMA_100'] and c['close'] > c['EMA_100']
def s1_14_short(c, p, pp=None): return trend_bear(c) and p['high'] >= p['EMA_100'] and c['close'] < c['EMA_100']
def s1_14_exit(c, p, s, pp=None): return c['close'] < c['EMA_100'] if s == 'long' else c['close'] > c['EMA_100']
def s1_15_long(c, p, pp=None): return trend_bull(c) and p['low'] <= p['EMA_200'] and c['close'] > c['EMA_200']
def s1_15_short(c, p, pp=None): return trend_bear(c) and p['high'] >= p['EMA_200'] and c['close'] < c['EMA_200']
def s1_15_exit(c, p, s, pp=None): return c['close'] < c['EMA_200'] if s == 'long' else c['close'] > c['EMA_200']
# --- Grupo 1.16 a 1.20: Pullback + RSI Oversold/Overbought ---
def s1_16_long(c, p, pp=None): return trend_bull(c) and c['close'] > c['EMA_21'] and c['RSI_14'] < 50
def s1_16_short(c, p, pp=None): return trend_bear(c) and c['close'] < c['EMA_21'] and c['RSI_14'] > 50
def s1_16_exit(c, p, s, pp=None): return c['RSI_14'] > 70 if s == 'long' else c['RSI_14'] < 30
def s1_17_long(c, p, pp=None): return trend_bull(c) and c['close'] > c['EMA_34'] and c['RSI_14'] < 50
def s1_17_short(c, p, pp=None): return trend_bear(c) and c['close'] < c['EMA_34'] and c['RSI_14'] > 50
def s1_17_exit(c, p, s, pp=None): return c['RSI_14'] > 70 if s == 'long' else c['RSI_14'] < 30
def s1_18_long(c, p, pp=None): return trend_bull(c) and c['close'] > c['EMA_50'] and c['RSI_14'] < 50
def s1_18_short(c, p, pp=None): return trend_bear(c) and c['close'] < c['EMA_50'] and c['RSI_14'] > 50
def s1_18_exit(c, p, s, pp=None): return c['RSI_14'] > 70 if s == 'long' else c['RSI_14'] < 30
def s1_19_long(c, p, pp=None): return trend_bull(c) and c['close'] > c['EMA_100'] and c['RSI_14'] < 50
def s1_19_short(c, p, pp=None): return trend_bear(c) and c['close'] < c['EMA_100'] and c['RSI_14'] > 50
def s1_19_exit(c, p, s, pp=None): return c['RSI_14'] > 70 if s == 'long' else c['RSI_14'] < 30
def s1_20_long(c, p, pp=None): return trend_bull(c) and c['close'] > c['EMA_200'] and c['RSI_14'] < 50
def s1_20_short(c, p, pp=None): return trend_bear(c) and c['close'] < c['EMA_200'] and c['RSI_14'] > 50
def s1_20_exit(c, p, s, pp=None): return c['RSI_14'] > 70 if s == 'long' else c['RSI_14'] < 30
# --- Grupo 1.21 a 1.30: Estratégias Únicas de Tendência ---
def s1_21_long(c, p, pp=None): return c['EMA_9'] > c['EMA_21'] > c['EMA_50'] and c['close'] > c['EMA_9']
def s1_21_short(c, p, pp=None): return c['EMA_9'] < c['EMA_21'] < c['EMA_50'] and c['close'] < c['EMA_9']
def s1_21_exit(c, p, s, pp=None): return c['close'] < c['EMA_21'] if s == 'long' else c['close'] > c['EMA_21']
def s1_22_long(c, p, pp=None): return c['MACD'] > 0 and p['MACD'] < 0 and trend_bull(c)
def s1_22_short(c, p, pp=None): return c['MACD'] < 0 and p['MACD'] > 0 and trend_bear(c)
def s1_22_exit(c, p, s, pp=None): return c['MACD'] < c['MACD_signal'] if s == 'long' else c['MACD'] > c['MACD_signal']
def s1_23_long(c, p, pp=None): return c['P_DI'] > c['M_DI'] and p['P_DI'] < p['M_DI'] and adx_ok(c)
def s1_23_short(c, p, pp=None): return c['M_DI'] > c['P_DI'] and p['M_DI'] < p['P_DI'] and adx_ok(c)
def s1_23_exit(c, p, s, pp=None): return c['P_DI'] < c['M_DI'] if s == 'long' else c['M_DI'] < c['P_DI']
def s1_24_long(c, p, pp=None): return c['ADX'] > 30 and c['ADX'] > p['ADX'] and trend_bull(c) and c['P_DI'] > c['M_DI']
def s1_24_short(c, p, pp=None): return c['ADX'] > 30 and c['ADX'] > p['ADX'] and trend_bear(c) and c['M_DI'] > c['P_DI']
def s1_24_exit(c, p, s, pp=None): return c['ADX'] < p['ADX']
def s1_25_long(c, p, pp=None): return c['MACD_hist'] > p['MACD_hist'] and p['MACD_hist'] < p[
    'MACD_hist_prev'] and trend_bull(c)
def s1_25_short(c, p, pp=None): return c['MACD_hist'] < p['MACD_hist'] and p['MACD_hist'] > p[
    'MACD_hist_prev'] and trend_bear(c)
def s1_25_exit(c, p, s, pp=None): return c['MACD_hist'] < 0 if s == 'long' else c['MACD_hist'] > 0
def s1_26_long(c, p, pp=None): return c['ADX'] > 25 and c['RSI_14'] > 55 and trend_bull(c)
def s1_26_short(c, p, pp=None): return c['ADX'] > 25 and c['RSI_14'] < 45 and trend_bear(c)
def s1_26_exit(c, p, s, pp=None): return c['RSI_14'] < 50 if s == 'long' else c['RSI_14'] > 50
def s1_27_long(c, p, pp=None): return c['MACD'] > c['MACD_signal'] and p['MACD'] < p['MACD_signal'] and c['MACD'] > 0
def s1_27_short(c, p, pp=None): return c['MACD'] < c['MACD_signal'] and p['MACD'] > p['MACD_signal'] and c['MACD'] < 0
def s1_27_exit(c, p, s, pp=None): return c['MACD'] < c['MACD_signal'] if s == 'long' else c['MACD'] > c['MACD_signal']
def s1_28_long(c, p, pp=None): return c['ADX'] > 40 and c['ADX'] < p['ADX'] and trend_bull(c) and c['close'] > c[
    'EMA_21']
def s1_28_short(c, p, pp=None): return c['ADX'] > 40 and c['ADX'] < p['ADX'] and trend_bear(c) and c['close'] < c[
    'EMA_21']
def s1_28_exit(c, p, s, pp=None): return c['close'] < c['EMA_50'] if s == 'long' else c['close'] > c['EMA_50']
def s1_29_long(c, p, pp=None): return c['P_DI'] > 30 and p['P_DI'] < 30 and trend_bull(c)
def s1_29_short(c, p, pp=None): return c['M_DI'] > 30 and p['M_DI'] < 30 and trend_bear(c)
def s1_29_exit(c, p, s, pp=None): return c['P_DI'] < 20 if s == 'long' else c['M_DI'] < 20
def s1_30_long(c, p, pp=None): return c['ADX'] > 30 and c['low'] <= c['EMA_21'] and c['close'] > c[
    'EMA_21'] and trend_bull(c)
def s1_30_short(c, p, pp=None): return c['ADX'] > 30 and c['high'] >= c['EMA_21'] and c['close'] < c[
    'EMA_21'] and trend_bear(c)
def s1_30_exit(c, p, s, pp=None): return c['close'] > c['BB_upper'] if s == 'long' else c['close'] < c['BB_lower']
# ------------------------------------------------------------------------------
# 2. MOMENTUM (30 ESTRATÉGIAS)
# ------------------------------------------------------------------------------
# --- 2.01 a 2.05: RSI Reversal (Sobrevenda/Sobrecompra simples) ---
def s2_01_long(c, p, pp=None): return c['RSI_14'] > 30 and p['RSI_14'] < 30
def s2_01_short(c, p, pp=None): return c['RSI_14'] < 70 and p['RSI_14'] > 70
def s2_01_exit(c, p, s, pp=None): return c['RSI_14'] > 50 if s == 'long' else c['RSI_14'] < 50
def s2_02_long(c, p, pp=None): return c['RSI_14'] > 25 and p['RSI_14'] < 25
def s2_02_short(c, p, pp=None): return c['RSI_14'] < 75 and p['RSI_14'] > 75
def s2_02_exit(c, p, s, pp=None): return c['RSI_14'] > 50 if s == 'long' else c['RSI_14'] < 50
def s2_03_long(c, p, pp=None): return c['RSI_14'] > 20 and p['RSI_14'] < 20
def s2_03_short(c, p, pp=None): return c['RSI_14'] < 80 and p['RSI_14'] > 80
def s2_03_exit(c, p, s, pp=None): return c['RSI_14'] > 50 if s == 'long' else c['RSI_14'] < 50
def s2_04_long(c, p, pp=None): return c['RSI_14'] > 35 and p['RSI_14'] < 35
def s2_04_short(c, p, pp=None): return c['RSI_14'] < 65 and p['RSI_14'] > 65
def s2_04_exit(c, p, s, pp=None): return c['RSI_14'] > 50 if s == 'long' else c['RSI_14'] < 50
def s2_05_long(c, p, pp=None): return c['RSI_14'] > 40 and p['RSI_14'] < 40
def s2_05_short(c, p, pp=None): return c['RSI_14'] < 60 and p['RSI_14'] > 60
def s2_05_exit(c, p, s, pp=None): return c['RSI_14'] > 50 if s == 'long' else c['RSI_14'] < 50
# --- 2.06 a 2.10: RSI Trend Dip (Entra a favor da tendência com RSI baixo) ---
def s2_06_long(c, p, pp=None): return trend_bull(c) and c['RSI_14'] < 45 and c['RSI_14'] > 30
def s2_06_short(c, p, pp=None): return trend_bear(c) and c['RSI_14'] > 55 and c['RSI_14'] < 70
def s2_06_exit(c, p, s, pp=None): return c['RSI_14'] > 60 if s == 'long' else c['RSI_14'] < 40
def s2_07_long(c, p, pp=None): return trend_bull(c) and c['RSI_14'] < 45 and c['RSI_14'] > 25
def s2_07_short(c, p, pp=None): return trend_bear(c) and c['RSI_14'] > 55 and c['RSI_14'] < 75
def s2_07_exit(c, p, s, pp=None): return c['RSI_14'] > 60 if s == 'long' else c['RSI_14'] < 40
def s2_08_long(c, p, pp=None): return trend_bull(c) and c['RSI_14'] < 45 and c['RSI_14'] > 20
def s2_08_short(c, p, pp=None): return trend_bear(c) and c['RSI_14'] > 55 and c['RSI_14'] < 80
def s2_08_exit(c, p, s, pp=None): return c['RSI_14'] > 60 if s == 'long' else c['RSI_14'] < 40
def s2_09_long(c, p, pp=None): return trend_bull(c) and c['RSI_14'] < 45 and c['RSI_14'] > 35
def s2_09_short(c, p, pp=None): return trend_bear(c) and c['RSI_14'] > 55 and c['RSI_14'] < 65
def s2_09_exit(c, p, s, pp=None): return c['RSI_14'] > 60 if s == 'long' else c['RSI_14'] < 40
def s2_10_long(c, p, pp=None): return trend_bull(c) and c['RSI_14'] < 45 and c['RSI_14'] > 40
def s2_10_short(c, p, pp=None): return trend_bear(c) and c['RSI_14'] > 55 and c['RSI_14'] < 60
def s2_10_exit(c, p, s, pp=None): return c['RSI_14'] > 60 if s == 'long' else c['RSI_14'] < 40
# --- 2.11 a 2.15: CCI Breakout (Várias forças) ---
def s2_11_long(c, p, pp=None): return c['CCI'] > 100 and p['CCI'] < 100
def s2_11_short(c, p, pp=None): return c['CCI'] < -100 and p['CCI'] > -100
def s2_11_exit(c, p, s, pp=None): return c['CCI'] < 0 if s == 'long' else c['CCI'] > 0
def s2_12_long(c, p, pp=None): return c['CCI'] > 150 and p['CCI'] < 150
def s2_12_short(c, p, pp=None): return c['CCI'] < -150 and p['CCI'] > -150
def s2_12_exit(c, p, s, pp=None): return c['CCI'] < 0 if s == 'long' else c['CCI'] > 0
def s2_13_long(c, p, pp=None): return c['CCI'] > 200 and p['CCI'] < 200
def s2_13_short(c, p, pp=None): return c['CCI'] < -200 and p['CCI'] > -200
def s2_13_exit(c, p, s, pp=None): return c['CCI'] < 0 if s == 'long' else c['CCI'] > 0
def s2_14_long(c, p, pp=None): return c['CCI'] > 50 and p['CCI'] < 50
def s2_14_short(c, p, pp=None): return c['CCI'] < -50 and p['CCI'] > -50
def s2_14_exit(c, p, s, pp=None): return c['CCI'] < 0 if s == 'long' else c['CCI'] > 0
def s2_15_long(c, p, pp=None): return c['CCI'] > 0 and p['CCI'] < 0
def s2_15_short(c, p, pp=None): return c['CCI'] < 0 and p['CCI'] > 0
# 13. Stoch Headroom: Estocástico K acima de 20.
def s2_15_01_short(data, *args): return s2_15_short(data, *args) and data['STOCH_K'] > 20
def s2_15_exit(c, p, s, pp=None): return c['CCI'] < -50 if s == 'long' else c['CCI'] > 50
# --- 2.16 a 2.20: Williams %R ---
def s2_16_long(c, p, pp=None): return c['WILLR'] > -80 and p['WILLR'] < -80
def s2_16_short(c, p, pp=None): return c['WILLR'] < -20 and p['WILLR'] > -20
def s2_16_exit(c, p, s, pp=None): return c['WILLR'] < -50 if s == 'long' else c['WILLR'] > -50
def s2_17_long(c, p, pp=None): return c['WILLR'] > -90 and p['WILLR'] < -90
def s2_17_short(c, p, pp=None): return c['WILLR'] < -10 and p['WILLR'] > -10
def s2_17_exit(c, p, s, pp=None): return c['WILLR'] < -50 if s == 'long' else c['WILLR'] > -50
def s2_18_long(c, p, pp=None): return c['WILLR'] > -75 and p['WILLR'] < -75
def s2_18_short(c, p, pp=None): return c['WILLR'] < -25 and p['WILLR'] > -25
def s2_18_exit(c, p, s, pp=None): return c['WILLR'] < -50 if s == 'long' else c['WILLR'] > -50
def s2_19_long(c, p, pp=None): return c['WILLR'] > -70 and p['WILLR'] < -70
def s2_19_short(c, p, pp=None): return c['WILLR'] < -30 and p['WILLR'] > -30
def s2_19_exit(c, p, s, pp=None): return c['WILLR'] < -50 if s == 'long' else c['WILLR'] > -50
def s2_20_long(c, p, pp=None): return c['WILLR'] > -50 and p['WILLR'] < -50
def s2_20_short(c, p, pp=None): return c['WILLR'] < -50 and p['WILLR'] > -50
def s2_20_exit(c, p, s, pp=None): return c['WILLR'] < -20 if s == 'long' else c['WILLR'] > -80
# --- 2.21 a 2.30: ROC (Rate of Change) ---
def s2_21_long(c, p, pp=None): return c['ROC'] > 0.2 and p['ROC'] < 0.2 and vol_ok(c)
def s2_21_short(c, p, pp=None): return c['ROC'] < -0.2 and p['ROC'] > -0.2 and vol_ok(c)
def s2_21_exit(c, p, s, pp=None): return c['ROC'] < 0 if s == 'long' else c['ROC'] > 0
def s2_22_long(c, p, pp=None): return c['ROC'] > 0.4 and p['ROC'] < 0.4 and vol_ok(c)
def s2_22_short(c, p, pp=None): return c['ROC'] < -0.4 and p['ROC'] > -0.4 and vol_ok(c)
def s2_22_exit(c, p, s, pp=None): return c['ROC'] < 0 if s == 'long' else c['ROC'] > 0
def s2_23_long(c, p, pp=None): return c['ROC'] > 0.6 and p['ROC'] < 0.6 and vol_ok(c)
def s2_23_short(c, p, pp=None): return c['ROC'] < -0.6 and p['ROC'] > -0.6 and vol_ok(c)
def s2_23_exit(c, p, s, pp=None): return c['ROC'] < 0 if s == 'long' else c['ROC'] > 0
def s2_24_long(c, p, pp=None): return c['ROC'] > 0.8 and p['ROC'] < 0.8 and vol_ok(c)
def s2_24_short(c, p, pp=None): return c['ROC'] < -0.8 and p['ROC'] > -0.8 and vol_ok(c)
def s2_24_exit(c, p, s, pp=None): return c['ROC'] < 0 if s == 'long' else c['ROC'] > 0
def s2_25_long(c, p, pp=None): return c['ROC'] > 1.0 and p['ROC'] < 1.0 and vol_ok(c)
def s2_25_short(c, p, pp=None): return c['ROC'] < -1.0 and p['ROC'] > -1.0 and vol_ok(c)
def s2_25_exit(c, p, s, pp=None): return c['ROC'] < 0 if s == 'long' else c['ROC'] > 0
def s2_26_long(c, p, pp=None): return c['ROC'] > 1.2 and p['ROC'] < 1.2 and vol_ok(c)
def s2_26_short(c, p, pp=None): return c['ROC'] < -1.2 and p['ROC'] > -1.2 and vol_ok(c)
def s2_26_exit(c, p, s, pp=None): return c['ROC'] < 0 if s == 'long' else c['ROC'] > 0
def s2_27_long(c, p, pp=None): return c['ROC'] > 1.4 and p['ROC'] < 1.4 and vol_ok(c)
def s2_27_short(c, p, pp=None): return c['ROC'] < -1.4 and p['ROC'] > -1.4 and vol_ok(c)
def s2_27_exit(c, p, s, pp=None): return c['ROC'] < 0 if s == 'long' else c['ROC'] > 0
def s2_28_long(c, p, pp=None): return c['ROC'] > 1.6 and p['ROC'] < 1.6 and vol_ok(c)
def s2_28_short(c, p, pp=None): return c['ROC'] < -1.6 and p['ROC'] > -1.6 and vol_ok(c)
def s2_28_exit(c, p, s, pp=None): return c['ROC'] < 0 if s == 'long' else c['ROC'] > 0
def s2_29_long(c, p, pp=None): return c['ROC'] > 1.8 and p['ROC'] < 1.8 and vol_ok(c)
def s2_29_short(c, p, pp=None): return c['ROC'] < -1.8 and p['ROC'] > -1.8 and vol_ok(c)
def s2_29_exit(c, p, s, pp=None): return c['ROC'] < 0 if s == 'long' else c['ROC'] > 0
def s2_30_long(c, p, pp=None): return c['ROC'] > 2.0 and p['ROC'] < 2.0 and vol_ok(c)
def s2_30_short(c, p, pp=None): return c['ROC'] < -2.0 and p['ROC'] > -2.0 and vol_ok(c)
def s2_30_exit(c, p, s, pp=None): return c['ROC'] < 0 if s == 'long' else c['ROC'] > 0
# ------------------------------------------------------------------------------
# 3. BREAKOUT (30 ESTRATÉGIAS)
# ------------------------------------------------------------------------------
# --- 3.01 a 3.10: BB Breakout ---
def s3_01_long(c, p, pp=None): return c['close'] > c['BB_upper'] and c['BBW'] > p['BBW']
def s3_01_short(c, p, pp=None): return c['close'] < c['BB_lower'] and c['BBW'] > p['BBW']
def s3_01_exit(c, p, s, pp=None): return c['close'] < c['BB_middle'] if s == 'long' else c['close'] > c['BB_middle']
def s3_02_long(c, p, pp=None): return c['close'] > c['BB_upper'] and c['BBW'] > 0.10
def s3_02_short(c, p, pp=None): return c['close'] < c['BB_lower'] and c['BBW'] > 0.10
def s3_02_exit(c, p, s, pp=None): return c['close'] < c['BB_middle'] if s == 'long' else c['close'] > c['BB_middle']
def s3_03_long(c, p, pp=None): return c['close'] > c['BB_upper'] and c['Vol_Ratio'] > 1.2
def s3_03_short(c, p, pp=None): return c['close'] < c['BB_lower'] and c['Vol_Ratio'] > 1.2
def s3_03_exit(c, p, s, pp=None): return c['close'] < c['BB_middle'] if s == 'long' else c['close'] > c['BB_middle']
def s3_04_long(c, p, pp=None): return c['close'] > c['BB_upper'] and c['ADX'] > 25
def s3_04_short(c, p, pp=None): return c['close'] < c['BB_lower'] and c['ADX'] > 25
def s3_04_exit(c, p, s, pp=None): return c['close'] < c['BB_middle'] if s == 'long' else c['close'] > c['BB_middle']
def s3_05_long(c, p, pp=None): return c['close'] > c['BB_upper'] and p['close'] < c['BB_upper']
def s3_05_short(c, p, pp=None): return c['close'] < c['BB_lower'] and p['close'] > c['BB_lower']
def s3_05_exit(c, p, s, pp=None): return c['close'] < c['EMA_9'] if s == 'long' else c['close'] > c['EMA_9']
def s3_06_long(c, p, pp=None): return c['close'] > c['BB_upper'] and c['RSI_14'] > 60
def s3_06_short(c, p, pp=None): return c['close'] < c['BB_lower'] and c['RSI_14'] < 40
def s3_06_exit(c, p, s, pp=None): return c['close'] < c['EMA_9'] if s == 'long' else c['close'] > c['EMA_9']
def s3_07_long(c, p, pp=None): return c['close'] > c['BB_upper'] and c['CCI'] > 100
def s3_07_short(c, p, pp=None): return c['close'] < c['BB_lower'] and c['CCI'] < -100
def s3_07_exit(c, p, s, pp=None): return c['close'] < c['EMA_21'] if s == 'long' else c['close'] > c['EMA_21']
def s3_08_long(c, p, pp=None): return c['close'] > c['BB_upper'] and c['MACD'] > 0
def s3_08_short(c, p, pp=None): return c['close'] < c['BB_lower'] and c['MACD'] < 0
def s3_08_exit(c, p, s, pp=None): return c['close'] < c['BB_middle'] if s == 'long' else c['close'] > c['BB_middle']
def s3_09_long(c, p, pp=None): return c['close'] > c['BB_upper'] and c['ROC'] > 0.5
def s3_09_short(c, p, pp=None): return c['close'] < c['BB_lower'] and c['ROC'] < -0.5
def s3_09_exit(c, p, s, pp=None): return c['close'] < c['EMA_9'] if s == 'long' else c['close'] > c['EMA_9']
def s3_10_long(c, p, pp=None): return c['close'] > c['BB_upper'] and c['high'] > p['high']
def s3_10_short(c, p, pp=None): return c['close'] < c['BB_lower'] and c['low'] < p['low']
def s3_10_exit(c, p, s, pp=None): return c['close'] < c['BB_middle'] if s == 'long' else c['close'] > c['BB_middle']
# --- 3.11 a 3.20: Channel Breakout (Donchian & Keltner) ---
def s3_11_long(c, p, pp=None): return c['close'] > c['DC_high']
def s3_11_short(c, p, pp=None): return c['close'] < c['DC_low']
def s3_11_exit(c, p, s, pp=None): return c['close'] < c['EMA_21'] if s == 'long' else c['close'] > c['EMA_21']
def s3_12_long(c, p, pp=None): return c['close'] > c['KC_upper']
def s3_12_short(c, p, pp=None): return c['close'] < c['KC_lower']
def s3_12_exit(c, p, s, pp=None): return c['close'] < c['EMA_21'] if s == 'long' else c['close'] > c['EMA_21']
def s3_13_long(c, p, pp=None): return c['close'] > c['DC_high'] and vol_ok(c)
def s3_13_short(c, p, pp=None): return c['close'] < c['DC_low'] and vol_ok(c)
def s3_13_exit(c, p, s, pp=None): return c['close'] < c['EMA_21'] if s == 'long' else c['close'] > c['EMA_21']
def s3_14_long(c, p, pp=None): return c['close'] > c['KC_upper'] and adx_ok(c)
def s3_14_short(c, p, pp=None): return c['close'] < c['KC_lower'] and adx_ok(c)
def s3_14_exit(c, p, s, pp=None): return c['close'] < c['EMA_21'] if s == 'long' else c['close'] > c['EMA_21']
def s3_15_long(c, p, pp=None): return c['close'] > c['DC_high'] and c['RSI_14'] > 55
def s3_15_short(c, p, pp=None): return c['close'] < c['DC_low'] and c['RSI_14'] < 45
def s3_15_exit(c, p, s, pp=None): return c['close'] < c['EMA_21'] if s == 'long' else c['close'] > c['EMA_21']
def s3_16_long(c, p, pp=None): return c['close'] > c['KC_upper'] and c['MACD'] > c['MACD_signal']
def s3_16_short(c, p, pp=None): return c['close'] < c['KC_lower'] and c['MACD'] < c['MACD_signal']
def s3_16_exit(c, p, s, pp=None): return c['close'] < c['EMA_21'] if s == 'long' else c['close'] > c['EMA_21']
def s3_17_long(c, p, pp=None): return c['close'] > c['DC_high'] and c['close'] > c['BB_upper']
def s3_17_short(c, p, pp=None): return c['close'] < c['DC_low'] and c['close'] < c['BB_lower']
def s3_17_exit(c, p, s, pp=None): return c['close'] < c['EMA_21'] if s == 'long' else c['close'] > c['EMA_21']
def s3_18_long(c, p, pp=None): return c['close'] > c['KC_upper'] and c['BBW'] > p['BBW']
def s3_18_short(c, p, pp=None): return c['close'] < c['KC_lower'] and c['BBW'] > p['BBW']
def s3_18_exit(c, p, s, pp=None): return c['close'] < c['EMA_21'] if s == 'long' else c['close'] > c['EMA_21']
def s3_19_long(c, p, pp=None): return c['close'] > c['DC_high'] and c['high'] > p['DC_high']
def s3_19_short(c, p, pp=None): return c['close'] < c['DC_low'] and c['low'] < p['DC_low']
def s3_19_exit(c, p, s, pp=None): return c['close'] < c['EMA_21'] if s == 'long' else c['close'] > c['EMA_21']
def s3_20_long(c, p, pp=None): return c['close'] > c['KC_upper'] and c['CCI'] > 100
def s3_20_short(c, p, pp=None): return c['close'] < c['KC_lower'] and c['CCI'] < -100
def s3_20_exit(c, p, s, pp=None): return c['close'] < c['EMA_21'] if s == 'long' else c['close'] > c['EMA_21']
# --- 3.21 a 3.30: Volatility Surge (ATR Multiples) ---
def s3_21_long(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 1.5) and c['close'] > c['open']
def s3_21_short(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 1.5) and c['close'] < c['open']
def s3_21_exit(c, p, s, pp=None): return c['close'] < c['EMA_9'] if s == 'long' else c['close'] > c['EMA_9']
def s3_22_long(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 1.6) and c['close'] > c['open']
def s3_22_short(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 1.6) and c['close'] < c['open']
def s3_22_exit(c, p, s, pp=None): return c['close'] < c['EMA_9'] if s == 'long' else c['close'] > c['EMA_9']
def s3_23_long(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 1.7) and c['close'] > c['open']
def s3_23_short(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 1.7) and c['close'] < c['open']
def s3_23_exit(c, p, s, pp=None): return c['close'] < c['EMA_9'] if s == 'long' else c['close'] > c['EMA_9']
def s3_24_long(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 1.8) and c['close'] > c['open']
def s3_24_short(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 1.8) and c['close'] < c['open']
def s3_24_exit(c, p, s, pp=None): return c['close'] < c['EMA_9'] if s == 'long' else c['close'] > c['EMA_9']
def s3_24_01_long(data, *args): return s3_24_long(data, *args) and data['EMA_50'] > data['EMA_200']
def s3_25_long(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 1.9) and c['close'] > c['open']
def s3_25_short(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 1.9) and c['close'] < c['open']
def s3_25_exit(c, p, s, pp=None): return c['close'] < c['EMA_9'] if s == 'long' else c['close'] > c['EMA_9']
def s3_26_long(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 2.0) and c['close'] > c['open']
def s3_26_short(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 2.0) and c['close'] < c['open']
def s3_26_exit(c, p, s, pp=None): return c['close'] < c['EMA_9'] if s == 'long' else c['close'] > c['EMA_9']
def s3_26_01_long(data, *args): return s3_26_long(data, *args) and data['ADX'] > 20
def s3_27_long(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 2.1) and c['close'] > c['open']
def s3_27_short(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 2.1) and c['close'] < c['open']
def s3_27_exit(c, p, s, pp=None): return c['close'] < c['EMA_9'] if s == 'long' else c['close'] > c['EMA_9']
def s3_28_long(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 2.2) and c['close'] > c['open']
def s3_28_short(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 2.2) and c['close'] < c['open']
def s3_28_exit(c, p, s, pp=None): return c['close'] < c['EMA_9'] if s == 'long' else c['close'] > c['EMA_9']
def s3_29_long(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 2.3) and c['close'] > c['open']
def s3_29_short(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 2.3) and c['close'] < c['open']
def s3_29_exit(c, p, s, pp=None): return c['close'] < c['EMA_9'] if s == 'long' else c['close'] > c['EMA_9']
def s3_30_long(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 2.5) and c['close'] > c['open']
def s3_30_short(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 2.5) and c['close'] < c['open']
def s3_30_exit(c, p, s, pp=None): return c['close'] < c['EMA_9'] if s == 'long' else c['close'] > c['EMA_9']
# ============================================================
# 1. TENDÊNCIA (1.31 - 1.40)
# ============================================================
def s1_31_long(c, p, pp=None): return c['close'] < c['EMA_9'] and p['close'] < p['EMA_9'] and c['MACD'] > c[
    'MACD_signal']
def s1_31_short(c, p, pp=None): return c['close'] > c['EMA_9'] and p['close'] > p['EMA_9'] and c['MACD'] < c[
    'MACD_signal']
def s1_31_exit(c, p, s, pp=None): return (c['close'] > c['EMA_21']) if s == 'long' else (c['close'] < c['EMA_21'])
def s1_32_long(c, p, pp=None): return c['close'] > c['DC_high'] and c['Vol_Ratio'] > 1.1 and c['close'] > c['EMA_9']
def s1_32_short(c, p, pp=None): return c['close'] < c['DC_low'] and c['Vol_Ratio'] > 1.1 and c['close'] < c['EMA_9']
def s1_32_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])
def s1_33_long(c, p, pp=None): return (c['low'] < p['low']) and (c['RSI_14'] > p['RSI_14']) and trend_bull(c)
def s1_33_short(c, p, pp=None): return (c['high'] > p['high']) and (c['RSI_14'] < p['RSI_14']) and trend_bear(c)
def s1_33_exit(c, p, s, pp=None): return (c['RSI_14'] > 60) if s == 'long' else (c['RSI_14'] < 40)
def s1_34_long(c, p, pp=None): return c['close'] > c['BB_upper'] and c['Vol_Ratio'] > 1.1 and c['close'] > c['EMA_9']
def s1_34_short(c, p, pp=None): return c['close'] < c['BB_lower'] and c['Vol_Ratio'] > 1.1 and c['close'] < c['EMA_9']
def s1_34_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])
def s1_35_long(c, p, pp=None): return (c['close'] > c['open']) and (p['close'] < p['open']) and c['low'] > c[
    'EMA_21'] and trend_bull(c)
def s1_35_short(c, p, pp=None): return (c['close'] < c['open']) and (p['close'] > p['open']) and c['high'] < c[
    'EMA_21'] and trend_bear(c)
def s1_35_exit(c, p, s, pp=None): return (c['close'] < c['EMA_9']) if s == 'long' else (c['close'] > c['EMA_9'])
def s1_36_long(c, p, pp=None): return (c['high'] <= p['high'] and c['low'] >= p['low']) and trend_bull(c) and c[
    'close'] > c['EMA_9']
def s1_36_short(c, p, pp=None): return (c['high'] <= p['high'] and c['low'] >= p['low']) and trend_bear(c) and c[
    'close'] < c['EMA_9']
def s1_36_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])
def s1_37_long(c, p, pp=None): return (
        c['close'] > c['open'] and (c['high'] - c['close']) / max(1e-9, c['high'] - c['low']) > 0.6) and c['low'] > \
    c['EMA_21']
def s1_37_short(c, p, pp=None): return (
        c['close'] < c['open'] and (c['close'] - c['low']) / max(1e-9, c['high'] - c['low']) > 0.6) and c['high'] < \
    c['EMA_21']
def s1_37_exit(c, p, s, pp=None): return (c['close'] < c['EMA_9']) if s == 'long' else (c['close'] > c['EMA_9'])
# ------- BTC FAVORABLE TREND STRATEGIES -------
def s1_38_long(c, p, pp=None): return trend_bull(c) and c['low'] <= c['EMA_21'] and c['MACD_hist'] > p['MACD_hist']
def s1_38_short(c, p, pp=None): return trend_bear(c) and c['high'] >= c['EMA_21'] and c['MACD_hist'] < p['MACD_hist']
def s1_38_exit(c, p, s, pp=None): return (c['close'] < c['EMA_9']) if s == 'long' else (c['close'] > c['EMA_9'])
def s1_39_long(c, p, pp=None): return trend_bull(c) and abs(c['close'] - c['EMA_34']) / c['close'] < 0.002 and c[
    'ADX'] > 18
def s1_39_short(c, p, pp=None): return trend_bear(c) and abs(c['close'] - c['EMA_34']) / c['close'] < 0.002 and c[
    'ADX'] > 18
def s1_39_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])
def s1_40_long(c, p, pp=None): return trend_bull(c) and p['close'] < p['BB_middle'] and c['close'] > c['BB_middle']
def s1_40_short(c, p, pp=None): return trend_bear(c) and p['close'] > p['BB_middle'] and c['close'] < c['BB_middle']
def s1_40_02_short(data, *args): return s1_40_short(data, *args) and ((data['EMA_100'] - data['close']) / data['close']) < 0.03
def s1_40_exit(c, p, s, pp=None): return (c['close'] < c['EMA_9']) if s == 'long' else (c['close'] > c['EMA_9'])
# ============================================================
# 2. MOMENTUM (2.31 - 2.41)
# ============================================================
def s2_31_long(c, p, pp=None): return c['close'] < c['BB_lower'] and c['MACD_hist'] > p['MACD_hist'] and c[
    'RSI_14'] < 40
def s2_31_short(c, p, pp=None): return c['close'] > c['BB_upper'] and c['MACD_hist'] < p['MACD_hist'] and c[
    'RSI_14'] > 60
def s2_31_exit(c, p, s, pp=None): return (c['close'] > c['BB_middle']) if s == 'long' else (c['close'] < c['BB_middle'])
def s2_32_long(c, p, pp=None): return c['close'] < c['DC_low'] and c['WILLR'] > p['WILLR'] and c['CCI'] > -200
def s2_32_short(c, p, pp=None): return c['close'] > c['DC_high'] and c['WILLR'] < p['WILLR'] and c['CCI'] < 200
def s2_32_exit(c, p, s, pp=None): return (c['close'] > c['EMA_9']) if s == 'long' else (c['close'] < c['EMA_9'])
def s2_33_long(c, p, pp=None): return c['close'] < c['BB_lower'] and c['Vol_Ratio'] > 0.9 and c['MACD_hist'] > 0
def s2_33_short(c, p, pp=None): return c['close'] > c['BB_upper'] and c['Vol_Ratio'] > 0.9 and c['MACD_hist'] < 0
def s2_33_exit(c, p, s, pp=None): return (c['RSI_14'] > 50) if s == 'long' else (c['RSI_14'] < 50)
def s2_34_long(c, p, pp=None): return c['close'] < c['BB_lower'] and p['close'] < p['BB_lower'] and c['close'] > p[
    'close']
def s2_34_short(c, p, pp=None): return c['close'] > c['BB_upper'] and p['close'] > p['BB_upper'] and c['close'] < p[
    'close']
def s2_34_exit(c, p, s, pp=None): return (c['close'] > c['BB_middle']) if s == 'long' else (c['close'] < c['BB_middle'])
def s2_35_long(c, p, pp=None): return (c['low'] < p['low']) and (c['MACD_hist'] > p['MACD_hist']) and c['RSI_14'] > p[
    'RSI_14']
def s2_35_short(c, p, pp=None): return (c['high'] > p['high']) and (c['MACD_hist'] < p['MACD_hist']) and c['RSI_14'] < \
    p['RSI_14']
def s2_35_exit(c, p, s, pp=None): return (c['MACD_hist'] < 0) if s == 'long' else (c['MACD_hist'] > 0)
def s2_36_long(c, p, pp=None): return (c['low'] < p['low']) and (c['CCI'] > p['CCI']) and c['Vol_Ratio'] > 0.9
def s2_36_short(c, p, pp=None): return (c['high'] > p['high']) and (c['CCI'] < p['CCI']) and c['Vol_Ratio'] > 0.9
def s2_36_exit(c, p, s, pp=None): return (c['close'] > c['EMA_9']) if s == 'long' else (c['close'] < c['EMA_9'])
def s2_37_long(c, p, pp=None): return (c['low'] < p['low']) and (c['ROC'] > p['ROC']) and c['MACD_hist'] > 0
def s2_37_short(c, p, pp=None): return (c['high'] > p['high']) and (c['ROC'] < p['ROC']) and c['MACD_hist'] < 0
def s2_37_exit(c, p, s, pp=None): return (c['ROC'] < 0) if s == 'long' else (c['ROC'] > 0)
# ------- BTC FAVORABLE MOMENTUM STRATEGIES -------
def s2_38_long(c, p, pp=None): return c['RSI_14'] < 25 and c['MACD_hist'] > p['MACD_hist']
def s2_38_short(c, p, pp=None): return c['RSI_14'] > 75 and c['MACD_hist'] < p['MACD_hist']
def s2_38_exit(c, p, s, pp=None): return (c['RSI_14'] > 45) if s == 'long' else (c['RSI_14'] < 55)
def s2_39_long(c, p, pp=None): return c['ROC'] > 0 and c['ROC'] > p['ROC'] and c['Vol_Ratio'] > 1.1
def s2_39_short(c, p, pp=None): return c['ROC'] < 0 and c['ROC'] < p['ROC'] and c['Vol_Ratio'] > 1.1
def s2_39_exit(c, p, s, pp=None): return (c['ROC'] < 0) if s == 'long' else (c['ROC'] > 0)
def s2_40_long(c, p, pp=None): return c['BBW'] < p['BBW'] and c['MACD_hist'] > 0 and c['Vol_Ratio'] > 1.0
def s2_40_short(c, p, pp=None): return c['BBW'] < p['BBW'] and c['MACD_hist'] < 0 and c['Vol_Ratio'] > 1.0
def s2_40_exit(c, p, s, pp=None): return (c['BBW'] > p['BBW'])
def s2_41_long(c, p, pp=None): return c['CCI'] < -150 and c['CCI'] > p['CCI']
def s2_41_short(c, p, pp=None): return c['CCI'] > 150 and c['CCI'] < p['CCI']
def s2_41_exit(c, p, s, pp=None): return (c['CCI'] > 0) if s == 'long' else (c['CCI'] < 0)
# ============================================================
# 3. BREAKOUT (3.31 - 3.39)
# ============================================================
def s3_31_long(c, p, pp=None): return c['close'] > c['BB_upper'] and p['close'] < p['BB_upper'] and c['low'] <= c[
    'BB_upper']
def s3_31_short(c, p, pp=None): return c['close'] < c['BB_lower'] and p['close'] > p['BB_lower'] and c['high'] >= c[
    'BB_lower']
def s3_31_exit(c, p, s, pp=None): return (c['close'] < c['BB_middle']) if s == 'long' else (c['close'] > c['BB_middle'])
def s3_32_long(c, p, pp=None): return c['close'] > c['BB_upper'] and p['close'] > p['BB_upper'] and c['low'] > p[
    'BB_upper']
def s3_32_short(c, p, pp=None): return c['close'] < c['BB_lower'] and p['close'] < p['BB_lower'] and c['high'] < p[
    'BB_lower']
def s3_32_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])
def s3_33_long(c, p, pp=None): return c['close'] > c['BB_upper'] and c['Vol_Ratio'] > 1.3 and c['ADX'] > 20
def s3_33_short(c, p, pp=None): return c['close'] < c['BB_lower'] and c['Vol_Ratio'] > 1.3 and c['ADX'] > 20
def s3_33_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])
def s3_34_long(c, p, pp=None): return c['low'] > p['low'] and c['Vol_Ratio'] > 1.5 and c['close'] > c['EMA_9']
def s3_34_short(c, p, pp=None): return c['high'] < p['high'] and c['Vol_Ratio'] > 1.5 and c['close'] < c['EMA_9']
def s3_34_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])
def s3_35_long(c, p, pp=None): return c['close'] > c['KC_upper'] and c['Vol_Ratio'] > 1.2 and c['P_DI'] > c['M_DI']
def s3_35_short(c, p, pp=None): return c['close'] < c['KC_lower'] and c['Vol_Ratio'] > 1.2 and c['M_DI'] > c['P_DI']
def s3_35_exit(c, p, s, pp=None): return (c['close'] < c['EMA_9']) if s == 'long' else (c['close'] > c['EMA_9'])
def s3_36_long(c, p, pp=None): return c['close'] > c['DC_high'] and p['close'] < p['DC_high'] and c['low'] <= c[
    'DC_high']
def s3_36_short(c, p, pp=None): return c['close'] < c['DC_low'] and p['close'] > p['DC_low'] and c['high'] >= c[
    'DC_low']
def s3_36_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])
# ------- BTC FAVORABLE BREAKOUT STRATEGIES -------
def s3_37_long(c, p, pp=None): return (c['high'] <= p['high'] and c['low'] >= p['low']) and trend_bull(c) and c[
    'close'] > c['DC_high']
def s3_37_short(c, p, pp=None): return (c['high'] <= p['high'] and c['low'] >= p['low']) and trend_bear(c) and c[
    'close'] < c['DC_low']
def s3_37_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])
def s3_38_long(c, p, pp=None): return c['BBW'] < 0.06 and c['close'] > c['BB_upper']
def s3_38_short(c, p, pp=None): return c['BBW'] < 0.06 and c['close'] < c['BB_lower']
def s3_38_exit(c, p, s, pp=None): return (c['close'] < c['BB_middle']) if s == 'long' else (c['close'] > c['BB_middle'])
def s3_39_long(c, p, pp=None): return c['close'] > c['DC_high'] and c['Vol_Ratio'] > 1.3 and c['P_DI'] > c['M_DI']
def s3_39_short(c, p, pp=None): return c['close'] < c['DC_low'] and c['Vol_Ratio'] > 1.3 and c['M_DI'] > c['P_DI']
def s3_39_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])
# =======================
# 1. TENDÊNCIA (1.41–1.43)
# =======================
def s1_41_long(c, p, pp=None): return trend_bull(c) and c['low'] <= c['VWAP'] and c['STOCH_K'] > c['STOCH_D']
def s1_41_short(c, p, pp=None): return trend_bear(c) and c['high'] >= c['VWAP'] and c['STOCH_K'] < c['STOCH_D']
def s1_41_exit(c, p, s, pp=None): return (c['close'] < c['EMA_9']) if s == 'long' else (c['close'] > c['EMA_9'])
def s1_42_long(c, p, pp=None): return trend_bull(c) and c['close'] < c['VWAP'] and c['STOCH_K'] < 20
def s1_42_short(c, p, pp=None): return trend_bear(c) and c['close'] > c['VWAP'] and c['STOCH_K'] > 80
def s1_42_exit(c, p, s, pp=None): return (c['STOCH_K'] > 50) if s == 'long' else (c['STOCH_K'] < 50)
def s1_43_long(c, p, pp=None): return trend_bull(c) and c['VWAP'] > c['EMA_21'] and c['STOCH_K'] > c['STOCH_D']
def s1_43_short(c, p, pp=None): return trend_bear(c) and c['VWAP'] < c['EMA_21'] and c['STOCH_K'] < c['STOCH_D']
def s1_43_exit(c, p, s, pp=None): return (c['close'] < c['VWAP']) if s == 'long' else (c['close'] > c['VWAP'])
# =======================
# 2. MOMENTUM (2.42–2.44)
# =======================
def s2_42_long(c, p, pp=None): return c['STOCH_K'] > c['STOCH_D'] and p['STOCH_K'] <= p['STOCH_D'] and c['close'] > c[
    'VWAP']
def s2_42_short(c, p, pp=None): return c['STOCH_K'] < c['STOCH_D'] and p['STOCH_K'] >= p['STOCH_D'] and c['close'] < c[
    'VWAP']
def s2_42_exit(c, p, s, pp=None): return (c['STOCH_K'] < c['STOCH_D']) if s == 'long' else (c['STOCH_K'] > c['STOCH_D'])
def s2_43_long(c, p, pp=None): return c['close'] > c['VWAP'] and p['close'] < p['VWAP'] and c['STOCH_K'] > c['STOCH_D']
def s2_43_short(c, p, pp=None): return c['close'] < c['VWAP'] and p['close'] > p['VWAP'] and c['STOCH_K'] < c['STOCH_D']
def s2_43_exit(c, p, s, pp=None): return (c['close'] < c['VWAP']) if s == 'long' else (c['close'] > c['VWAP'])
def s2_44_long(c, p, pp=None): return c['STOCH_K'] < 15 and c['STOCH_K'] > p['STOCH_K']
def s2_44_short(c, p, pp=None): return c['STOCH_K'] > 85 and c['STOCH_K'] < p['STOCH_K']
def s2_44_exit(c, p, s, pp=None): return (c['STOCH_K'] > 50) if s == 'long' else (c['STOCH_K'] < 50)
# =======================
# 3. BREAKOUT (3.40–3.42)
# =======================
def s3_40_long(c, p, pp=None): return c['close'] > c['DC_high'] and c['close'] > c['VWAP'] and c['STOCH_K'] > c[
    'STOCH_D']
def s3_40_short(c, p, pp=None): return c['close'] < c['DC_low'] and c['close'] < c['VWAP'] and c['STOCH_K'] < c[
    'STOCH_D']
def s3_40_exit(c, p, s, pp=None): return (c['close'] < c['VWAP']) if s == 'long' else (c['close'] > c['VWAP'])
def s3_41_long(c, p, pp=None): return c['BBW'] < 0.07 and c['close'] > c['VWAP'] and c['STOCH_K'] > 50
def s3_41_short(c, p, pp=None): return c['BBW'] < 0.07 and c['close'] < c['VWAP'] and c['STOCH_K'] < 50
def s3_41_exit(c, p, s, pp=None): return (c['close'] < c['BB_middle']) if s == 'long' else (c['close'] > c['BB_middle'])
def s3_42_long(c, p, pp=None): return p['close'] < p['VWAP'] and c['close'] > c['VWAP'] and c['STOCH_K'] > c['STOCH_D']
def s3_42_short(c, p, pp=None): return p['close'] > p['VWAP'] and c['close'] < c['VWAP'] and c['STOCH_K'] < c['STOCH_D']
def s3_42_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])


# ============================================================
# 1. BTC — Estratégias Otimizadas
# ============================================================

def s1_44_long(c,p,pp=None): return trend_bull(c) and c['close'] >= c['VWAP'] and c['STOCH_K'] < 35 and c['STOCH_K'] > p['STOCH_K']
def s1_44_short(c,p,pp=None): return trend_bear(c) and c['close'] <= c['VWAP'] and c['STOCH_K'] > 65 and c['STOCH_K'] < p['STOCH_K']
def s1_44_exit(c,p,s,pp=None): return (c['close'] < c['EMA_9']) if s=='long' else (c['close'] > c['EMA_9'])

def s1_45_long(c,p,pp=None): return p['close'] < p['VWAP'] and c['close'] > c['VWAP'] and c['STOCH_K'] > c['STOCH_D']
def s1_45_short(c,p,pp=None): return p['close'] > p['VWAP'] and c['close'] < c['VWAP'] and c['STOCH_K'] < c['STOCH_D']
def s1_45_exit(c,p,s,pp=None): return (c['close'] < c['VWAP']) if s=='long' else (c['close'] > c['VWAP'])

def s2_45_long(c,p,pp=None): return c['STOCH_K'] < 12 and c['STOCH_K'] > p['STOCH_K'] and c['close'] > c['VWAP']*0.995
def s2_45_short(c,p,pp=None): return c['STOCH_K'] > 88 and c['STOCH_K'] < p['STOCH_K'] and c['close'] < c['VWAP']*1.005
def s2_45_exit(c,p,s,pp=None): return (c['STOCH_K'] > 45) if s=='long' else (c['STOCH_K'] < 55)


# ============================================================
# ETH — Estratégias (renumeradas 1.46, 2.46 e 3.43)
# ============================================================

def s1_46_long(c,p,pp=None): return c['close'] > c['VWAP'] and p['close'] < p['VWAP'] and c['STOCH_K'] > 30 and c['STOCH_D'] > 25
def s1_46_short(c,p,pp=None): return c['close'] < c['VWAP'] and p['close'] > p['VWAP'] and c['STOCH_K'] < 70 and c['STOCH_D'] < 75
def s1_46_exit(c,p,s,pp=None): return (c['close'] < c['VWAP']) if s=='long' else (c['close'] > c['VWAP'])

def s2_46_long(c,p,pp=None): return c['STOCH_K'] < 20 and c['STOCH_D'] < 20 and c['close'] > c['VWAP']*0.99
def s2_46_short(c,p,pp=None): return c['STOCH_K'] > 80 and c['STOCH_D'] > 80 and c['close'] < c['VWAP']*1.01
def s2_46_exit(c,p,s,pp=None): return (c['STOCH_K'] > 50) if s=='long' else (c['STOCH_K'] < 50)

def s3_43_long(c,p,pp=None): return c['BBW'] < 0.065 and c['close'] > c['VWAP'] and c['STOCH_K'] > c['STOCH_D']
def s3_43_short(c,p,pp=None): return c['BBW'] < 0.065 and c['close'] < c['VWAP'] and c['STOCH_K'] < c['STOCH_D']
def s3_43_exit(c,p,s,pp=None): return (c['close'] < c['BB_middle']) if s=='long' else (c['close'] > c['BB_middle'])

def s1_50_long(c, p, pp=None): return trend_bull(c) and c['MACD_hist'] > 0 and c['MACD_hist'] > p['MACD_hist'] and c['close'] > c['VWAP']
def s1_50_short(c, p, pp=None): return trend_bear(c) and c['MACD_hist'] < 0 and c['MACD_hist'] < p['MACD_hist'] and c['close'] < c['VWAP']
def s1_50_exit(c, p, s, pp=None): return (c['close'] < c['EMA_9']) if s == 'long' else (c['close'] > c['EMA_9'])

# Chikou Free: Chikou Span acima do preço de 26 períodos atrás (momentum limpo).
def s1_50_01_long(data, *args): return s1_50_long(data, *args) and data['Chikou'] > data['close_prev']

# ============================================================
# ALT — Estratégias (renumeradas 2.47, 3.44, 3.45)
# ============================================================

def s2_47_long(c,p,pp=None): return c['STOCH_K'] < 10 and c['close'] > c['VWAP']*0.98
def s2_47_short(c,p,pp=None): return c['STOCH_K'] > 90 and c['close'] < c['VWAP']*1.02
def s2_47_exit(c,p,s,pp=None): return (c['STOCH_K'] > 40) if s=='long' else (c['STOCH_K'] < 60)

def s3_44_long(c,p,pp=None): return c['Vol_Ratio'] > 1.5 and c['close'] > c['VWAP'] and c['STOCH_K'] > 60
def s3_44_short(c,p,pp=None): return c['Vol_Ratio'] > 1.5 and c['close'] < c['VWAP'] and c['STOCH_K'] < 40
def s3_44_exit(c,p,s,pp=None): return (c['close'] < c['VWAP']) if s=='long' else (c['close'] > c['VWAP'])
# 15. Aroon Dominance: Aroon Up > 70.
def s3_44_01_long(data, *args): return s3_44_long(data, *args) and data['Aroon_Up'] > 70


def s3_45_long(c,p,pp=None): return c['close'] > c['DC_high'] and c['close'] > c['VWAP'] and c['STOCH_K'] > c['STOCH_D']
def s3_45_short(c,p,pp=None): return c['close'] < c['DC_low'] and c['close'] < c['VWAP'] and c['STOCH_K'] < c['STOCH_D']
def s3_45_exit(c,p,s,pp=None): return (c['close'] < c['VWAP']) if s=='long' else (c['close'] > c['VWAP'])


def s4_04_exit(c, p, s, pp=None): return s3_23_exit(c, p, s, pp) if s == 'long' else s3_41_exit(c, p, s, pp)
def s4_39_exit(c, p, s, pp=None): return s3_31_exit(c, p, s, pp) if s == 'long' else s1_40_exit(c, p, s, pp)
# ==============================================================================
# =================== [FIM] FUNÇÕES DE ESTRATÉGIA (def) ======================
# ==============================================================================

def s1_47_long(c, p, pp=None): return trend_bull(c) and c['close'] > c['EMA_9'] and c['P_DI'] > 25 and p['P_DI'] < c['P_DI']
def s1_47_short(c, p, pp=None): return trend_bear(c) and c['close'] < c['EMA_9'] and c['M_DI'] > 25 and p['M_DI'] < c['M_DI']
def s1_47_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])

def s1_48_long(c, p, pp=None): return trend_bull(c) and c['RSI_9'] > 50 and p['RSI_9'] <= 50 and c['close'] > c['BB_middle']
def s1_48_short(c, p, pp=None): return trend_bear(c) and c['RSI_9'] < 50 and p['RSI_9'] >= 50 and c['close'] < c['BB_middle']
def s1_48_exit(c, p, s, pp=None): return (c['close'] < c['EMA_34']) if s == 'long' else (c['close'] > c['EMA_34'])

def s1_49_long(c, p, pp=None): return c['close'] > c['EMA_50'] and p['low'] <= p['EMA_50'] and c['close'] > p['close'] and c['ADX'] > 20
def s1_49_short(c, p, pp=None): return c['close'] < c['EMA_50'] and p['high'] >= p['EMA_50'] and c['close'] < p['close'] and c['ADX'] > 20
def s1_49_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])

def s1_51_long(c, p, pp=None): return c['close'] > c['EMA_100'] and p['close'] < p['EMA_100'] and c['RSI_14'] > 60 and c['Vol_Ratio'] > 1.0
def s1_51_short(c, p, pp=None): return c['close'] < c['EMA_100'] and p['close'] > p['EMA_100'] and c['RSI_14'] < 40 and c['Vol_Ratio'] > 1.0
def s1_51_exit(c, p, s, pp=None): return (c['close'] < c['EMA_50']) if s == 'long' else (c['close'] > c['EMA_50'])

def s1_52_long(c, p, pp=None): return c['EMA_9'] > c['EMA_21'] and c['EMA_21'] > c['EMA_50'] and c['low'] <= c['EMA_9'] and c['close'] > c['EMA_9']
def s1_52_short(c, p, pp=None): return c['EMA_9'] < c['EMA_21'] and c['EMA_21'] < c['EMA_50'] and c['high'] >= c['EMA_9'] and c['close'] < c['EMA_9']
def s1_52_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])

def s1_53_long(c, p, pp=None): return trend_bull(c) and c['close'] > c['BB_upper'] and c['MACD_hist'] > 0 and p['MACD_hist'] <= 0
def s1_53_short(c, p, pp=None): return trend_bear(c) and c['close'] < c['BB_lower'] and c['MACD_hist'] < 0 and p['MACD_hist'] >= 0
def s1_53_exit(c, p, s, pp=None): return (c['close'] < c['BB_middle']) if s == 'long' else (c['close'] > c['BB_middle'])

def s1_54_long(c, p, pp=None): return c['close'] > c['EMA_21'] and c['close'] > c['VWAP'] and c['STOCH_K'] > c['STOCH_D'] and c['ADX'] > 20
def s1_54_short(c, p, pp=None): return c['close'] < c['EMA_21'] and c['close'] < c['VWAP'] and c['STOCH_K'] < c['STOCH_D'] and c['ADX'] > 20
def s1_54_exit(c, p, s, pp=None): return (c['close'] < c['VWAP']) if s == 'long' else (c['close'] > c['VWAP'])

# ============================================================
# 1. TENDÊNCIA (1.47 - 1.60) — BTC Focus (Corrigido s1_55)
# ============================================================

def s1_55_long(c, p, pp=None):
    divisor = max(1e-10, c['high'] - c['low'])
    return trend_bull(c) and (c['close'] - c['low']) / divisor > 0.7 and c['Vol_Ratio'] > 1.2

def s1_55_short(c, p, pp=None):
    divisor = max(1e-10, c['high'] - c['low'])
    return trend_bear(c) and (c['high'] - c['close']) / divisor > 0.7 and c['Vol_Ratio'] > 1.2
def s1_55_exit(c, p, s, pp=None): return (c['close'] < c['EMA_9']) if s == 'long' else (c['close'] > c['EMA_9'])

def s1_56_long(c, p, pp=None): return c['EMA_12'] > c['EMA_26'] and p['EMA_12'] < p['EMA_26'] and c['close'] > c['EMA_200'] and c['RSI_14'] > 50
def s1_56_short(c, p, pp=None): return c['EMA_12'] < c['EMA_26'] and p['EMA_12'] > p['EMA_26'] and c['close'] < c['EMA_200'] and c['RSI_14'] < 50
def s1_56_exit(c, p, s, pp=None): return (c['close'] < c['BB_middle']) if s == 'long' else (c['close'] > c['BB_middle'])

def s1_57_long(c, p, pp=None): return c['MACD'] > 0 and c['close'] > c['MACD'] and p['close'] <= p['MACD'] and c['ADX'] > 25
def s1_57_short(c, p, pp=None): return c['MACD'] < 0 and c['close'] < c['MACD'] and p['close'] >= p['MACD'] and c['ADX'] > 25
def s1_57_exit(c, p, s, pp=None): return (c['MACD'] < c['MACD_signal']) if s == 'long' else (c['MACD'] > c['MACD_signal'])

def s1_58_long(c, p, pp=None): return trend_bull(c) and c['close'] > c['EMA_34'] and p['close'] < p['EMA_34'] and c['low'] > c['low_prev']
def s1_58_short(c, p, pp=None): return trend_bear(c) and c['close'] < c['EMA_34'] and p['close'] > p['EMA_34'] and c['high'] < c['high_prev']
def s1_58_exit(c, p, s, pp=None): return (c['close'] < c['EMA_9']) if s == 'long' else (c['close'] > c['EMA_9'])

def s1_59_long(c, p, pp=None): return c['close'] > c['VWAP'] and p['low'] <= p['VWAP'] and c['close'] > c['high_prev'] and c['Vol_Ratio'] > 1.1
def s1_59_short(c, p, pp=None): return c['close'] < c['VWAP'] and p['high'] >= p['VWAP'] and c['close'] < c['low_prev'] and c['Vol_Ratio'] > 1.1
def s1_59_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])

def s1_60_long(c, p, pp=None): return c['P_DI'] > c['M_DI'] and c['ADX'] > 25 and c['close'] > c['BB_middle']
def s1_60_short(c, p, pp=None): return c['M_DI'] > c['P_DI'] and c['ADX'] > 25 and c['close'] < c['BB_middle']
def s1_60_exit(c, p, s, pp=None): return (c['P_DI'] < c['M_DI']) if s == 'long' else (c['M_DI'] < c['P_DI'])

# Antiga 1 (Tendencia Rapida) -> 1.61
def s1_61_long(c, p, pp=None): return c['close'] > c['EMA_200'] and p['EMA_9'] < p['EMA_21'] and c['EMA_9'] > c['EMA_21'] and c['RSI_9'] > 55 and c['ADX'] > 20
def s1_61_short(c, p, pp=None): return c['close'] < c['EMA_200'] and p['EMA_9'] > p['EMA_21'] and c['EMA_9'] < c['EMA_21'] and c['RSI_9'] < 45 and c['ADX'] > 20
def s1_61_exit(c, p, s, pp=None): return (c['EMA_9'] < c['EMA_21'] and c['RSI_9'] < 48) if s == 'long' else (c['EMA_9'] > c['EMA_21'] and c['RSI_9'] > 52)

# Antiga 5 (Swing Curto) -> 1.62
def s1_62_long(c, p, pp=None): return c['close'] > c['EMA_200'] and p['EMA_21'] < p['EMA_50'] and c['EMA_21'] > c['EMA_50'] and c['ADX'] > 25
def s1_62_short(c, p, pp=None): return c['close'] < c['EMA_200'] and p['EMA_21'] > p['EMA_50'] and c['EMA_21'] < c['EMA_50'] and c['ADX'] > 25
def s1_62_exit(c, p, s, pp=None): return (c['EMA_9'] < c['EMA_21']) if s == 'long' else (c['EMA_9'] > c['EMA_21'])

# Antiga 7 (Tendencia Confirmada) -> 1.63
def s1_63_long(c, p, pp=None): return c['close'] > c['EMA_200'] and c['EMA_9'] > c['EMA_21'] and c['EMA_21'] > c['EMA_50'] and p['low'] <= p['EMA_21'] and c['close'] > c['EMA_21'] and c['ADX'] > 25
def s1_63_short(c, p, pp=None): return c['close'] < c['EMA_200'] and c['EMA_9'] < c['EMA_21'] and c['EMA_21'] < c['EMA_50'] and p['high'] >= p['EMA_21'] and c['close'] < c['EMA_21'] and c['ADX'] > 25
def s1_63_exit(c, p, s, pp=None): return (c['EMA_21'] < c['EMA_50']) if s == 'long' else (c['EMA_21'] > c['EMA_50'])

# Antiga 9 (MACD/RSI Trend Follower) -> 1.64
def s1_64_long(c, p, pp=None): return c['close'] > c['EMA_200'] and (p['MACD'] < p['MACD_signal'] and c['MACD'] > c['MACD_signal'] and c['MACD'] > 0 and c['RSI_14'] > 52 and c['ADX'] > 20)
def s1_64_short(c, p, pp=None): return c['close'] < c['EMA_200'] and (p['MACD'] > p['MACD_signal'] and c['MACD'] < c['MACD_signal'] and c['MACD'] < 0 and c['RSI_14'] < 48 and c['ADX'] > 20)
def s1_64_exit(c, p, s, pp=None): return ((c['MACD'] < c['MACD_signal'] and c['RSI_14'] < 50) if s == 'long' else (c['MACD'] > c['MACD_signal'] and c['RSI_14'] > 50))

# ============================================================
# 2. MOMENTUM (2.48 - 2.58) — BTC Focus
# ============================================================

def s2_48_long(c, p, pp=None): return c['RSI_14'] < 30 and c['STOCH_K'] < 20 and c['close'] > c['EMA_21'] and c['Vol_Ratio'] > 1.0
def s2_48_short(c, p, pp=None): return c['RSI_14'] > 70 and c['STOCH_K'] > 80 and c['close'] < c['EMA_21'] and c['Vol_Ratio'] > 1.0
def s2_48_exit(c, p, s, pp=None): return (c['RSI_14'] > 50) if s == 'long' else (c['RSI_14'] < 50)

def s2_49_long(c, p, pp=None): return c['WILLR'] < -85 and c['WILLR'] > p['WILLR'] and c['MACD_hist'] > 0
def s2_49_short(c, p, pp=None): return c['WILLR'] > -15 and c['WILLR'] < p['WILLR'] and c['MACD_hist'] < 0
def s2_49_exit(c, p, s, pp=None): return (c['WILLR'] > -50) if s == 'long' else (c['WILLR'] < -50)

def s2_50_long(c, p, pp=None): return c['close'] < c['BB_lower'] and c['close'] > c['low_prev'] and c['MACD_hist'] > 0
def s2_50_short(c, p, pp=None): return c['close'] > c['BB_upper'] and c['close'] < c['high_prev'] and c['MACD_hist'] < 0
def s2_50_exit(c, p, s, pp=None): return (c['close'] > c['BB_middle']) if s == 'long' else (c['close'] < c['BB_middle'])

def s2_51_long(c, p, pp=None): return c['CCI'] < -150 and c['CCI'] > p['CCI'] and c['RSI_14'] < 40
def s2_51_short(c, p, pp=None): return c['CCI'] > 150 and c['CCI'] < p['CCI'] and c['RSI_14'] > 60
def s2_51_exit(c, p, s, pp=None): return (c['CCI'] > 0) if s == 'long' else (c['CCI'] < 0)

def s2_52_long(c, p, pp=None): return c['STOCH_K'] < 25 and p['STOCH_K'] <= 25 and c['STOCH_K'] > c['STOCH_D'] and trend_bull(c)
def s2_52_short(c, p, pp=None): return c['STOCH_K'] > 75 and p['STOCH_K'] >= 75 and c['STOCH_K'] < c['STOCH_D'] and trend_bear(c)
def s2_52_exit(c, p, s, pp=None): return (c['STOCH_K'] > 50) if s == 'long' else (c['STOCH_K'] < 50)

def s2_53_long(c, p, pp=None): return c['ROC'] > 0 and c['ROC'] < 0.5 and p['ROC'] < c['ROC_prev'] and c['Vol_Ratio'] > 1.0
def s2_53_short(c, p, pp=None): return c['ROC'] < 0 and c['ROC'] > -0.5 and p['ROC'] > c['ROC_prev'] and c['Vol_Ratio'] > 1.0
def s2_53_exit(c, p, s, pp=None): return (c['ROC'] < 0.1) if s == 'long' else (c['ROC'] > -0.1)

def s2_54_long(c, p, pp=None): return c['RSI_14'] < 35 and c['close'] > c['VWAP'] and c['MACD_hist'] > 0
def s2_54_short(c, p, pp=None): return c['RSI_14'] > 65 and c['close'] < c['VWAP'] and c['MACD_hist'] < 0
def s2_54_exit(c, p, s, pp=None): return (c['RSI_14'] > 50) if s == 'long' else (c['RSI_14'] < 50)

def s2_55_long(c, p, pp=None): return c['close'] < c['KC_lower'] and c['close'] > c['open'] and c['RSI_14'] < 40
def s2_55_short(c, p, pp=None): return c['close'] > c['KC_upper'] and c['close'] < c['open'] and c['RSI_14'] > 60
def s2_55_exit(c, p, s, pp=None): return (c['close'] > c['EMA_21']) if s == 'long' else (c['close'] < c['EMA_21'])

def s2_56_long(c, p, pp=None): return c['STOCH_K'] < 20 and c['MACD'] > 0 and p['STOCH_K'] < c['STOCH_K']
def s2_56_short(c, p, pp=None): return c['STOCH_K'] > 80 and c['MACD'] < 0 and p['STOCH_K'] > c['STOCH_K']
def s2_56_exit(c, p, s, pp=None): return (c['STOCH_K'] > 50) if s == 'long' else (c['STOCH_K'] < 50)

def s2_57_long(c, p, pp=None): return c['MACD_hist'] > 0 and p['MACD_hist'] < 0 and c['RSI_14'] > 50 and c['Vol_Ratio'] > 1.1
def s2_57_short(c, p, pp=None): return c['MACD_hist'] < 0 and p['MACD_hist'] > 0 and c['RSI_14'] < 50 and c['Vol_Ratio'] > 1.1
def s2_57_exit(c, p, s, pp=None): return (c['MACD_hist'] < 0) if s == 'long' else (c['MACD_hist'] > 0)

def s2_58_long(c, p, pp=None): return c['close'] < c['BB_lower'] and c['low'] < c['low_prev'] and c['RSI_14'] > p['RSI_14']
def s2_58_short(c, p, pp=None): return c['close'] > c['BB_upper'] and c['high'] > c['high_prev'] and c['RSI_14'] < p['RSI_14']
def s2_58_exit(c, p, s, pp=None): return (c['RSI_14'] > 50) if s == 'long' else (c['RSI_14'] < 50)

# Antiga 2 (Momentum BB-Middle) -> 2.59
def s2_59_long(c, p, pp=None): return c['close'] > c['EMA_200'] and c['close'] > c['BB_middle'] and c['MACD'] > c['MACD_signal']
def s2_59_short(c, p, pp=None): return c['close'] < c['EMA_200'] and c['close'] < c['BB_middle'] and c['MACD'] < c['MACD_signal']
def s2_59_exit(c, p, s, pp=None): return (c['MACD'] < 0) if s == 'long' else (c['MACD'] > 0)

# Antiga 3 (Scalping Intraday) -> 2.60
def s2_60_long(c, p, pp=None): return p['close'] < p['VWAP'] and c['close'] > c['VWAP'] and c['STOCH_K'] > p['STOCH_D']
def s2_60_short(c, p, pp=None): return p['close'] > p['VWAP'] and c['close'] < c['VWAP'] and c['STOCH_K'] < p['STOCH_D']
def s2_60_exit(c, p, s, pp=None): return (c['STOCH_K'] > 75 and c['STOCH_K'] < c['STOCH_D']) if s == 'long' else (c['STOCH_K'] < 25 and c['STOCH_K'] > c['STOCH_D'])

# Antiga 6 (Reversao Forte) -> 2.61
def s2_61_long(c, p, pp=None): return p['close'] < p['BB_lower'] and c['close'] > p['BB_lower'] and c['RSI_9'] < 30
def s2_61_short(c, p, pp=None): return p['close'] > p['BB_upper'] and c['close'] < c['BB_upper'] and c['RSI_9'] > 70
def s2_61_exit(c, p, s, pp=None): return (c['close'] > c['BB_middle']) if s == 'long' else (c['close'] < c['BB_middle'])

# Antiga 8 (Scalper Volume) -> 2.62
def s2_62_long(c, p, pp=None): return (c['close'] > c['EMA_9'] and c['volume'] > c['Volume_MA'] * 1.8 and c['RSI_9'] < 58 and c['STOCH_K'] < 80)
def s2_62_short(c, p, pp=None): return (c['close'] < c['EMA_9'] and c['volume'] > c['Volume_MA'] * 1.8 and c['RSI_9'] > 42 and c['STOCH_K'] > 20)
def s2_62_exit(c, p, s, pp=None): return ((c['close'] < c['EMA_9'] or c['STOCH_K'] > 80) if s == 'long' else (c['close'] > c['EMA_9'] or c['STOCH_K'] < 20))

# Antiga 10 (Momentum Explosivo) -> 2.63
def s2_63_long(c, p, pp=None): return c['close'] > c['EMA_200'] and p['MACD'] < p['MACD_signal'] and c['MACD'] > c['MACD_signal'] and c['MACD'] > 0 and c['ADX'] > 20 and c['close'] > c['VWAP']
def s2_63_short(c, p, pp=None): return c['close'] < c['EMA_200'] and p['MACD'] > p['MACD_signal'] and c['MACD'] < c['MACD_signal'] and c['MACD'] < 0 and c['ADX'] > 20 and c['close'] < c['VWAP']
def s2_63_exit(c, p, s, pp=None): return (c['MACD'] < 0) if s == 'long' else (c['MACD'] > 0)

# Antiga 11 (Scalping Reversal VWAP-Stoch) -> 2.64
def s2_64_long(c, p, pp=None): return c['EMA_9'] > c['EMA_21'] and p['STOCH_K'] < 20 and c['STOCH_K'] > p['STOCH_D'] and c['close'] > c['VWAP']
def s2_64_short(c, p, pp=None): return c['EMA_9'] < c['EMA_21'] and p['STOCH_K'] > 80 and c['STOCH_K'] < p['STOCH_D'] and c['close'] < c['VWAP']
def s2_64_exit(c, p, s, pp=None): return (c['STOCH_K'] > 75 and c['STOCH_K'] < c['STOCH_D']) if s == 'long' else (c['STOCH_K'] < 25 and c['STOCH_K'] > c['STOCH_D'])

# ============================================================
# 3. BREAKOUT (3.46 - 3.55) — BTC Focus
# ============================================================

def s3_46_long(c, p, pp=None): return c['BBW'] < 0.05 and c['close'] > c['BB_upper'] and c['close'] > c['VWAP']
def s3_46_short(c, p, pp=None): return c['BBW'] < 0.05 and c['close'] < c['BB_lower'] and c['close'] < c['VWAP']
def s3_46_exit(c, p, s, pp=None): return (c['close'] < c['BB_middle']) if s == 'long' else (c['close'] > c['BB_middle'])

def s3_47_long(c, p, pp=None): return c['close'] > c['DC_high'] and c['P_DI'] > c['M_DI'] and c['Vol_Ratio'] > 1.5
def s3_47_short(c, p, pp=None): return c['close'] < c['DC_low'] and c['M_DI'] > c['P_DI'] and c['Vol_Ratio'] > 1.5
def s3_47_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])

def s3_48_long(c, p, pp=None): return c['close'] > c['KC_upper'] and c['RSI_14'] > 65 and c['ADX'] > 20
def s3_48_short(c, p, pp=None): return c['close'] < c['KC_lower'] and c['RSI_14'] < 35 and c['ADX'] > 20
def s3_48_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])

def s3_49_long(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 2.5) and c['close'] > c['open'] and c['MACD'] > 0
def s3_49_short(c, p, pp=None): return (c['high'] - c['low']) > (c['ATR'] * 2.5) and c['close'] < c['open'] and c['MACD'] < 0
def s3_49_exit(c, p, s, pp=None): return (c['ATR'] < c['ATR_prev'])

def s3_50_long(c, p, pp=None): return c['close'] > c['BB_upper'] and p['close'] < p['BB_upper'] and c['Vol_Ratio'] > 1.2 and c['STOCH_K'] > 70
def s3_50_short(c, p, pp=None): return c['close'] < c['BB_lower'] and p['close'] > p['BB_lower'] and c['Vol_Ratio'] > 1.2 and c['STOCH_K'] < 30
def s3_50_exit(c, p, s, pp=None): return (c['close'] < c['BB_middle']) if s == 'long' else (c['close'] > c['BB_middle'])

def s3_51_long(c, p, pp=None): return c['close'] > c['DC_high'] and c['close'] > c['EMA_9'] and c['ADX'] > 25
def s3_51_short(c, p, pp=None): return c['close'] < c['DC_low'] and c['close'] < c['EMA_9'] and c['ADX'] > 25
def s3_51_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])

def s3_52_long(c, p, pp=None): return c['BBW'] < 0.06 and c['close'] > c['VWAP'] and c['Vol_Ratio'] > 1.1
def s3_52_short(c, p, pp=None): return c['BBW'] < 0.06 and c['close'] < c['VWAP'] and c['Vol_Ratio'] > 1.1
def s3_52_exit(c, p, s, pp=None): return (c['close'] < c['BB_middle']) if s == 'long' else (c['close'] > c['BB_middle'])

def s3_53_long(c, p, pp=None): return c['close'] > c['KC_upper'] and c['close'] > c['high_prev'] and c['RSI_14'] > 55
def s3_53_short(c, p, pp=None): return c['close'] < c['KC_lower'] and c['close'] < c['low_prev'] and c['RSI_14'] < 45
def s3_53_exit(c, p, s, pp=None): return (c['close'] < c['EMA_21']) if s == 'long' else (c['close'] > c['EMA_21'])

def s3_54_long(c, p, pp=None): return c['close'] > c['DC_high'] and c['close'] > c['BB_upper'] and c['ADX'] > 20
def s3_54_short(c, p, pp=None): return c['close'] < c['DC_low'] and c['close'] < c['BB_lower'] and c['ADX'] > 20
def s3_54_exit(c, p, s, pp=None): return (c['close'] < c['EMA_50']) if s == 'long' else (c['close'] > c['EMA_50'])

def s3_55_long(c, p, pp=None): return c['close'] > c['BB_upper'] and c['MACD_hist'] > 0 and c['close'] > c['VWAP']
def s3_55_short(c, p, pp=None): return c['close'] < c['BB_lower'] and c['MACD_hist'] < 0 and c['close'] < c['VWAP']
def s3_55_exit(c, p, s, pp=None): return (c['close'] < c['BB_middle']) if s == 'long' else (c['close'] > c['BB_middle'])

def s3_56_long(c, p, pp=None): return p['close'] > p['BB_upper'] and c['volume'] > p['volume']
def s3_56_short(c, p, pp=None): return p['close'] < p['BB_lower'] and c['volume'] > p['volume']
def s3_56_exit(c, p, s, pp=None): return (c['BBW'] < c['BBW_min_120'] * 1.2) or (c['ATR'] < c['ATR_MA'] * 0.9)

def s2_65_long(c, p, pp=None):
    # Tendência: Preço acima da TEMA 9
    # Força: Corpo do candle maior que o pavio superior (mostra convicção compradora)
    return c['close'] > c['TEMA_9'] and c['body_size'] > c['upper_wick']

def s2_65_short(c, p, pp=None):
    # Tendência: Preço abaixo da TEMA 9
    # Força: Corpo do candle maior que o pavio inferior (mostra convicção vendedora)
    return c['close'] < c['TEMA_9'] and c['body_size'] > c['lower_wick']

def s2_65_exit(c, p, s, pp=None):
    # Saída padrão rápida (Cruzamento da EMA 9 simples)
    return (c['close'] < c['EMA_9']) if s == 'long' else (c['close'] > c['EMA_9'])

###
def s5_01_long(c, p):
    return (c['RSI_14'] < p['RSI_14']) and (c['WILLR'] < -20) and (c['Chikou'] > p['close'])

def s5_01_short(c, p):
    return (c['Vol_Ratio'] > 1.0) and (p['high'] >= p['EMA_200']) and (c['M_DI'] > c['P_DI'])

def s_pnl_long_29(c, p, pp=None):
    return (c['RSI_14'] > p['RSI_14']) and (c['WILLR'] < -60) and ((c['close'] > c['Senkou_Span_A']) & (c['close'] < c['Senkou_Span_B']) | (c['close'] < c['Senkou_Span_A']) & (c['close'] > c['Senkou_Span_B'])) and (c['ADX'] > 20)
def s_pnl_short_203(c, p, pp=None):
    return (c['close'] < c['EMA_50']) and (c['ROC'] < -0.5) and (c['STOCH_K'] < p['STOCH_K']) and ((c['high'] < p['high']) & (c['low'] > p['low']))

###

#
# NOVO DICIONÁRIO STRATEGIES
#
STRATEGIES = {
    # --- TENDÊNCIA (30) ---
    "1.01 Trend Cross 9/21": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_01_long, "short_entry": s1_01_short, "exit": s1_01_exit},
    "1.02 Trend Cross 12/34": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_02_long, "short_entry": s1_02_short, "exit": s1_02_exit},
    "1.03 Trend Cross 21/50": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_03_long, "short_entry": s1_03_short, "exit": s1_03_exit},
    "1.04 Trend Cross 50/100": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_04_long, "short_entry": s1_04_short, "exit": s1_04_exit},
    "1.05 Trend Cross 50/200": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_05_long, "short_entry": s1_05_short, "exit": s1_05_exit},
    "1.06 Trend Cross 9/21 + ADX": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_06_long, "short_entry": s1_06_short, "exit": s1_06_exit},
    "1.07 Trend Cross 12/34 + ADX": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_07_long, "short_entry": s1_07_short, "exit": s1_07_exit},
    "1.08 Trend Cross 21/50 + ADX": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_08_long, "short_entry": s1_08_short, "exit": s1_08_exit},
    "1.09 Trend Cross 50/100 + ADX": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_09_long, "short_entry": s1_09_short, "exit": s1_09_exit},
    "1.10 Trend Cross 50/200 + ADX": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_10_long, "short_entry": s1_10_short, "exit": s1_10_exit},
    "1.11 Pullback EMA 21": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_11_long, "short_entry": s1_11_short, "exit": s1_11_exit},
    "1.12 Pullback EMA 34": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_12_long, "short_entry": s1_12_short, "exit": s1_12_exit},
    "1.13 Pullback EMA 50": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_13_long, "short_entry": s1_13_short, "exit": s1_13_exit},
    "1.14 Pullback EMA 100": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_14_long, "short_entry": s1_14_short, "exit": s1_14_exit},
    "1.15 Pullback EMA 200": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_15_long, "short_entry": s1_15_short, "exit": s1_15_exit},
    "1.16 Pullback EMA 21 + RSI": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_16_long, "short_entry": s1_16_short, "exit": s1_16_exit},
    "1.17 Pullback EMA 34 + RSI": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_17_long, "short_entry": s1_17_short, "exit": s1_17_exit},
    "1.18 Pullback EMA 50 + RSI": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_18_long, "short_entry": s1_18_short, "exit": s1_18_exit},
    "1.19 Pullback EMA 100 + RSI": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_19_long, "short_entry": s1_19_short, "exit": s1_19_exit},
    "1.20 Pullback EMA 200 + RSI": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_20_long, "short_entry": s1_20_short, "exit": s1_20_exit},
    "1.21 Triple EMA": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_21_long, "short_entry": s1_21_short, "exit": s1_21_exit},
    "1.22 MACD Zero Cross": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_22_long, "short_entry": s1_22_short, "exit": s1_22_exit},
    "1.23 DI Cross": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_23_long, "short_entry": s1_23_short, "exit": s1_23_exit},
    "1.24 ADX Strong Trend": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_24_long, "short_entry": s1_24_short, "exit": s1_24_exit},
    "1.25 MACD Hist Re-Entry": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_25_long, "short_entry": s1_25_short, "exit": s1_25_exit},
    "1.26 ADX + RSI Confluence": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_26_long, "short_entry": s1_26_short, "exit": s1_26_exit},
    "1.27 MACD Signal Cross": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_27_long, "short_entry": s1_27_short, "exit": s1_27_exit},
    "1.28 Trend Exhaustion": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_28_long, "short_entry": s1_28_short, "exit": s1_28_exit},
    "1.29 P_DI Breakout": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_29_long, "short_entry": s1_29_short, "exit": s1_29_exit},
    "1.30 The Holy Grail": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_30_long, "short_entry": s1_30_short, "exit": s1_30_exit},
    "1.31 MR EMA9 Bounce": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_31_long, "short_entry": s1_31_short, "exit": s1_31_exit},
    "1.32 DC Breakout + Vol": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_32_long, "short_entry": s1_32_short, "exit": s1_32_exit},
    "1.33 RSI Divergence Trend": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_33_long, "short_entry": s1_33_short, "exit": s1_33_exit},
    "1.34 Breakout + Vol Filter": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_34_long, "short_entry": s1_34_short, "exit": s1_34_exit},
    "1.35 Engulfing EMA21": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_35_long, "short_entry": s1_35_short, "exit": s1_35_exit},
    "1.36 Inside Bar Trend": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_36_long, "short_entry": s1_36_short, "exit": s1_36_exit},
    "1.37 Pinbar EMA21": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_37_long, "short_entry": s1_37_short, "exit": s1_37_exit},
    "1.38 EMA21 MACD Accel": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_38_long, "short_entry": s1_38_short, "exit": s1_38_exit},
    "1.39 EMA34 Compression": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_39_long, "short_entry": s1_39_short, "exit": s1_39_exit},
    "1.40 BB-Middle Reclaim": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_40_long, "short_entry": s1_40_short, "exit": s1_40_exit},
    "1.41 VWAP Bounce + Stoch": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_41_long, "short_entry": s1_41_short, "exit": s1_41_exit},
    "1.42 VWAP Pullback Stoch OS/OB": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_42_long, "short_entry": s1_42_short, "exit": s1_42_exit},
    "1.43 VWAP + EMA21 Trend Filter": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_43_long, "short_entry": s1_43_short, "exit": s1_43_exit},
    "1.44 VWAP Shallow Pullback": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_44_long, "short_entry": s1_44_short, "exit": s1_44_exit},
    "1.45 VWAP Reclaim Trend": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_45_long, "short_entry": s1_45_short, "exit": s1_45_exit},
    "1.46 Anti-Fakeout VWAP": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_46_long, "short_entry": s1_46_short, "exit": s1_46_exit},
    "1.47 DI-Accel Trend": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_47_long, "short_entry": s1_47_short, "exit": s1_47_exit},
    "1.48 RSI Flip BB": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_48_long, "short_entry": s1_48_short, "exit": s1_48_exit},
    "1.49 EMA50 Pullback ADX": {"timeframe": "1h", "min_candles": 200, "long_entry": s1_49_long, "short_entry": s1_49_short, "exit": s1_49_exit},
    "1.50 MACD-Accel VWAP": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_50_long, "short_entry": s1_50_short, "exit": s1_50_exit},
    "1.50.01 MACD-Accel + Chikou": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_50_01_long, "short_entry": s1_50_short, "exit": s1_50_exit},
    "1.51 EMA100 Vol Break": {"timeframe": "1h", "min_candles": 200, "long_entry": s1_51_long, "short_entry": s1_51_short, "exit": s1_51_exit},
    "1.52 Triple EMA Pullback": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_52_long, "short_entry": s1_52_short, "exit": s1_52_exit},
    "1.53 BB Break MACD Flip": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_53_long, "short_entry": s1_53_short, "exit": s1_53_exit},
    "1.54 VWAP Stoch Trend": {"timeframe": "1h", "min_candles": 200, "long_entry": s1_54_long, "short_entry": s1_54_short, "exit": s1_54_exit},
    "1.55 Pinbar Vol Trend": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_55_long, "short_entry": s1_55_short, "exit": s1_55_exit},
    "1.56 EMA12/26 Trend Follow": {"timeframe": "1h", "min_candles": 200, "long_entry": s1_56_long, "short_entry": s1_56_short, "exit": s1_56_exit},
    "1.57 MACD Cross Trend": {"timeframe": "1h", "min_candles": 200, "long_entry": s1_57_long, "short_entry": s1_57_short, "exit": s1_57_exit},
    "1.58 EMA34 Reclaim": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_58_long, "short_entry": s1_58_short, "exit": s1_58_exit},
    "1.59 VWAP Reclaim Vol": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_59_long, "short_entry": s1_59_short, "exit": s1_59_exit},
    "1.60 DI/ADX Conf Trend": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_60_long, "short_entry": s1_60_short, "exit": s1_60_exit},
    "1.61 Tendencia Rapida (Old 1)": {"timeframe": "15m", "min_candles": 200, "long_entry": s1_61_long, "short_entry": s1_61_short, "exit": s1_61_exit},
    "1.62 Swing Curto (Old 5)": {"timeframe": "1h", "min_candles": 200, "long_entry": s1_62_long, "short_entry": s1_62_short, "exit": s1_62_exit},
    "1.63 Tendencia Confirmada (Old 7)": {"timeframe": "1h", "min_candles": 200, "long_entry": s1_63_long, "short_entry": s1_63_short, "exit": s1_63_exit},
    "1.64 MACD/RSI Trend Follower (Old 9)": {"timeframe": "1h", "min_candles": 200, "long_entry": s1_64_long, "short_entry": s1_64_short, "exit": s1_64_exit},
    # --- MOMENTUM (30) ---
    "2.01 RSI Reversal 30": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_01_long, "short_entry": s2_01_short, "exit": s2_01_exit},
    "2.02 RSI Reversal 25": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_02_long, "short_entry": s2_02_short, "exit": s2_02_exit},
    "2.03 RSI Reversal 20": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_03_long, "short_entry": s2_03_short, "exit": s2_03_exit},
    "2.04 RSI Reversal 35": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_04_long, "short_entry": s2_04_short, "exit": s2_04_exit},
    "2.05 RSI Reversal 40": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_05_long, "short_entry": s2_05_short, "exit": s2_05_exit},
    "2.06 RSI Dip 30": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_06_long, "short_entry": s2_06_short, "exit": s2_06_exit},
    "2.07 RSI Dip 25": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_07_long, "short_entry": s2_07_short, "exit": s2_07_exit},
    "2.08 RSI Dip 20": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_08_long, "short_entry": s2_08_short, "exit": s2_08_exit},
    "2.09 RSI Dip 35": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_09_long, "short_entry": s2_09_short, "exit": s2_09_exit},
    "2.10 RSI Dip 40": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_10_long, "short_entry": s2_10_short, "exit": s2_10_exit},
    "2.11 CCI Break 100": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_11_long, "short_entry": s2_11_short, "exit": s2_11_exit},
    "2.12 CCI Break 150": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_12_long, "short_entry": s2_12_short, "exit": s2_12_exit},
    "2.13 CCI Break 200": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_13_long, "short_entry": s2_13_short, "exit": s2_13_exit},
    "2.14 CCI Break 50": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_14_long, "short_entry": s2_14_short, "exit": s2_14_exit},
    "2.15 CCI Break 0": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_15_long, "short_entry": s2_15_short, "exit": s2_15_exit},
    "2.15.01 CCI Break 0 + Stoch Room": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_15_long, "short_entry": s2_15_01_short, "exit": s2_15_exit},
    "2.16 WillR 80": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_16_long, "short_entry": s2_16_short, "exit": s2_16_exit},
    "2.17 WillR 90": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_17_long, "short_entry": s2_17_short, "exit": s2_17_exit},
    "2.18 WillR 75": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_18_long, "short_entry": s2_18_short, "exit": s2_18_exit},
    "2.19 WillR 70": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_19_long, "short_entry": s2_19_short, "exit": s2_19_exit},
    "2.20 WillR 50": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_20_long, "short_entry": s2_20_short, "exit": s2_20_exit},
    "2.21 ROC 0.2": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_21_long, "short_entry": s2_21_short, "exit": s2_21_exit},
    "2.22 ROC 0.4": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_22_long, "short_entry": s2_22_short, "exit": s2_22_exit},
    "2.23 ROC 0.6": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_23_long, "short_entry": s2_23_short, "exit": s2_23_exit},
    "2.24 ROC 0.8": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_24_long, "short_entry": s2_24_short, "exit": s2_24_exit},
    "2.25 ROC 1.0": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_25_long, "short_entry": s2_25_short, "exit": s2_25_exit},
    "2.26 ROC 1.2": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_26_long, "short_entry": s2_26_short, "exit": s2_26_exit},
    "2.27 ROC 1.4": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_27_long, "short_entry": s2_27_short, "exit": s2_27_exit},
    "2.28 ROC 1.6": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_28_long, "short_entry": s2_28_short, "exit": s2_28_exit},
    "2.29 ROC 1.8": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_29_long, "short_entry": s2_29_short, "exit": s2_29_exit},
    "2.30 ROC 2.0": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_30_long, "short_entry": s2_30_short, "exit": s2_30_exit},
    "2.31 BB + MACD Divergence": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_31_long, "short_entry": s2_31_short, "exit": s2_31_exit},
    "2.32 Donchian Overshoot": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_32_long, "short_entry": s2_32_short, "exit": s2_32_exit},
    "2.33 BB + Vol Confirmation": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_33_long, "short_entry": s2_33_short, "exit": s2_33_exit},
    "2.34 Double BB Touch": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_34_long, "short_entry": s2_34_short, "exit": s2_34_exit},
    "2.35 MACD Divergence": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_35_long, "short_entry": s2_35_short, "exit": s2_35_exit},
    "2.36 CCI Divergence + Vol": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_36_long, "short_entry": s2_36_short, "exit": s2_36_exit},
    "2.37 ROC Divergence": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_37_long, "short_entry": s2_37_short, "exit": s2_37_exit},
    "2.38 Panic Reversal": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_38_long, "short_entry": s2_38_short, "exit": s2_38_exit},
    "2.39 ROC Spike": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_39_long, "short_entry": s2_39_short, "exit": s2_39_exit},
    "2.40 Squeeze Momentum": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_40_long, "short_entry": s2_40_short, "exit": s2_40_exit},
    "2.41 CCI Recovery": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_41_long, "short_entry": s2_41_short, "exit": s2_41_exit},
    "2.42 Stoch Flip + VWAP": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_42_long, "short_entry": s2_42_short, "exit": s2_42_exit},
    "2.43 VWAP Reclaim Momentum": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_43_long, "short_entry": s2_43_short, "exit": s2_43_exit},
    "2.44 Stoch Extreme Reversal": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_44_long, "short_entry": s2_44_short, "exit": s2_44_exit},
    "2.45 Stoch Panic Reversal": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_45_long, "short_entry": s2_45_short, "exit": s2_45_exit},
    "2.46 Stoch Range Reversal": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_46_long, "short_entry": s2_46_short, "exit": s2_46_exit},
    "2.47 Stoch Extreme Snapback": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_47_long, "short_entry": s2_47_short, "exit": s2_47_exit},
    "2.48 RSI/Stoch Vol Reversal": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_48_long, "short_entry": s2_48_short, "exit": s2_48_exit},
    "2.49 WillR/MACD Reversal": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_49_long, "short_entry": s2_49_short, "exit": s2_49_exit},
    "2.50 BB Reversal LH/HL": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_50_long, "short_entry": s2_50_short, "exit": s2_50_exit},
    "2.51 CCI/RSI Extreme": {"timeframe": "1h", "min_candles": 200, "long_entry": s2_51_long, "short_entry": s2_51_short, "exit": s2_51_exit},
    "2.52 Stoch Trend Reversal": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_52_long, "short_entry": s2_52_short, "exit": s2_52_exit},
    "2.53 ROC Retracement": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_53_long, "short_entry": s2_53_short, "exit": s2_53_exit},
    "2.54 RSI/VWAP Momentum": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_54_long, "short_entry": s2_54_short, "exit": s2_54_exit},
    "2.55 KC Reversal Candle": {"timeframe": "1h", "min_candles": 200, "long_entry": s2_55_long, "short_entry": s2_55_short, "exit": s2_55_exit},
    "2.56 Stoch/MACD Reversal": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_56_long, "short_entry": s2_56_short, "exit": s2_56_exit},
    "2.57 MACD-RSI Vol Burst": {"timeframe": "1h", "min_candles": 200, "long_entry": s2_57_long, "short_entry": s2_57_short, "exit": s2_57_exit},
    "2.58 BB RSI Divergence": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_58_long, "short_entry": s2_58_short, "exit": s2_58_exit},
    "2.59 Momentum BB-Middle (Old 2)": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_59_long, "short_entry": s2_59_short, "exit": s2_59_exit},
    "2.60 Scalping Intraday (Old 3)": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_60_long, "short_entry": s2_60_short, "exit": s2_60_exit},
    "2.61 Reversao Forte (Old 6)": {"timeframe": "1h", "min_candles": 200, "long_entry": s2_61_long, "short_entry": s2_61_short, "exit": s2_61_exit},
    "2.62 Scalper Volume (Old 8)": {"timeframe": "1m", "min_candles": 200, "long_entry": s2_62_long, "short_entry": s2_62_short, "exit": s2_62_exit},
    "2.63 Momentum Explosivo (Old 10)": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_63_long, "short_entry": s2_63_short, "exit": s2_63_exit},
    "2.64 Scalping Reversal VWAP-Stoch (Old 11)": {"timeframe": "15m", "min_candles": 200, "long_entry": s2_64_long, "short_entry": s2_64_short, "exit": s2_64_exit},
    "2.65 TEMA Heikin Sim": {"timeframe": "5m", "min_candles": 200, "long_entry": s2_65_long, "short_entry": s2_65_short, "exit": s2_65_exit},
    # --- BREAKOUT (30) ---
    "3.01 BB Breakout": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_01_long, "short_entry": s3_01_short, "exit": s3_01_exit},
    "3.02 BB Squeeze": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_02_long, "short_entry": s3_02_short, "exit": s3_02_exit},
    "3.03 BB + Vol": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_03_long, "short_entry": s3_03_short, "exit": s3_03_exit},
    "3.04 BB + ADX": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_04_long, "short_entry": s3_04_short, "exit": s3_04_exit},
    "3.05 BB Rejection": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_05_long, "short_entry": s3_05_short, "exit": s3_05_exit},
    "3.06 BB + RSI": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_06_long, "short_entry": s3_06_short, "exit": s3_06_exit},
    "3.07 BB + CCI": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_07_long, "short_entry": s3_07_short, "exit": s3_07_exit},
    "3.08 BB + MACD": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_08_long, "short_entry": s3_08_short, "exit": s3_08_exit},
    "3.09 BB + ROC": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_09_long, "short_entry": s3_09_short, "exit": s3_09_exit},
    "3.10 BB HighLow": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_10_long, "short_entry": s3_10_short, "exit": s3_10_exit},
    "3.11 Donchian Break": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_11_long, "short_entry": s3_11_short, "exit": s3_11_exit},
    "3.12 Keltner Break": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_12_long, "short_entry": s3_12_short, "exit": s3_12_exit},
    "3.13 Donchian + Vol": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_13_long, "short_entry": s3_13_short, "exit": s3_13_exit},
    "3.14 Keltner + ADX": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_14_long, "short_entry": s3_14_short, "exit": s3_14_exit},
    "3.15 Donchian + RSI": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_15_long, "short_entry": s3_15_short, "exit": s3_15_exit},
    "3.16 Keltner + MACD": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_16_long, "short_entry": s3_16_short, "exit": s3_16_exit},
    "3.17 Donchian/BB": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_17_long, "short_entry": s3_17_short, "exit": s3_17_exit},
    "3.18 Keltner/BB": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_18_long, "short_entry": s3_18_short, "exit": s3_18_exit},
    "3.19 Donchian HL": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_19_long, "short_entry": s3_19_short, "exit": s3_19_exit},
    "3.20 Keltner CCI": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_20_long, "short_entry": s3_20_short, "exit": s3_20_exit},
    "3.21 Volatility 1.5x": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_21_long, "short_entry": s3_21_short, "exit": s3_21_exit},
    "3.22 Volatility 1.6x": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_22_long, "short_entry": s3_22_short, "exit": s3_22_exit},
    "3.23 Volatility 1.7x": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_23_long, "short_entry": s3_23_short, "exit": s3_23_exit},
    "3.24 Volatility 1.8x": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_24_long, "short_entry": s3_24_short, "exit": s3_24_exit},
    "3.24.01 Volat1.8x + Gold Cross": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_24_01_long, "short_entry": s3_24_short, "exit": s3_24_exit},
    "3.25 Volatility 1.9x": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_25_long, "short_entry": s3_25_short, "exit": s3_25_exit},
    "3.26 Volatility 2.0x": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_26_long, "short_entry": s3_26_short, "exit": s3_26_exit},
    "3.26.01 Volat2x + ADX > 20": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_26_01_long, "short_entry": s3_26_short, "exit": s3_26_exit},
    "3.27 Volatility 2.1x": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_27_long, "short_entry": s3_27_short, "exit": s3_27_exit},
    "3.28 Volatility 2.2x": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_28_long, "short_entry": s3_28_short, "exit": s3_28_exit},
    "3.29 Volatility 2.3x": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_29_long, "short_entry": s3_29_short, "exit": s3_29_exit},
    "3.30 Volatility 2.5x": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_30_long, "short_entry": s3_30_short, "exit": s3_30_exit},
    "3.31 BB Retest Breakout": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_31_long, "short_entry": s3_31_short, "exit": s3_31_exit},
    "3.32 Strong Confirm Breakout": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_32_long, "short_entry": s3_32_short, "exit": s3_32_exit},
    "3.33 BB Breakout + Volume": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_33_long, "short_entry": s3_33_short, "exit": s3_33_exit},
    "3.34 Micro Breakout + Vol": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_34_long, "short_entry": s3_34_short, "exit": s3_34_exit},
    "3.35 KC Breakout + DI": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_35_long, "short_entry": s3_35_short, "exit": s3_35_exit},
    "3.36 Donchian Retest": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_36_long, "short_entry": s3_36_short, "exit": s3_36_exit},
    "3.37 High-Tight Flag": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_37_long, "short_entry": s3_37_short, "exit": s3_37_exit},
    "3.38 BB Squeeze Explosion": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_38_long, "short_entry": s3_38_short, "exit": s3_38_exit},
    "3.39 DI Breakout": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_39_long, "short_entry": s3_39_short, "exit": s3_39_exit},
    "3.40 VWAP Donchian Breakout": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_40_long, "short_entry": s3_40_short, "exit": s3_40_exit},
    "3.41 VWAP Squeeze Breakout": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_41_long, "short_entry": s3_41_short, "exit": s3_41_exit},
    "3.42 VWAP Retest Breakout": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_42_long, "short_entry": s3_42_short, "exit": s3_42_exit},
    "3.43 VWAP Squeeze Breakout": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_43_long, "short_entry": s3_43_short, "exit": s3_43_exit},
    "3.44 High Vol VWAP Breakout": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_44_long, "short_entry": s3_44_short, "exit": s3_44_exit},
    "3.44.01 High Vol + Aroon Dom": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_44_01_long, "short_entry": s3_44_short, "exit": s3_44_exit},
    "3.45 Donchian VWAP Explosion": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_45_long, "short_entry": s3_45_short, "exit": s3_45_exit},
    "3.46 BB Squeeze Vol Conf": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_46_long, "short_entry": s3_46_short, "exit": s3_46_exit},
    "3.47 DC Breakout DI Vol": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_47_long, "short_entry": s3_47_short, "exit": s3_47_exit},
    "3.48 KC Breakout RSI ADX": {"timeframe": "1h", "min_candles": 200, "long_entry": s3_48_long, "short_entry": s3_48_short, "exit": s3_48_exit},
    "3.49 ATR Extreme Breakout": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_49_long, "short_entry": s3_49_short, "exit": s3_49_exit},
    "3.50 BB Squeeze Retest": {"timeframe": "1h", "min_candles": 200, "long_entry": s3_50_long, "short_entry": s3_50_short, "exit": s3_50_exit},
    "3.51 DC Breakout EMA ADX": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_51_long, "short_entry": s3_51_short, "exit": s3_51_exit},
    "3.52 VWAP Squeeze Breakout": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_52_long, "short_entry": s3_52_short, "exit": s3_52_exit},
    "3.53 KC Breakout HL RSI": {"timeframe": "1h", "min_candles": 200, "long_entry": s3_53_long, "short_entry": s3_53_short, "exit": s3_53_exit},
    "3.54 Double Channel Break": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_54_long, "short_entry": s3_54_short, "exit": s3_54_exit},
    "3.55 BB Break VWAP MACD": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_55_long, "short_entry": s3_55_short, "exit": s3_55_exit},
    "3.56 Breakout Volume (Old 4)": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_56_long, "short_entry": s3_56_short, "exit": s3_56_exit},
    # --- MIXED (40) ---
    "4.04 Mixed Vol 1.7x / VWAP Sqz": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_23_long, "short_entry": s3_41_short, "exit": s4_04_exit},
    "4.39 Mixed BB Retest Breakout / BB-Middle Reclaim": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_31_long, "short_entry": s1_40_short, "exit": s4_39_exit},
    "4.01 Mixed BB + Short Sniper": {"timeframe": "15m", "min_candles": 200, "long_entry": s3_31_long, "short_entry": s1_40_02_short, "exit": s4_39_exit},
    '5.01 Chikou Dual-Edge Reversal': {'timeframe': '15m', 'long_entry': s5_01_long, 'short_entry': s5_01_short, 'exit': False},
    'SOL v1': {'timeframe': '15m', 'long_entry': s_pnl_long_29, 'short_entry': s_pnl_short_203, 'exit': False},
}



def get_btc_trend_filter(data_source_exchanges: list) -> dict:
    """
    Calcula a tendência do BTC baseada em múltiplos cruzamentos de EMA.
    Retorna um dicionário com o estado de cada par de EMAs.
    """
    btc_data = None
    for exchange in data_source_exchanges:
        try:
            # Busca dados suficientes para a maior média (14) + estabilização
            data = get_historical_klines('BTC/USDT', '1h', limit=200, exchange_name=exchange)
            if data is not None and not data.empty:
                btc_data = data
                break
        except Exception as e:
            print(f"[BTC Trend] Falha ao obter dados de {exchange}: {e}")
            continue

    # Estrutura padrão de retorno caso falhe
    default_trends = {
        "ema_3_8": "long_short",
        "ema_8_13": "long_short",
        "ema_7_14": "long_short"
    }

    if btc_data is None or btc_data.empty or 'close' not in btc_data.columns:
        print("Não foi possível obter dados válidos do BTC. Filtro desativado temporariamente.")
        return default_trends

    # Cálculo das EMAs necessárias (3, 7, 8, 13, 14)
    btc_data['EMA_3'] = btc_data['close'].ewm(span=3, adjust=False).mean()
    btc_data['EMA_7'] = btc_data['close'].ewm(span=7, adjust=False).mean()
    btc_data['EMA_8'] = btc_data['close'].ewm(span=8, adjust=False).mean()
    btc_data['EMA_13'] = btc_data['close'].ewm(span=13, adjust=False).mean()
    btc_data['EMA_14'] = btc_data['close'].ewm(span=14, adjust=False).mean()

    # Pega apenas os valores do ÚLTIMO candle fechado
    last_row = btc_data.iloc[-1]

    trends = {}

    # Lógica para EMA 3 e 8
    if last_row['EMA_3'] > last_row['EMA_8']:
        trends['ema_3_8'] = "long_only"
    elif last_row['EMA_3'] < last_row['EMA_8']:
        trends['ema_3_8'] = "short_only"
    else:
        trends['ema_3_8'] = "long_short"

    # Lógica para EMA 8 e 13
    if last_row['EMA_8'] > last_row['EMA_13']:
        trends['ema_8_13'] = "long_only"
    elif last_row['EMA_8'] < last_row['EMA_13']:
        trends['ema_8_13'] = "short_only"
    else:
        trends['ema_8_13'] = "long_short"

    # Lógica para EMA 7 e 14 (Padrão Antigo)
    if last_row['EMA_7'] > last_row['EMA_14']:
        trends['ema_7_14'] = "long_only"
    elif last_row['EMA_7'] < last_row['EMA_14']:
        trends['ema_7_14'] = "short_only"
    else:
        trends['ema_7_14'] = "long_short"

    return trends
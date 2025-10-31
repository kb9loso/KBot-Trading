# strategies.py
import pandas as pd
import numpy as np
from data_fetcher import get_historical_klines

STRATEGIES = {
    "1. Tendencia Rapida (15m)": {
        "description": "Busca entrar no início de tendências rápidas, filtrado pela EMA 200 para operar a favor da tendência principal.",
        "timeframe": "15m",
        "min_candles": 200, # Aumentado para EMA 200
        "long_entry": lambda c, p: c['close'] > c['EMA_200'] and p['EMA_9'] < p['EMA_21'] and c['EMA_9'] > c['EMA_21'] and c['RSI_9'] > 55 and c['ADX'] > 20,
        "short_entry": lambda c, p: c['close'] < c['EMA_200'] and p['EMA_9'] > p['EMA_21'] and c['EMA_9'] < c['EMA_21'] and c['RSI_9'] < 45 and c['ADX'] > 20,
        "exit": lambda c, p, side: (c['EMA_9'] < c['EMA_21'] and c['RSI_9'] < 48) if side == 'long' else (c['EMA_9'] > c['EMA_21'] and c['RSI_9'] > 52)
    },
    "2. Momentum Explosivo (15m)": {
        "description": "Entra em movimentos fortes quando o MACD cruza, filtrado pela EMA 200.",
        "timeframe": "15m",
        "min_candles": 200, # Aumentado para EMA 200
        "long_entry": lambda c, p: c['close'] > c['EMA_200'] and p['MACD'] < p['MACD_signal'] and c['MACD'] > c['MACD_signal'] and c['MACD'] > 0 and c['ADX'] > 20 and c['close'] > c['VWAP'],
        "short_entry": lambda c, p: c['close'] < c['EMA_200'] and p['MACD'] > p['MACD_signal'] and c['MACD'] < c['MACD_signal'] and c['MACD'] < 0 and c['ADX'] > 20 and c['close'] < c['VWAP'],
        "exit": lambda c, p, side: (c['MACD'] < 0) if side == 'long' else (c['MACD'] > 0)
    },
    "3. Scalping Intraday (15m)": {
        "description": "Estratégia de scalping que busca saídas de condições de sobrevenda/sobrecompra a favor de uma microtendência.",
        "timeframe": "15m",
        "min_candles": 30,
        "long_entry": lambda c, p: c['EMA_9'] > c['EMA_21'] and p['STOCH_K'] < 20 and c['STOCH_K'] > p['STOCH_D'] and c['close'] > c['VWAP'],
        "short_entry": lambda c, p: c['EMA_9'] < c['EMA_21'] and p['STOCH_K'] > 80 and c['STOCH_K'] < p['STOCH_D'] and c['close'] < c['VWAP'],
        "exit": lambda c, p, side: (c['STOCH_K'] > 75 and c['STOCH_K'] < c['STOCH_D']) if side == 'long' else (c['STOCH_K'] < 25 and c['STOCH_K'] > c['STOCH_D'])
    },
    "4. Breakout Curto (15m)": {
        "description": "Versão robusta: busca rompimentos confirmados (Bollinger + Volume + MACD), mas filtra ruído com ATR e tendência (EMA_50).",
        "timeframe": "15m",
        "min_candles": 150,
        "long_entry": lambda c, p: p['close'] > p['BB_upper'] and c['close'] > p['close'] and c['close'] > c['EMA_50'] and c['volume'] > c['Volume_MA'] * 1.5 and c['MACD'] > c['MACD_signal'] and c['ATR'] < c['ATR_MA'] * 1.3,
        "short_entry": lambda c, p: p['close'] < p['BB_lower'] and c['close'] < p['close'] and c['close'] < c['EMA_50'] and c['volume'] > c['Volume_MA'] * 1.5 and c['MACD'] < c['MACD_signal'] and c['ATR'] < c['ATR_MA'] * 1.3,
        "exit": lambda c, p, side: c['BBW'] < c['BBW_min_120'] * 1.2 or c['ATR'] < c['ATR_MA'] * 0.9
    },
    "5. Swing Curto (1h)": {
        "description": "Estratégia de Swing Trade que busca pegar o corpo principal de tendências, agora filtrada pela EMA 200.",
        "timeframe": "1h",
        "min_candles": 200, # Aumentado para EMA 200
        "long_entry": lambda c, p: c['close'] > c['EMA_200'] and p['EMA_21'] < p['EMA_50'] and c['EMA_21'] > c['EMA_50'] and c['ADX'] > 25,
        "short_entry": lambda c, p: c['close'] < c['EMA_200'] and p['EMA_21'] > p['EMA_50'] and c['EMA_21'] < c['EMA_50'] and c['ADX'] > 25,
        "exit": lambda c, p, side: (c['EMA_9'] < c['EMA_21']) if side == 'long' else (c['EMA_9'] > c['EMA_21'])
    },
    "6. Reversao Forte (1h)": {
        "description": "Busca reversões em extremos de mercado. Não utiliza filtro de tendência para poder operar contra ela.",
        "timeframe": "1h",
        "min_candles": 30,
        "long_entry": lambda c, p: p['close'] < p['BB_lower'] and c['close'] > p['BB_lower'] and c['RSI_9'] < 30,
        "short_entry": lambda c, p: p['close'] > p['BB_upper'] and c['close'] < c['BB_upper'] and c['RSI_9'] > 70,
        "exit": lambda c, p, side: (c['close'] > c['BB_middle']) if side == 'long' else (c['close'] < c['BB_middle'])
    },
    "7. Tendencia Confirmada (1h)": {
        "description": "Entra em tendências fortes e estabelecidas, aproveitando pullbacks, agora com filtro da EMA 200.",
        "timeframe": "1h",
        "min_candles": 200, # Aumentado para EMA 200
        "long_entry": lambda c, p: c['close'] > c['EMA_200'] and c['EMA_9'] > c['EMA_21'] and c['EMA_21'] > c['EMA_50'] and p['low'] <= p['EMA_21'] and c['close'] > c['EMA_21'] and c['ADX'] > 25,
        "short_entry": lambda c, p: c['close'] < c['EMA_200'] and c['EMA_9'] < c['EMA_21'] and c['EMA_21'] < c['EMA_50'] and p['high'] >= p['EMA_21'] and c['close'] < c['EMA_21'] and c['ADX'] > 25,
        "exit": lambda c, p, side: (c['EMA_21'] < c['EMA_50']) if side == 'long' else (c['EMA_21'] > c['EMA_50'])
    },
    "8. Scalper Volume (1m)": {
        "description": "Scalping de altíssima frequência. Não utiliza filtro de tendência para maior agilidade.",
        "timeframe": "1m",
        "min_candles": 50,
        "long_entry": lambda c, p: (c['close'] > c['EMA_9'] and c['volume'] > c['Volume_MA'] * 1.8 and c['RSI_9'] < 58 and c['STOCH_K'] < 80),
        "short_entry": lambda c, p: (c['close'] < c['EMA_9'] and c['volume'] > c['Volume_MA'] * 1.8 and c['RSI_9'] > 42 and c['STOCH_K'] > 20),
        "exit": lambda c, p, side: ((c['close'] < c['EMA_9'] or c['STOCH_K'] > 80) if side == 'long' else (c['close'] > c['EMA_9'] or c['STOCH_K'] < 20))
    },
    "9. MACD/RSI Trend Follower (1h)": {
        "description": "Segue tendências com confirmação múltipla, agora com o filtro adicional da EMA 200.",
        "timeframe": "1h",
        "min_candles": 200, # Aumentado para EMA 200
        "long_entry": lambda c, p: c['close'] > c['EMA_200'] and (p['MACD'] < p['MACD_signal'] and c['MACD'] > c['MACD_signal'] and c['MACD'] > 0 and c['RSI_14'] > 52 and c['ADX'] > 20),
        "short_entry": lambda c, p: c['close'] < c['EMA_200'] and (p['MACD'] > p['MACD_signal'] and c['MACD'] < c['MACD_signal'] and c['MACD'] < 0 and c['RSI_14'] < 48 and c['ADX'] > 20),
        "exit": lambda c, p, side: ((c['MACD'] < c['MACD_signal'] and c['RSI_14'] < 50) if side == 'long' else (c['MACD'] > c['MACD_signal'] and c['RSI_14'] > 50))
    },
    "10. Momentum Flip (1m)": {
        "description": "Prevê a direção do próximo candle com base em momentum, volume e reversão leve. Entra e sai no próximo candle. Ativos testados (BTC e XRP). Utilizar somente com taxa 0.",
        "timeframe": "1m",
        "min_candles": 50,
        "long_entry": lambda c, p: c['close'] > c['open'] * 1.0003 and c['close'] > c['EMA_9'] and c['MACD'] > c['MACD_signal'] and c['RSI_9'] > 50,
        "short_entry": lambda c, p: c['close'] < c['open'] * 0.9997 and c['close'] < c['EMA_9'] and c['MACD'] < c['MACD_signal'] and c['RSI_9'] < 50,
        "exit": lambda c, p, side: True,
    },

    "11. Volume Seguro (5m)": {
        "description": "Foca em trades de qualidade com filtros simples para gerar volume sustentável SL (0.008, 0.01, 0.012) TP (2.0, 2.5, 3.0)",
        "timeframe": "5m",
        "min_candles": 100,
        "long_entry": lambda c, p: c['close'] > c['EMA_21'] and c['EMA_21'] > c['EMA_50'] and c['volume'] > c['Volume_MA'] * 1.2 and c['RSI_9'] > 40 and c['RSI_9'] < 70 and c['MACD'] > c['MACD_signal'] and c['ADX'] > 10,
        "short_entry": lambda c, p: c['close'] < c['EMA_21'] and c['EMA_21'] < c['EMA_50'] and c['volume'] > c['Volume_MA'] * 1.2 and c['RSI_9'] > 30 and c['RSI_9'] < 60 and c['MACD'] < c['MACD_signal'] and c['ADX'] > 10,
        "exit": lambda c, p, side: (c['RSI_9'] > 75 or c['close'] < c['EMA_21']) if side == 'long' else (c['RSI_9'] < 25 or c['close'] > c['EMA_21']),
    },

    "12. Volume Plus (3m)": {
        "description": "Otimizada para mais trades mantendo rentabilidade - foco em volume sustentável SL (0.008, 0.01, 0.012) TP (2.0, 2.5, 3.0)",
        "timeframe": "3m",
        "min_candles": 80,
        "long_entry": lambda c, p: c['close'] > c['EMA_21'] and c['EMA_21'] > c['EMA_50'] and c['volume'] > c['Volume_MA'] * 1.1 and c['RSI_9'] > 35 and c['RSI_9'] < 75 and c['MACD'] > c['MACD_signal'] and c['ADX'] > 8,
        "short_entry": lambda c, p: c['close'] < c['EMA_21'] and c['EMA_21'] < c['EMA_50'] and c['volume'] > c['Volume_MA'] * 1.1 and c['RSI_9'] > 25 and c['RSI_9'] < 65 and c['MACD'] < c['MACD_signal'] and c['ADX'] > 8,
        "exit": lambda c, p, side: (c['RSI_9'] > 80 or c['close'] < c['EMA_9']) if side == 'long' else (c['RSI_9'] < 20 or c['close'] > c['EMA_9']),
    },
}

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # ============ MÉDIAS MÓVEIS EXPONENCIAIS ============
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
    # Placeholder para EMA de timeframe maior (ex: 4h ou diário)
    df['EMA_50_htf'] = df['EMA_50'].shift(1)

    # ============ RSI ============
    def calc_rsi(series, length=14):
        delta = series.diff(1)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/length, min_periods=length).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length).mean()
        rs = avg_gain / (avg_loss.replace(0, 1e-10))
        return 100 - (100 / (1 + rs))

    df['RSI_14'] = calc_rsi(df['close'], 14)
    df['RSI_9'] = calc_rsi(df['close'], 9)

    # ============ OBV ============
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(window=20).mean()

    # ============ BANDAS DE BOLLINGER ============
    period_bb = 20
    df['BB_middle'] = df['close'].rolling(window=period_bb).mean()
    df['BB_std'] = df['close'].rolling(window=period_bb).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    df['BBW'] = (df['BB_upper'] - df['BB_lower']) / (df['BB_middle'].replace(0, 1e-10))
    df['BBW_min_120'] = df['BBW'].rolling(window=120).min()

    # ============ ATR ============
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.ewm(alpha=1/14, adjust=False).mean()
    df['ATR_MA'] = df['ATR'].rolling(window=50).mean()

    # ============ MACD ============
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ============ ADX ============
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[(plus_dm < minus_dm) | (plus_dm < 0)] = 0
    minus_dm[(minus_dm < plus_dm) | (minus_dm < 0)] = 0

    tr_components = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1)

    tr = tr_components.max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / (atr.replace(0, 1e-10)))
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / (atr.replace(0, 1e-10)))
    dx = (abs(plus_di - minus_di) / ((plus_di + minus_di).replace(0, 1e-10))) * 100
    df['ADX'] = dx.ewm(alpha=1/14).mean()

    # ============ ESTOCÁSTICO ============
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['STOCH_K'] = 100 * ((df['close'] - low_14) / ((high_14 - low_14).replace(0, 1e-10)))
    df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()

    # ============ VWAP ============
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap_num = (typical_price * df['volume']).cumsum()
    vwap_den = df['volume'].cumsum().replace(0, 1e-10)
    df['VWAP'] = vwap_num / vwap_den

    # ============ MÉDIA DE VOLUME ============
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()

    # ============ VALORES ANTERIORES ============
    prev_cols = [
        'close', 'high', 'low', 'volume', 'EMA_9', 'EMA_21', 'EMA_50',
        'RSI_9', 'RSI_14', 'MACD', 'MACD_signal', 'STOCH_K', 'STOCH_D',
        'VWAP', 'Volume_MA', 'ATR', 'ATR_MA', 'BBW'
    ]
    for col in prev_cols:
        df[f'{col}_prev'] = df[col].shift(1)

    return df


def get_btc_trend_filter(data_source_exchanges: list) -> str:
    btc_data = None
    used_exchange = None
    for exchange in data_source_exchanges:
        try:
            data = get_historical_klines('BTC/USDT', '1h', limit=200, exchange_name=exchange)
            if data is not None and not data.empty:
                btc_data = data
                used_exchange = exchange
                break
        except Exception as e:
            print(f"[BTC Trend] Falha ao obter dados de {exchange}: {e}")
            continue

    if btc_data is None or btc_data.empty or 'close' not in btc_data.columns:
        print("Não foi possível obter dados válidos do BTC. Filtro desativado temporariamente.")
        return "long_short"

    btc_data['EMA_Short'] = btc_data['close'].ewm(span=7, adjust=False).mean()
    btc_data['EMA_Long'] = btc_data['close'].ewm(span=14, adjust=False).mean()

    # Sinal bruto (1 para alta, -1 para baixa)
    raw_signal = np.where(btc_data['EMA_Short'] > btc_data['EMA_Long'], 1, -1)
    recent = pd.Series(raw_signal).iloc[-3:].mean()

    if recent > 0.2:
        trend = "long_only"
    elif recent < -0.2:
        trend = "short_only"
    else:
        trend = "long_short"

    return trend
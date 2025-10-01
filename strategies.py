# strategies.py
import pandas as pd
import numpy as np

STRATEGIES = {
    "1. Tendencia Rapida (15m)": {
        "description": "Busca entrar no início de tendências rápidas, confirmando a força com ADX e o momento com RSI.",
        "timeframe": "15m",
        "min_candles": 50,
        "long_entry": lambda c, p: p['EMA_9'] < p['EMA_21'] and c['EMA_9'] > c['EMA_21'] and c['RSI_9'] > 55 and c['ADX'] > 20,
        "short_entry": lambda c, p: p['EMA_9'] > p['EMA_21'] and c['EMA_9'] < c['EMA_21'] and c['RSI_9'] < 45 and c['ADX'] > 20,
        # Saída aprimorada: exige que o RSI também confirme a perda de momentum.
        "exit": lambda c, p, side: (c['EMA_9'] < c['EMA_21'] and c['RSI_9'] < 48) if side == 'long' else (c['EMA_9'] > c['EMA_21'] and c['RSI_9'] > 52)
    },
    "2. Momentum Explosivo (15m)": {
        "description": "Entra em movimentos fortes quando o MACD cruza, com confirmação de tendência (ADX) e viés de preço (VWAP).",
        "timeframe": "15m",
        "min_candles": 50,
        "long_entry": lambda c, p: p['MACD'] < p['MACD_signal'] and c['MACD'] > c['MACD_signal'] and c['MACD'] > 0 and c['ADX'] > 20 and c['close'] > c['VWAP'],
        "short_entry": lambda c, p: p['MACD'] > p['MACD_signal'] and c['MACD'] < c['MACD_signal'] and c['MACD'] < 0 and c['ADX'] > 20 and c['close'] < c['VWAP'],
        # Saída aprimorada: exige que o MACD não apenas cruze a linha de sinal, mas também o eixo zero.
        "exit": lambda c, p, side: (c['MACD'] < 0) if side == 'long' else (c['MACD'] > 0)
    },
    "3. Scalping Intraday (15m)": {
        "description": "Estratégia de scalping que busca saídas de condições de sobrevenda/sobrecompra a favor de uma microtendência.",
        "timeframe": "15m",
        "min_candles": 30,
        "long_entry": lambda c, p: c['EMA_9'] > c['EMA_21'] and p['STOCH_K'] < 20 and c['STOCH_K'] > p['STOCH_D'] and c['close'] > c['VWAP'],
        "short_entry": lambda c, p: c['EMA_9'] < c['EMA_21'] and p['STOCH_K'] > 80 and c['STOCH_K'] < p['STOCH_D'] and c['close'] < c['VWAP'],
        # Saída aprimorada: Sai quando o estocástico cruza sua linha de sinal (%D) na zona oposta, confirmando a exaustão.
        "exit": lambda c, p, side: (c['STOCH_K'] > 75 and c['STOCH_K'] < c['STOCH_D']) if side == 'long' else (c['STOCH_K'] < 25 and c['STOCH_K'] > c['STOCH_D'])
    },
    "4. Breakout Curto (15m)": {
        "description": "Entra em rompimentos de volatilidade (Bandas de Bollinger) confirmados por volume e momento (MACD).",
        "timeframe": "15m",
        "min_candles": 30,
        "long_entry": lambda c, p: c['close'] > c['BB_upper'] and c['volume'] > c['Volume_MA'] * 1.5 and c['MACD'] > c['MACD_signal'],
        "short_entry": lambda c, p: c['close'] < c['BB_lower'] and c['volume'] > c['Volume_MA'] * 1.5 and c['MACD'] < c['MACD_signal'],
        # Saída Lógica: Sair quando o preço retorna à média é uma boa prática para breakouts. Sem alteração.
        "exit": lambda c, p, side: (c['close'] < c['BB_middle']) if side == 'long' else (c['close'] > c['BB_middle'])
    },
    "5. Swing Curto (1h)": {
        "description": "Estratégia de Swing Trade que busca pegar o corpo principal de tendências de médio prazo no gráfico de 1 hora.",
        "timeframe": "1h",
        "min_candles": 60,
        "long_entry": lambda c, p: p['EMA_21'] < p['EMA_50'] and c['EMA_21'] > c['EMA_50'] and c['ADX'] > 25,
        "short_entry": lambda c, p: p['EMA_21'] > p['EMA_50'] and c['EMA_21'] < c['EMA_50'] and c['ADX'] > 25,
        # Saída aprimorada: Usa um cruzamento de médias mais rápido (9/21) como um "aviso prévio" para sair da tendência.
        "exit": lambda c, p, side: (c['EMA_9'] < c['EMA_21']) if side == 'long' else (c['EMA_9'] > c['EMA_21'])
    },
    "6. Reversao Forte (1h)": {
        "description": "Busca reversões em extremos de mercado, mas aguarda uma confirmação de que o preço está reagindo antes de entrar.",
        "timeframe": "1h",
        "min_candles": 30,
        "long_entry": lambda c, p: p['close'] < p['BB_lower'] and c['close'] > p['BB_lower'] and c['RSI_9'] < 30,
        "short_entry": lambda c, p: p['close'] > p['BB_upper'] and c['close'] < c['BB_upper'] and c['RSI_9'] > 70,
        # Saída Lógica: Sair quando o preço atinge a média é o objetivo da reversão. Sem alteração.
        "exit": lambda c, p, side: (c['close'] > c['BB_middle']) if side == 'long' else (c['close'] < c['BB_middle'])
    },
    "7. Tendencia Confirmada (1h)": {
        "description": "Entra em tendências fortes e estabelecidas, aproveitando pullbacks (correções) até a média móvel para comprar.",
        "timeframe": "1h",
        "min_candles": 60,
        "long_entry": lambda c, p: c['EMA_9'] > c['EMA_21'] and c['EMA_21'] > c['EMA_50'] and p['low'] <= p['EMA_21'] and c['close'] > c['EMA_21'] and c['ADX'] > 25,
        "short_entry": lambda c, p: c['EMA_9'] < c['EMA_21'] and c['EMA_21'] < c['EMA_50'] and p['high'] >= p['EMA_21'] and c['close'] < c['EMA_21'] and c['ADX'] > 25,
        # Saída aprimorada: O sinal de saída é o enfraquecimento da tendência de médio prazo (cruzamento 21/50), não apenas a de curto prazo.
        "exit": lambda c, p, side: (c['EMA_21'] < c['EMA_50']) if side == 'long' else (c['EMA_21'] > c['EMA_50'])
    },
    "8. Scalper Volume (1m)": {
        "description": "Estratégia de scalping visando muitas operações, saídas rápidas. Direcional: long ou short.",
        "timeframe": "1m",
        "min_candles": 50,
        "long_entry": lambda c, p: c['close'] > c['EMA_9'] and c['volume'] > c['Volume_MA'] * 1.5 and c['RSI_9'] < 60,
        "short_entry": lambda c, p: c['close'] < c['EMA_9'] and c['volume'] > c['Volume_MA'] * 1.5 and c['RSI_9'] > 40,
        # Saída Lógica: A lógica original já era robusta para scalping (saída rápida por preço ou exaustão). Sem alteração.
        "exit": lambda c, p, side: (c['close'] < c['EMA_9'] or c['STOCH_K'] > 80) if side == 'long' else (c['close'] > c['EMA_9'] or c['STOCH_K'] < 20)
    },
    "9. MACD/RSI Trend Follower (1h)": {
        "description": "Entrada quando o MACD cruza confirmando força do RSI, filtrado por tendência válida (ADX).",
        "timeframe": "1h",
        "min_candles": 100,  # MACD(26), RSI(14) e ADX(14)
        "long_entry": lambda c, p: (p['MACD'] < p['MACD_signal'] and c['MACD'] > c['MACD_signal'] and c['MACD'] > 0 and c['RSI_14'] > 52 and c['ADX'] > 20), # confirma tendência
        "short_entry": lambda c, p: (p['MACD'] > p['MACD_signal'] and c['MACD'] < c['MACD_signal'] and c['MACD'] < 0 and c['RSI_14'] < 48 and c['ADX'] > 20),
        "exit": lambda c, p, side: ((c['MACD'] < c['MACD_signal'] and c['RSI_14'] < 50) if side == 'long' else (c['MACD'] > c['MACD_signal'] and c['RSI_14'] > 50))
    }
    # ,
    #
    # "10. Bollinger Squeeze Breakout (1h)": {
    #     "description": "Detecta squeeze de volatilidade e entra no rompimento com confirmação de volume.",
    #     "timeframe": "1h",
    #     "min_candles": 150,  # precisa do histórico do BBW + volume médio
    #     "long_entry": lambda c, p: (p['BBW'] < p['BBW_min_120'] and c['close'] > c['BB_upper'] and c['volume'] > p['volume_mean_20']),
    #     "short_entry": lambda c, p: (p['BBW'] < p['BBW_min_120'] and c['close'] < c['BB_lower'] and c['volume'] > p['volume_mean_20']
    #     ),
    #     "exit": lambda c, p, side: ((c['close'] < c['BB_middle']) if side == 'long' else (c['close'] > c['BB_middle']))
    # }
}


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos os indicadores necessários para as estratégias."""
    # EMAs
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # EMA de timeframe maior (ex: 4h/dia) -> mock, precisa calcular externamente
    df['EMA_50_htf'] = df['EMA_50'].shift(1)  # Placeholder (usar candles de timeframe maior no real)

    # RSI (padrão 14 e curto 9)
    for length in [14, 9]:
        delta = df['close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
        avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
        rs = avg_gain / avg_loss
        df[f'RSI_{length}'] = 100 - (100 / (1 + rs))

    # OBV
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(window=20).mean()

    # Bandas de Bollinger
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    # df['BBW'] = (df['BB_upper'] - df['BB_lower']) / (df['BB_middle'] + 1e-12)
    # df['BBW_min_120'] = df['BBW'].rolling(window=120).min()

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ADX
    plus_dm = df['high'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm = df['low'].diff()
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat(
        [df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))],
        axis=1).max(axis=1)
    atr_adx = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / 14).mean() / atr_adx)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1 / 14).mean() / atr_adx))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.ewm(alpha=1 / 14).mean()

    # Estocástico
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['STOCH_K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()

    # VWAP
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

    # Média de Volume
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()

    return df.dropna().copy()

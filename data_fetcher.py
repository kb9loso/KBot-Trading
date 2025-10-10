# KBot-Trading/data_fetcher.py
import pandas as pd
import ccxt
from datetime import datetime


def get_historical_klines(
        market: str,
        timeframe: str,
        limit: int = 1000,
        exchange_name: str = 'binance',
        start_timestamp_ms: int = None
) -> pd.DataFrame | None:
    """
    Busca o histórico de velas (klines) para um mercado de uma exchange específica.
    Este metodo é genérico e pode ser usado para qualquer exchange suportada pela CCXT.

    :param market: O símbolo do mercado (ex: 'BTC/USDT').
    :param timeframe: O intervalo da vela (ex: '15m', '1h', '1d').
    :param limit: O número de velas a serem buscadas.
    :param exchange_name: O nome da exchange (ex: 'binance', 'bybit', 'apex').
    :param start_timestamp_ms: O timestamp inicial em milissegundos para a busca.
    :return: Um DataFrame do pandas com os dados ou None em caso de erro.
    """
    try:
        # --- Configurações Específicas por Exchange ---
        if exchange_name.lower() == 'binance':
            exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        elif exchange_name.lower() == 'bybit':
            exchange = ccxt.bybit({'options': {'defaultType': 'linear'}})
        else:
            # Para outras exchanges como a Apex, usa a inicialização padrão
            exchange = getattr(ccxt, exchange_name.lower())()

        if not exchange.has['fetchOHLCV']:
            print(f"A exchange '{exchange_name}' não suporta a busca de klines (OHLCV).")
            return None

        # --- Busca Paginada para Grandes Períodos ---
        all_klines = []
        since = start_timestamp_ms

        while True:
            klines = exchange.fetch_ohlcv(market, timeframe, since=since, limit=limit)
            if not klines:
                break
            all_klines.extend(klines)
            # Atualiza o 'since' para a próxima página de resultados
            since = klines[-1][0] + 1
            # Se o número de klines recebido for menor que o limite, significa que chegamos ao fim
            if len(klines) < limit:
                break

        if not all_klines:
            return None

        # --- Conversão para DataFrame ---
        df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        return df

    except Exception as e:
        print(f"Erro ao buscar dados históricos de '{exchange_name}' para {market}: {e}")
        return None
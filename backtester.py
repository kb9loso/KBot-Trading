# backtester.py
import json
import pandas as pd
import ccxt
import warnings
from datetime import datetime, timedelta

# Importa a lógica de indicadores e estratégias do arquivo central
from strategies import STRATEGIES, add_all_indicators

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# ==============================================================================
# ===================== PAINEL DE OTIMIZAÇÃO E CONTROLE ========================
# ==============================================================================

# Parâmetros padrão para a otimização
INITIAL_CAPITAL = 100.0
RISK_PER_TRADE_PCT = 0.01
STOP_LOSS_LEVELS_PCT = [0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
TAKE_PROFIT_RRRS = [2.0, 2.5, 3.0, 4.0]


# ==============================================================================
# ======================== FIM DO PAINEL DE CONTROLE ===========================
# ==============================================================================


def get_exchange_data(exchange_id, symbol, timeframe, since_timestamp):
    """Busca dados históricos de uma exchange usando ccxt a partir de um timestamp."""
    try:
        if exchange_id.lower() == 'binance':
            exchange = ccxt.binance({
                'options': {'defaultType': 'future'},
            })
        elif exchange_id.lower() == 'bybit':
            exchange = ccxt.bybit({
                'options': {'defaultType': 'linear'},
            })
        else:
            exchange = getattr(ccxt, exchange_id.lower())()
        if not exchange.has['fetchOHLCV']: return None

        all_klines = []
        while True:
            klines = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=1000)
            if not klines: break
            all_klines.extend(klines)
            since_timestamp = klines[-1][0] + 1

        df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Erro ao buscar dados: {e}")
        return None


# ==============================================================================
# ===================== FUNÇÃO run_backtest ATUALIZADA =========================
# ==============================================================================

def run_backtest(df, initial_capital, rrr, sl_pct, leverage, strategy, risk_per_trade):
    """Executa o backtest APENAS com Stop Loss e Take Profit."""
    capital = initial_capital
    position_size = 0.0
    position_type = 0  # 0: fora, 1: long, -1: short
    entry_price = 0
    trades = []
    stop_loss_price = 0
    take_profit_price = 0
    entry_date = None

    min_c = strategy.get('min_candles', 1)

    if len(df) < min_c:
        return initial_capital, pd.DataFrame(trades)

    for i in range(min_c, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # --- LÓGICA DE GESTÃO DE POSIÇÃO ABERTA (APENAS SL/TP) ---
        if position_type != 0:
            exit_price, result = None, None
            side = 'long' if position_type == 1 else 'short'

            # Verifica apenas as condições de Stop Loss e Take Profit
            if side == 'long':
                if current_row['low'] <= stop_loss_price:
                    exit_price, result = stop_loss_price, 'loss'
                elif current_row['high'] >= take_profit_price:
                    exit_price, result = take_profit_price, 'win'
            else:  # Short
                if current_row['high'] >= stop_loss_price:
                    exit_price, result = stop_loss_price, 'loss'
                elif current_row['low'] <= take_profit_price:
                    exit_price, result = take_profit_price, 'win'

            # Se uma condição de saída foi atingida, fecha a operação
            if exit_price is not None:
                pnl = (exit_price - entry_price) * position_size if position_type == 1 else (
                                                                                                        entry_price - exit_price) * position_size
                capital += pnl
                trades.append({'entry_date': entry_date, 'exit_date': current_row.name, 'entry_price': entry_price,
                               'exit_price': exit_price, 'result': result, 'pnl': pnl, 'type': side})

                position_type = 0

                if capital <= 0:
                    break

        # --- LÓGICA DE ENTRADA DE POSIÇÃO ---
        if position_type == 0 and capital > 0:
            open_pos = False
            if strategy.get('long_entry') and strategy['long_entry'](current_row, prev_row):
                position_type = 1
                open_pos = True
            elif strategy.get('short_entry') and strategy['short_entry'](current_row, prev_row):
                position_type = -1
                open_pos = True

            if open_pos:
                entry_price = current_row['close']
                sl_distance_in_price = entry_price * sl_pct

                amount_to_risk = capital * risk_per_trade
                if sl_distance_in_price > 0:
                    position_size = amount_to_risk / sl_distance_in_price
                else:
                    position_size = 0

                if position_type == 1:
                    stop_loss_price = entry_price - sl_distance_in_price
                    take_profit_price = entry_price + (sl_distance_in_price * rrr)
                else:
                    stop_loss_price = entry_price + sl_distance_in_price
                    take_profit_price = entry_price - (sl_distance_in_price * rrr)

                entry_date = current_row.name

    # --- LÓGICA DE FECHAMENTO AO FINAL DOS DADOS ---
    if position_type != 0:
        final_price = df.iloc[-1]['close']
        pnl = (final_price - entry_price) * position_size if position_type == 1 else (
                                                                                                 entry_price - final_price) * position_size
        capital += pnl
        side = 'long' if position_type == 1 else 'short'
        trades.append({
            'entry_date': entry_date,
            'exit_date': df.index[-1],
            'entry_price': entry_price,
            'exit_price': final_price,
            'result': 'win' if pnl > 0 else 'loss',
            'pnl': pnl,
            'type': side
        })

    return capital, pd.DataFrame(trades)


# ==============================================================================
# ======================== FIM DA FUNÇÃO ATUALIZADA ============================
# ==============================================================================


def analyze_performance(initial_capital, final_capital, trades_df):
    """Analisa e formata os resultados de um backtest, detalhando por long e short."""
    if trades_df is None or trades_df.empty:
        return {
            "Capital Final": initial_capital, "Retorno %": 0, "Total de Trades": 0, "Taxa de Acerto %": 0,
            "Long Trades": 0, "Long Win Rate %": 0, "Long Return %": 0,
            "Short Trades": 0, "Short Win Rate %": 0, "Short Return %": 0,
        }

    net_return = final_capital - initial_capital
    total_return_pct = (net_return / initial_capital) * 100 if initial_capital > 0 else 0
    total_trades = len(trades_df)
    total_win_rate = (trades_df['result'] == 'win').sum() / total_trades * 100 if total_trades > 0 else 0

    long_trades = trades_df[trades_df['type'] == 'long']
    long_total = len(long_trades)
    long_wins = (long_trades['result'] == 'win').sum()
    long_win_rate = (long_wins / long_total) * 100 if long_total > 0 else 0
    long_pnl = long_trades['pnl'].sum()
    long_return_pct = (long_pnl / initial_capital) * 100 if initial_capital > 0 else 0

    short_trades = trades_df[trades_df['type'] == 'short']
    short_total = len(short_trades)
    short_wins = (short_trades['result'] == 'win').sum()
    short_win_rate = (short_wins / short_total) * 100 if short_total > 0 else 0
    short_pnl = short_trades['pnl'].sum()
    short_return_pct = (short_pnl / initial_capital) * 100 if initial_capital > 0 else 0

    return {
        "Capital Final": final_capital,
        "Retorno %": total_return_pct,
        "Total de Trades": total_trades,
        "Taxa de Acerto %": total_win_rate,
        "Long Trades": long_total,
        "Long Win Rate %": long_win_rate,
        "Long Return %": long_return_pct,
        "Short Trades": short_total,
        "Short Win Rate %": short_win_rate,
        "Short Return %": short_return_pct,
    }


def run_full_backtest(symbol, leverage):
    """Orquestra a execução do backtest, usando uma lista de exchanges como fallback."""
    print(f"\nIniciando backtest otimizado para {symbol}...")

    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        exchange_priority_list = config.get('data_source_exchanges', ['binance'])
        print(f"Usando exchanges para dados do backtest (em ordem): {exchange_priority_list}")
    except Exception as e:
        print(f"AVISO: Não foi possível ler config.json, usando ['binance'] como padrão. Erro: {e}")
        exchange_priority_list = ['binance']

    timeframe_map = {'m': 60, 'h': 3600, 'd': 86400}
    required_timeframes = set(details['timeframe'] for details in STRATEGIES.values())
    data_with_indicators = {}

    for tf in sorted(list(required_timeframes)):
        try:
            if tf == '1m':
                print(f"AVISO: Timeframe de '{tf}' será pulado conforme configuração.")
                continue
            tf_value = int(tf[:-1])
            tf_unit = tf[-1]
            tf_seconds = tf_value * timeframe_map[tf_unit]
            candles_in_30_days = (30 * 86400) // tf_seconds
            max_warmup_candles = max(s.get('min_candles', 0) for s in STRATEGIES.values() if s['timeframe'] == tf)
            total_candles_to_fetch = candles_in_30_days + max_warmup_candles

            now_ms = int(datetime.now().timestamp() * 1000)
            start_timestamp_ms = now_ms - (total_candles_to_fetch * tf_seconds * 1000)
            price_data = None
            for exchange_name in exchange_priority_list:
                data = get_exchange_data(exchange_name, symbol, tf, start_timestamp_ms)
                if data is not None and not data.empty:
                    price_data = data
                    print(f"Dados para {tf} obtidos com sucesso de '{exchange_name}'.")
                    break

            if price_data is not None and not price_data.empty:
                data_with_indicators[tf] = add_all_indicators(price_data.copy())
            else:
                print(
                    f"ERRO FINAL: Não foi possível obter dados suficientes para o timeframe {tf} em nenhuma exchange.")

        except Exception as e:
            print(f"Erro ao processar o timeframe {tf}: {e}")

    if not data_with_indicators: return None

    best_results_per_strategy = {}

    for name, logic in STRATEGIES.items():
        current_tf = logic['timeframe']
        if current_tf not in data_with_indicators: continue

        df_for_backtest = data_with_indicators[current_tf]

        best_strategy_run = None
        best_strategy_return = -float('inf')

        for sl_pct in STOP_LOSS_LEVELS_PCT:
            for rrr in TAKE_PROFIT_RRRS:
                final_capital, trades_df = run_backtest(df_for_backtest.copy(), INITIAL_CAPITAL, rrr, sl_pct, leverage,
                                                        logic, RISK_PER_TRADE_PCT)
                perf = analyze_performance(INITIAL_CAPITAL, final_capital, trades_df)

                if perf["Retorno %"] > best_strategy_return:
                    best_strategy_return = perf["Retorno %"]
                    best_strategy_run = {"Estratégia": name, "Stop Loss %": f"{sl_pct * 100:.1f}%",
                                         "Take Profit RRR": f"{rrr}:1", **perf}

        if best_strategy_run:
            best_results_per_strategy[name] = best_strategy_run

    if not best_results_per_strategy: return []

    final_results = sorted(best_results_per_strategy.values(), key=lambda x: x['Retorno %'], reverse=True)
    return final_results
# bot_logic.py
import pandas as pd
import time
import json
from threading import Event
from decimal import Decimal, ROUND_HALF_UP
import logging  # <-- INÍCIO DA MODIFICAÇÃO

from exchange_factory import get_client
from strategies import STRATEGIES, add_all_indicators, get_btc_trend_filter
from database import insert_successful_order
from data_fetcher import get_historical_klines


def calculate_order_amount(capital, entry_price, sl_price, risk_percentage):
    """Calcula a quantidade (amount) da ordem com base no risco definido."""
    capital_at_risk = capital * risk_percentage
    price_risk_per_unit = abs(entry_price - sl_price)
    if price_risk_per_unit == 0: return 0.0
    return capital_at_risk / price_risk_per_unit


def round_to_increment(value: float, increment: float) -> str:
    """Arredonda um valor para o múltiplo mais próximo de um incremento (tick/lot size)."""
    if increment <= 0: return f"{value:.8f}"
    value_dec = Decimal(str(value))
    increment_dec = Decimal(str(increment))
    rounded_value = (value_dec / increment_dec).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * increment_dec
    return "{:f}".format(rounded_value.normalize())


class TradingBot:
    # --- INÍCIO DA MODIFICAÇÃO (__init__) ---
    def __init__(self, account_config: dict, global_config: dict, exchange_name: str):
        self.account_config = account_config
        self.account_name = account_config.get('account_name', 'Desconhecida')
        self.exchange_name = exchange_name
        self.global_config = global_config
        # self.logs = logs_list # <-- Removido

        # Obtém um logger específico para esta conta
        self.logger = logging.getLogger(self.account_name)

        self.client = get_client(exchange_name, account_config)
        # --- FIM DA MODIFICAÇÃO ---

        self.stop_event = Event()
        self.STRATEGIES = STRATEGIES
        self.market_info_cache = {}

    # --- INÍCIO DA MODIFICAÇÃO (método log) ---
    def log(self, message, level='INFO'):
        """Envia uma mensagem de log para o sistema de logging central."""
        level = level.upper()

        if level == 'SUCCESS':
            self.logger.log(logging.SUCCESS, message)
        elif level == 'WARNING':
            self.logger.warning(message)
        elif level == 'ERROR':
            self.logger.error(message)
        elif level == 'DEBUG':
            self.logger.debug(message)
        else:  # Trata INFO e EXECUTION como INFO
            self.logger.info(message)

    # --- FIM DA MODIFICAÇÃO ---

    def _apply_initial_settings(self):
        """Sincroniza as configurações de forma inteligente, evitando chamadas desnecessárias."""
        self.log("Iniciando sincronização de configurações (Alavancagem e Margem)...")
        # ... (restante da função sem alterações)
        all_open_positions = self.client.get_open_positions()
        open_position_symbols = {pos['symbol'].upper() for pos in all_open_positions}
        try:
            account_settings = self.client.get_account_settings()
            settings_map = {s['symbol'].upper(): s for s in account_settings}
        except (AttributeError, NotImplementedError):
            self.log("A exchange não suporta get_account_settings. Pulando sincronização de margem.", level='WARNING')
            settings_map = {}
        if open_position_symbols:
            self.log(
                f"Posições abertas encontradas: {list(open_position_symbols)}. Configurações não serão alteradas para estes ativos.",
                level='WARNING')
        for setup in self.account_config.get('markets_to_trade', []):
            symbol = setup.get('base_currency').upper()
            if not symbol or symbol in open_position_symbols:
                continue
            desired_leverage = setup.get('leverage')
            desired_isolated = True
            current_settings = settings_map.get(symbol)
            if not current_settings:
                self.log(f"{symbol} está com configurações padrão. Atualizando...")
                if desired_leverage:
                    self.client.update_leverage(symbol, int(desired_leverage))
                self.client.update_margin_mode(symbol, is_isolated=desired_isolated)
            else:
                if desired_leverage and int(current_settings.get('leverage')) != int(desired_leverage):
                    self.log(
                        f"Alavancagem de {symbol} divergente ({current_settings.get('leverage')}x). Atualizando para {desired_leverage}x...")
                    self.client.update_leverage(symbol, int(desired_leverage))
                if current_settings.get('isolated') != desired_isolated:
                    self.log(f"Modo de margem de {symbol} divergente. Atualizando para ISOLATED...")
                    self.client.update_margin_mode(symbol, is_isolated=desired_isolated)
        self.log("Sincronização de configurações concluída.")

    def _manage_trailing_stop(self, setup: dict, position: dict, all_open_orders: list, market_data: pd.DataFrame):
        api_symbol = position['symbol'].upper()
        position_side = "long" if position.get('side') == 'bid' else "short"
        entry_price = float(position.get('entry_price'))
        last_candle = market_data.iloc[-2]
        current_sl_order = next((o for o in all_open_orders if
                                 o.get('symbol', '').upper() == api_symbol and o.get('order_type', '').startswith(
                                     'stop')), None)
        if not self.market_info_cache.get(api_symbol):
            self.market_info_cache[api_symbol] = self.client.get_market_info(api_symbol)
        tick_size = float(self.market_info_cache[api_symbol].get('tick_size', 0.01))

        if not current_sl_order:
            self.log(f"Posição em {api_symbol} sem SL. Criando SL inicial com base no setup.", level='WARNING')
            sl_config = setup['stop_loss_config']
            sl_distance = entry_price * sl_config['value'] if sl_config['type'] == 'percentage' else last_candle[
                                                                                                         'ATR'] * sl_config.get(
                'value', 1.0)
            initial_sl_price = entry_price - sl_distance if position_side == 'long' else entry_price + sl_distance
            sl_rounded_str = round_to_increment(initial_sl_price, tick_size)
            result = self.client.set_position_tpsl(
                symbol=api_symbol,
                position_side=position.get('side'),
                position_size=str(position.get('amount')),
                tick_size=tick_size,
                stop_loss_price=sl_rounded_str
            )
            self.log(f"Resultado da criação do SL inicial: {result}", level='EXECUTION')
            if not (result and result.get('success')):
                self.log(f"Falha ao criar SL inicial para {api_symbol}.", level='ERROR')
            return

        tsl_config = setup.get('trailing_stop_config')
        if not tsl_config or tsl_config.get('type') == 'none':
            return False

        current_sl_price = float(current_sl_order['stop_price'])
        potential_new_sl = None

        if tsl_config['type'] == 'breakeven':
            if (position_side == 'long' and current_sl_price >= entry_price) or \
                    (position_side == 'short' and current_sl_price <= entry_price):
                return
            trigger_rrr = tsl_config.get('breakeven_trigger_rrr', 1.0)
            sl_config = setup['stop_loss_config']
            initial_sl_distance = entry_price * sl_config['value'] if sl_config['type'] == 'percentage' else \
                last_candle['ATR'] * sl_config['value']
            breakeven_buffer = 0.15 * last_candle['ATR']
            if position_side == 'long':
                breakeven_trigger_price = entry_price + (initial_sl_distance * trigger_rrr)
                if last_candle['high'] >= breakeven_trigger_price:
                    potential_new_sl = entry_price + breakeven_buffer
            else:
                breakeven_trigger_price = entry_price - (initial_sl_distance * trigger_rrr)
                if last_candle['low'] <= breakeven_trigger_price:
                    potential_new_sl = entry_price - breakeven_buffer
            if potential_new_sl:
                self.log(f"Breakeven ATINGIDO para {api_symbol}! Potencial novo SL: ${potential_new_sl:.4f}")

        elif tsl_config['type'] == 'atr':
            atr_value = last_candle['ATR']
            current_price = last_candle['close']
            atr_multiple = setup.get('tsl_atr_multiple', 2.0)
            atr_pct = atr_value / current_price
            if atr_pct > 0.015:
                atr_multiple *= 1.25
            elif atr_pct < 0.005:
                atr_multiple *= 0.9
            min_distance = max(atr_value * 2, current_price * 0.002)
            safety_buffer = 2.5 * atr_value
            high_water_mark = market_data['high'].max()
            low_water_mark = market_data['low'].min()
            if position_side == 'long':
                potential_new_sl = high_water_mark - (atr_multiple * atr_value)
                potential_new_sl = min(potential_new_sl, current_price - safety_buffer)
                if current_price - potential_new_sl < min_distance: potential_new_sl = current_price - min_distance
            else:
                potential_new_sl = low_water_mark + (atr_multiple * atr_value)
                potential_new_sl = max(potential_new_sl, current_price + safety_buffer)
                if potential_new_sl - current_price < min_distance: potential_new_sl = current_price + min_distance

        if potential_new_sl is None:
            return False

        if tsl_config.get('type') == 'atr' and ((position_side == 'long' and potential_new_sl < entry_price) or (
                position_side == 'short' and potential_new_sl > entry_price)):
            return False

        is_improvement = (position_side == 'long' and potential_new_sl > current_sl_price) or \
                         (position_side == 'short' and potential_new_sl < current_sl_price)

        if is_improvement:
            update_buffer = 1.0 * last_candle['ATR']
            if abs(potential_new_sl - current_sl_price) < update_buffer:
                return False

            self.log(
                f"TRAILING STOP (UPDATE) para {api_symbol}: Movendo de ${current_sl_price:.4f} para ${potential_new_sl:.4f}")

            new_sl_rounded_str = round_to_increment(potential_new_sl, tick_size)
            cancel_sl_result = self.client.cancel_stop_order(api_symbol, current_sl_order['order_id'])
            self.log(f"Resultado do cancelamento do SL antigo: {cancel_sl_result}", level='EXECUTION')

            if not (cancel_sl_result and cancel_sl_result.get('success')):
                self.log(f"FALHA ao cancelar SL antigo para {api_symbol}. Abortando TSL.", level='ERROR')
                return False

            if tsl_config.get('remove_tp_on_trail', False):
                tp_order = next((o for o in all_open_orders if
                                 o.get('symbol', '').upper() == api_symbol and o.get('order_type').startswith(
                                     'take_profit')), None)
                if tp_order:
                    cancel_tp_result = self.client.cancel_stop_order(api_symbol, tp_order['order_id'])
                    self.log(f"Resultado da remoção do TP: {cancel_tp_result}", level='EXECUTION')

            result = self.client.set_position_tpsl(
                symbol=api_symbol,
                position_side=position.get('side'),
                position_size=str(position.get('amount')),
                tick_size=tick_size,
                stop_loss_price=new_sl_rounded_str
            )
            self.log(f"Resultado da criação do novo SL: {result}", level='EXECUTION')
            if result and result.get('success'):
                return True

        return False

    def run(self):
        self.log(f"Bot para a conta '{self.account_name}' iniciado.", level='SUCCESS')
        num_setups = len(self.account_config.get('markets_to_trade', []))
        self.log(f"Monitorando {num_setups} setups de trading.")

        check_interval = self.account_config.get('check_interval_seconds', 180)
        data_source_exchanges = self.global_config.get('data_source_exchanges', ['binance'])
        btc_trend_direction = "long_short"

        while not self.stop_event.is_set():
            loop_start_time = time.time()
            try:
                if any(s.get('direction_mode') == 'btc_trend' for s in self.account_config.get('markets_to_trade', [])):
                    btc_trend_direction = get_btc_trend_filter(data_source_exchanges)
                    if btc_trend_direction == "long_only":
                        self.log("Filtro de Tendência BTC: ATIVO (Apenas Long).")
                    elif btc_trend_direction == "short_only":
                        self.log("Filtro de Tendência BTC: ATIVO (Apenas Short).")

                account_info = self.client.get_account_info()
                if not account_info:
                    self.log("Não foi possível obter informações da conta no loop, pulando ciclo.", level='ERROR')
                    time.sleep(check_interval)
                    continue
                balance = float(account_info.get('balance', 0.0))

                all_open_orders = self.client.get_open_orders()
                for order in all_open_orders:
                    if order.get('order_type', '').lower() == 'limit' and not order.get('reduce_only', False):
                        self.log(
                            f"Cancelando ordem limite de ENTRADA {order.get('order_id')} para {order.get('symbol')}")
                        cancel_result = self.client.cancel_order(order.get('symbol'), order.get('order_id'))
                        self.log(f"Resultado do cancelamento: {cancel_result}", level='EXECUTION')

                all_open_positions = self.client.get_open_positions()
                open_positions_map = {pos['symbol'].upper(): pos for pos in all_open_positions}

                if open_positions_map:
                    self.log(f"Posicoes abertas encontradas para: {list(open_positions_map.keys())}")

                assets_without_signal = []

                for setup in self.account_config.get('markets_to_trade', []):
                    if self.stop_event.is_set(): break
                    api_symbol = setup["base_currency"].upper()

                    if api_symbol in open_positions_map:
                        position = open_positions_map[api_symbol]
                        ccxt_symbol = f"{setup['base_currency']}/{setup['quote_currency']}"
                        strategy_name = setup["strategy_name"]
                        strategy_logic = self.STRATEGIES[strategy_name]
                        timeframe = strategy_logic.get('timeframe', '15m')
                        min_candles = strategy_logic.get('min_candles', 200)

                        market_data = None
                        for exchange_name in data_source_exchanges:
                            data = get_historical_klines(ccxt_symbol, timeframe, limit=min_candles,
                                                         exchange_name=exchange_name)
                            if data is not None and not data.empty:
                                market_data = data
                                break
                        if market_data is None or market_data.empty:
                            self.log(f"Nao foi possivel obter dados para monitorar {api_symbol}.", level='ERROR')
                            continue

                        df = add_all_indicators(market_data)

                        if self._manage_trailing_stop(setup, position, all_open_orders, df):
                            self.log(f"Ressincronizando estado das ordens para {api_symbol} após TSL.")
                            all_open_orders = self.client.get_open_orders()

                        last_candle = df.iloc[-2]
                        entry_price = float(position.get('entry_price'))
                        position_side = "long" if position.get('side') == 'bid' else "short"
                        sl_config = setup['stop_loss_config']
                        sl_distance = entry_price * sl_config['value'] if sl_config['type'] == 'percentage' else \
                            last_candle['ATR'] * sl_config['value']
                        initial_stop_loss_price = entry_price - sl_distance if position_side == 'long' else entry_price + sl_distance
                        emergency_exit = False
                        closing_side = None

                        if position_side == 'long' and last_candle['low'] < initial_stop_loss_price:
                            self.log(
                                f"SAÍDA DE EMERGÊNCIA: Preço ({last_candle['low']}) ultrapassou o SL inicial da estratégia ({initial_stop_loss_price:.4f}) para {api_symbol}. Fechando a mercado.",
                                level='WARNING')
                            emergency_exit = True
                            closing_side = "SELL"
                        elif position_side == 'short' and last_candle['high'] > initial_stop_loss_price:
                            self.log(
                                f"SAÍDA DE EMERGÊNCIA: Preço ({last_candle['high']}) ultrapassou o SL inicial da estratégia ({initial_stop_loss_price:.4f}) para {api_symbol}. Fechando a mercado.",
                                level='WARNING')
                            emergency_exit = True
                            closing_side = "BUY"

                        if emergency_exit:
                            if not self.market_info_cache.get(api_symbol):
                                self.market_info_cache[api_symbol] = self.client.get_market_info(api_symbol)
                            tick_size = float(self.market_info_cache[api_symbol].get('tick_size', 0.01))
                            result = self.client.create_market_order(symbol=api_symbol, side=closing_side,
                                                                     amount=str(position['amount']), reduce_only=True,
                                                                     tick_size=tick_size)
                            self.log(f"Resultado da ordem de fechamento de emergência: {result}", level='EXECUTION')
                            continue

                        exit_mode = setup.get('exit_mode', 'passivo')
                        if exit_mode == 'ativo' and strategy_logic.get('exit'):
                            position_side = "long" if position.get('side') == 'bid' else "short"
                            if strategy_logic['exit'](df.iloc[-2], df.iloc[-3], position_side):
                                self.log(
                                    f"SINAL DE SAÍDA ESTRATÉGICA DETECTADO para {api_symbol}! Enviando ordem limite de fechamento.",
                                    level='WARNING')
                                closing_side = "SELL" if position_side == 'long' else "BUY"
                                if not self.market_info_cache.get(api_symbol):
                                    self.market_info_cache[api_symbol] = self.client.get_market_info(api_symbol)
                                market_info = self.market_info_cache[api_symbol]
                                tick_size = float(market_info.get('tick_size', 0.01))
                                orderbook = self.client.get_orderbook(api_symbol)
                                if not orderbook:
                                    self.log(f"Não foi possível obter o order book para fechar {api_symbol}. Pulando.",
                                             level='ERROR')
                                    continue
                                limit_price = 0
                                price_buffer = 5 * tick_size
                                if closing_side == "SELL":
                                    best_bid_price = float(orderbook['l'][0][0]['p'])
                                    limit_price = best_bid_price + price_buffer
                                else:
                                    best_ask_price = float(orderbook['l'][1][0]['p'])
                                    limit_price = best_ask_price - price_buffer
                                limit_price_str = round_to_increment(limit_price, tick_size)
                                result = self.client.create_limit_order(
                                    symbol=api_symbol,
                                    side=closing_side,
                                    amount=str(position['amount']),
                                    price=limit_price_str,
                                    reduce_only=True,
                                    tick_size=tick_size
                                )
                                self.log(f"Resultado do envio da ordem de fechamento: {result}", level='EXECUTION')
                                continue
                        continue

                    ccxt_symbol = f"{setup['base_currency']}/{setup['quote_currency']}"
                    strategy_name = setup["strategy_name"]
                    if strategy_name not in self.STRATEGIES:
                        self.log(f"Estrategia '{strategy_name}' para {api_symbol} nao encontrada. Pulando.",
                                 level='WARNING')
                        continue

                    strategy_logic = self.STRATEGIES[strategy_name]
                    timeframe = strategy_logic.get('timeframe', '15m')
                    min_candles = strategy_logic.get('min_candles', 200)
                    market_data = None

                    for exchange_name in data_source_exchanges:
                        data = get_historical_klines(ccxt_symbol, timeframe, limit=min_candles,
                                                     exchange_name=exchange_name)
                        if data is not None and not data.empty:
                            market_data = data
                            break
                    if market_data is None or market_data.empty:
                        self.log(f"Nao foi possivel obter dados para {ccxt_symbol} em {data_source_exchanges}.",
                                 level='ERROR')
                        continue

                    df = add_all_indicators(market_data)
                    last_candle, prev_candle = df.iloc[-2], df.iloc[-3]
                    side = None
                    direction_mode = setup.get('direction_mode', 'long_short')
                    can_go_long = True
                    can_go_short = True

                    if direction_mode == 'btc_trend':
                        can_go_long = (btc_trend_direction == 'long_only')
                        can_go_short = (btc_trend_direction == 'short_only')
                    else:
                        can_go_long = (direction_mode in ['long_short', 'long_only'])
                        can_go_short = (direction_mode in ['long_short', 'short_only'])

                    if can_go_long and strategy_logic['long_entry'](last_candle, prev_candle):
                        side = "BUY"
                    elif can_go_short and strategy_logic['short_entry'](last_candle, prev_candle):
                        side = "SELL"

                    if side:
                        self.log(f"SINAL {side} DETECTADO para {api_symbol} (Estrategia: {strategy_name})",
                                 level='SUCCESS')
                        self.log(
                            f"Configurando alavancagem e modo de margem para {api_symbol} antes de abrir a ordem...")
                        try:
                            leverage_to_set = setup.get('leverage')
                            self.client.update_leverage(api_symbol, int(leverage_to_set))
                            self.client.update_margin_mode(api_symbol, is_isolated=True)
                            self.log(
                                f"Configurações para {api_symbol} aplicadas: Alavancagem {leverage_to_set}x, Modo ISOLATED.")
                        except Exception as e:
                            self.log(f"Falha ao configurar margem/alavancagem para {api_symbol}: {e}", level="ERROR")
                            continue

                        if not self.market_info_cache.get(api_symbol):
                            self.market_info_cache[api_symbol] = self.client.get_market_info(api_symbol)
                        market_info = self.market_info_cache[api_symbol]
                        tick_size = float(market_info.get('tick_size', 0.01))
                        lot_size = float(market_info.get('lot_size', 0.01))
                        orderbook = self.client.get_orderbook(api_symbol)

                        if not orderbook:
                            self.log(f"Não foi possível obter o order book para {api_symbol}. Pulando.", level='ERROR')
                            continue

                        if side == "BUY":
                            limit_price = float(orderbook['l'][0][0]['p']) - tick_size
                        else:
                            limit_price = float(orderbook['l'][1][0]['p']) + tick_size

                        limit_price_str = round_to_increment(limit_price, tick_size)
                        entry_price = float(limit_price_str)
                        sl_config = setup['stop_loss_config']
                        sl_distance = entry_price * sl_config['value'] if sl_config['type'] == 'percentage' else \
                            last_candle['ATR'] * sl_config['value']
                        rrr = setup['take_profit_rrr']
                        risk_percentage = setup['risk_per_trade']
                        stop_loss_price = entry_price - sl_distance if side == "BUY" else entry_price + sl_distance
                        take_profit_price = entry_price + (sl_distance * rrr) if side == "BUY" else entry_price - (
                                sl_distance * rrr)
                        amount_raw = calculate_order_amount(balance, entry_price, stop_loss_price, risk_percentage)
                        amount_rounded_str = round_to_increment(amount_raw, lot_size)
                        stop_loss_price_rounded_str = round_to_increment(stop_loss_price, tick_size)
                        take_profit_price_rounded_str = round_to_increment(take_profit_price, tick_size)

                        if float(amount_rounded_str) > 0:
                            log_details = (
                                f"ORDEM LIMITE PREPARADA -> Ativo: {api_symbol}, Lado: {side}, Preço: ${limit_price_str}, Qtd: {amount_rounded_str}, TP: ${take_profit_price_rounded_str}, SL: ${stop_loss_price_rounded_str}")
                            self.log(log_details)
                            payload_para_ordem = {
                                "symbol": api_symbol, "side": side, "amount": amount_rounded_str,
                                "price": limit_price_str, "take_profit_price": take_profit_price_rounded_str,
                                "stop_loss_price": stop_loss_price_rounded_str, "tick_size": tick_size
                            }
                            result = self.client.create_limit_order(**payload_para_ordem)
                            self.log(f"Resultado do envio da ordem: {result}", level='EXECUTION')

                            if result and result.get('success') and isinstance(result.get('data'), dict):
                                order_data_response = result['data']
                                order_id_value = order_data_response.get('orderId') or order_data_response.get(
                                    'order_id')

                                if order_id_value:
                                    order_record_to_db = {
                                        "order_id": order_id_value,
                                        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                                        "full_payload": payload_para_ordem,
                                        "account_name": self.account_name,
                                        "exchange": self.exchange_name,
                                        "ativo": api_symbol,
                                        "lado": side,
                                        "quantidade": amount_rounded_str,
                                        "strategy_name": setup.get('strategy_name'),
                                        "leverage": setup.get('leverage'),
                                        "risk_per_trade": setup.get('risk_per_trade'),
                                        "take_profit_rrr": setup.get('take_profit_rrr'),
                                        "direction_mode": setup.get('direction_mode'),
                                        "exit_mode": setup.get('exit_mode'),
                                        "stop_loss_config": setup.get('stop_loss_config'),
                                        "trailing_stop_config": setup.get('trailing_stop_config'),
                                        "preco_entrada_aprox": limit_price_str,
                                        "stop_loss_price": stop_loss_price_rounded_str,
                                        "take_profit_price": take_profit_price_rounded_str,
                                    }
                                    insert_successful_order(order_record_to_db)
                                    self.log(
                                        f"Ordem {order_id_value} salva no banco de dados de historico.")
                                else:
                                    self.log(f"Falha ao extrair ID da ordem da resposta: {order_data_response}",
                                             level='WARNING')
                            elif result and not result.get('success'):
                                self.log(f"Falha ao criar ordem: {result.get('error', 'Erro desconhecido')}",
                                         level='ERROR')
                        else:
                            self.log(f"Quantidade da ordem para {api_symbol} é zero.", level='WARNING')
                    else:
                        assets_without_signal.append(api_symbol)

                if assets_without_signal:
                    self.log(f"Nenhum sinal de entrada detectado para: {', '.join(assets_without_signal)}")

            except Exception as e:
                self.log(f"CRITICO no loop: {e}", level='ERROR')
                # Adiciona o traceback ao log
                self.logger.error("Traceback do erro:", exc_info=True)

            finally:
                elapsed_time = time.time() - loop_start_time
                sleep_duration = max(0, check_interval - elapsed_time)
                if self.stop_event.wait(timeout=sleep_duration):
                    break

        self.log("Bot foi parado.", level="WARNING")

    def stop(self):
        self.stop_event.set()
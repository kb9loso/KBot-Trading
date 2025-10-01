# bot_logic.py
import pandas as pd
import time
import json
from threading import Event
from decimal import Decimal, ROUND_HALF_UP

from pacifica_client import PacificaClient
from strategies import STRATEGIES, add_all_indicators


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


def log_order_to_history(order_data: dict):
    """Salva os detalhes de uma ordem bem-sucedida em um arquivo JSON."""
    history_file = "historico.json"
    try:
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []
        history.append(order_data)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"ERRO CRITICO: Falha ao salvar no arquivo de historico: {e}")


class TradingBot:
    def __init__(self, account_config: dict, global_config: dict, logs_list: list, exchange_name: str):
        self.account_config = account_config
        self.account_name = account_config.get('account_name', 'Desconhecida')
        self.exchange_name = exchange_name
        self.global_config = global_config
        self.logs = logs_list

        self.client = PacificaClient(
            main_public_key=account_config['main_public_key'],
            agent_private_key=account_config['agent_private_key']
        )

        self.stop_event = Event()
        self.STRATEGIES = STRATEGIES
        self.market_info_cache = {}

    def log(self, message, level='INFO'):
        """Adiciona uma mensagem de log estruturada com timestamp, nível e conta."""
        timestamp = pd.Timestamp.now(tz='America/Sao_Paulo').strftime('%Y-%m-%d %H:%M:%S')
        log_entry = {
            "timestamp": timestamp,
            "account": self.account_name,
            "level": level,
            "message": message
        }
        self.logs.append(log_entry)
        print(f"[{timestamp}] [{self.account_name}] [{level}] {message}")
        while len(self.logs) > 200:  # Aumentei um pouco o limite de logs
            self.logs.pop(0)

    def _apply_initial_settings(self):
        """Sincroniza as configurações de forma inteligente, evitando chamadas desnecessárias."""
        self.log("Iniciando sincronização de configurações (Alavancagem e Margem)...")

        # 1. Busca os estados atuais uma única vez
        all_open_positions = self.client.get_open_positions()
        open_position_symbols = {pos['symbol'].upper() for pos in all_open_positions}

        account_settings = self.client.get_account_settings()
        settings_map = {s['symbol'].upper(): s for s in account_settings}

        if open_position_symbols:
            self.log(
                f"Posições abertas encontradas: {list(open_position_symbols)}. Configurações não serão alteradas para estes ativos.",
                level='WARNING')

        for setup in self.account_config.get('markets_to_trade', []):
            symbol = setup.get('base_currency').upper()
            if not symbol or symbol in open_position_symbols:
                continue

            # 2. Define as configurações desejadas
            desired_leverage = setup.get('leverage')
            desired_isolated = True  # Sempre queremos margem isolada

            # 3. Compara com as configurações atuais da exchange
            current_settings = settings_map.get(symbol)

            if not current_settings:
                # Se não há settings, significa que está no padrão da exchange. Devemos atualizar.
                self.log(f"{symbol} está com configurações padrão. Atualizando...")
                if desired_leverage:
                    self.client.update_leverage(symbol, int(desired_leverage))
                self.client.update_margin_mode(symbol, is_isolated=desired_isolated)
            else:
                # Se já existe configuração, verifica se precisa de atualização
                if desired_leverage and int(current_settings.get('leverage')) != int(desired_leverage):
                    self.log(
                        f"Alavancagem de {symbol} divergente ({current_settings.get('leverage')}x). Atualizando para {desired_leverage}x...")
                    self.client.update_leverage(symbol, int(desired_leverage))

                if current_settings.get('isolated') != desired_isolated:
                    self.log(f"Modo de margem de {symbol} divergente. Atualizando para ISOLATED...")
                    self.client.update_margin_mode(symbol, is_isolated=desired_isolated)

        self.log("Sincronização de configurações concluída.")

    def _manage_trailing_stop(self, setup: dict, position: dict, all_open_orders: list, market_data: pd.DataFrame):
        """
        Gerencia o trailing stop para uma posição aberta de forma 100% stateless.
        Cria um SL se não existir, ou atualiza o SL existente se houver melhoria.
        """
        api_symbol = position['symbol'].upper()
        tsl_config = setup.get('trailing_stop_config')
        if not tsl_config:
            return

        self.log(f"Gerenciando Trailing Stop para {api_symbol} (Tipo: {tsl_config.get('type')})...")

        position_side = "long" if position.get('side') == 'bid' else "short"
        entry_price = float(position.get('entry_price'))
        last_candle = market_data.iloc[-1]

        # 1. Encontra a ordem de Stop Loss atual, se existir
        current_sl_order = next((o for o in all_open_orders if
                                 o.get('symbol', '').upper() == api_symbol and o.get('order_type', '').startswith(
                                     'stop_loss')), None)

        if not self.market_info_cache.get(api_symbol):
            self.market_info_cache[api_symbol] = self.client.get_market_info(api_symbol)
        tick_size = float(self.market_info_cache[api_symbol].get('tick_size', 0.01))

        # 2. Se não existir um SL, cria o SL inicial com base no setup
        if not current_sl_order:
            self.log(f"Posição em {api_symbol} sem SL. Criando SL inicial com base no setup.", level='WARNING')

            sl_config = setup['stop_loss_config']
            sl_distance = entry_price * sl_config['value'] if sl_config['type'] == 'percentage' else last_candle[
                                                                                                         'ATR'] * \
                                                                                                     sl_config['value']
            initial_sl_price = entry_price - sl_distance if position_side == 'long' else entry_price + sl_distance
            sl_rounded_str = round_to_increment(initial_sl_price, tick_size)

            result = self.client.set_position_tpsl(api_symbol, position.get('side'), tick_size,
                                                   stop_loss_price=sl_rounded_str)
            if result and result.get('success'):
                self.log(f"SL inicial para {api_symbol} criado com sucesso.")
            else:
                self.log(f"Falha ao criar SL inicial para {api_symbol}. Resposta: {result}", level='ERROR')
            return  # Encerra a função para este ativo no ciclo atual

        # 3. Se um SL já existe, calcula o potencial novo SL para o trailing
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

            breakeven_buffer = 0.1 * last_candle['ATR']

            if position_side == 'long':
                breakeven_trigger_price = entry_price + (initial_sl_distance * trigger_rrr)
                if last_candle['high'] >= breakeven_trigger_price:
                    potential_new_sl = entry_price + breakeven_buffer
            else:  # Short
                breakeven_trigger_price = entry_price - (initial_sl_distance * trigger_rrr)
                if last_candle['low'] <= breakeven_trigger_price:
                    potential_new_sl = entry_price - breakeven_buffer

            if potential_new_sl: self.log(
                f"Breakeven ATINGIDO para {api_symbol}! Potencial novo SL: ${potential_new_sl:.4f}")

        elif tsl_config['type'] == 'atr':
            high_water_mark = market_data['high'].max()
            low_water_mark = market_data['low'].min()
            atr_multiple = tsl_config.get('atr_multiple', 2.0)
            safety_buffer = 2 * last_candle['ATR']
            if position_side == 'long':
                potential_new_sl = high_water_mark - (atr_multiple * last_candle['ATR'])
                potential_new_sl = min(potential_new_sl, last_candle['close'] - safety_buffer)

            else:  # short
                potential_new_sl = low_water_mark + (atr_multiple * last_candle['ATR'])
                potential_new_sl = max(potential_new_sl, last_candle['close'] + safety_buffer)

        if potential_new_sl is None:
            return False

        if tsl_config.get('type') == 'atr':
            if (position_side == 'long' and potential_new_sl < entry_price) or \
                    (position_side == 'short' and potential_new_sl > entry_price):
                return False

        is_improvement = (position_side == 'long' and potential_new_sl > current_sl_price) or \
                         (position_side == 'short' and potential_new_sl < current_sl_price)

        if is_improvement:
            update_buffer = 0.5 * last_candle['ATR']
            if abs(potential_new_sl - current_sl_price) < update_buffer:
                return False
            self.log(
                f"TRAILING STOP (UPDATE) para {api_symbol}: Movendo de ${current_sl_price:.4f} para ${potential_new_sl:.4f}")

            new_sl_rounded_str = round_to_increment(potential_new_sl, tick_size)
            cancel_sl_result = self.client.cancel_stop_order(api_symbol, current_sl_order['order_id'])
            if not (cancel_sl_result and cancel_sl_result.get('success')):
                self.log(f"FALHA ao cancelar SL antigo para {api_symbol}. Abortando TSL.", level='ERROR')
                return False

            self.log(f"SL antigo ({current_sl_order['order_id']}) cancelado com sucesso.")

            if tsl_config.get('remove_tp_on_trail', False):
                tp_order = next((o for o in all_open_orders if
                                 o.get('symbol', '').upper() == api_symbol and o.get('order_type').startswith(
                                     'take_profit')), None)
                if tp_order:
                    self.log(f"Removendo ordem Take Profit ({tp_order['order_id']})...")
                    self.client.cancel_stop_order(api_symbol, tp_order['order_id'])

            result = self.client.set_position_tpsl(
                symbol=api_symbol,
                side=position.get('side'),
                tick_size=tick_size,
                stop_loss_price=new_sl_rounded_str
            )

            if result and result.get('success'):
                self.log(f"Novo SL para {api_symbol} criado com sucesso.")
                return True
            else:
                self.log(f"FALHA ao criar novo SL para {api_symbol}. Resposta: {result}", level='ERROR')

        return False

    def run(self):
        """Loop principal do bot."""
        self.log(f"Bot para a conta '{self.account_name}' iniciado.")
        num_setups = len(self.account_config.get('markets_to_trade', []))
        self.log(f"Monitorando {num_setups} setups de trading.")

        initial_account_info = self.client.get_account_info()
        if not initial_account_info:
            self.log("Nao foi possivel obter informacoes da conta na inicializacao.", level='ERROR')
            return

        balance = float(initial_account_info.get('balance', 0.0))
        check_interval = self.account_config.get('check_interval_seconds', 180)
        data_source_exchanges = self.global_config.get('data_source_exchanges', ['binance'])

        while not self.stop_event.is_set():
            loop_start_time = time.time()
            try:
                all_open_positions = self.client.get_open_positions()
                open_positions_map = {pos['symbol'].upper(): pos for pos in all_open_positions}
                all_open_orders = self.client.get_open_orders()

                if open_positions_map:
                    self.log(f"Posicoes abertas encontradas para: {list(open_positions_map.keys())}")

                assets_without_signal = []

                for setup in self.account_config.get('markets_to_trade', []):
                    if self.stop_event.is_set(): break
                    api_symbol = setup["base_currency"].upper()

                    if api_symbol in open_positions_map:
                        exit_mode = setup.get('exit_mode', 'passivo')
                        if exit_mode == 'ativo':
                            position = open_positions_map[api_symbol]
                            ccxt_symbol = f"{setup['base_currency']}/{setup['quote_currency']}"
                            strategy_name = setup["strategy_name"]
                            strategy_logic = self.STRATEGIES[strategy_name]
                            timeframe = strategy_logic.get('timeframe', '15m')
                            min_candles = strategy_logic.get('min_candles', 200)

                            market_data = None
                            for exchange_name in data_source_exchanges:
                                data = self.client.get_historical_klines(ccxt_symbol, timeframe, limit=min_candles,
                                                                         exchange_name=exchange_name)
                                if data is not None and not data.empty:
                                    market_data = data
                                    break

                            if market_data is None or market_data.empty:
                                self.log(f"Nao foi possivel obter dados para monitorar {api_symbol}.", level='ERROR')
                                continue

                            df = add_all_indicators(market_data)
                            orders_were_updated = self._manage_trailing_stop(setup, position, all_open_orders, df)

                            if orders_were_updated:
                                self.log(
                                    f"Ressincronizando estado das ordens para {api_symbol} após atualização do TSL.")
                                all_open_orders = self.client.get_open_orders()

                            position_side = "long" if position.get('side') == 'bid' else "short"
                            entry_price = float(position.get('entry_price'))
                            current_sl_order = next((o for o in all_open_orders if
                                                     o.get('symbol', '').upper() == api_symbol and o.get('order_type',
                                                                                                         '').startswith(
                                                         'stop_loss')), None)
                            is_risk_free = False
                            if current_sl_order:
                                current_sl_price = float(current_sl_order['stop_price'])
                                if (position_side == 'long' and current_sl_price >= entry_price) or \
                                        (position_side == 'short' and current_sl_price <= entry_price):
                                    is_risk_free = True

                            if not is_risk_free:
                                last_candle, prev_candle = df.iloc[-2], df.iloc[-3]
                                if strategy_logic.get('exit') and strategy_logic['exit'](last_candle, prev_candle,
                                                                                         position_side):
                                    self.log(f"SINAL DE SAÍDA ESTRATÉGICA DETECTADO para {api_symbol}!",
                                             level='WARNING')
                                    closing_side = "SELL" if position_side == 'long' else "BUY"
                                    result = self.client.close_market_order(api_symbol, closing_side,
                                                                            position['amount'])
                                    self.log(f"Resultado do envio da ordem de fechamento: {result}")
                        continue

                    ccxt_symbol = f"{setup['base_currency']}/{setup['quote_currency']}"
                    strategy_name = setup["strategy_name"]

                    if strategy_name not in self.STRATEGIES:
                        self.log(
                            f"Estrategia '{strategy_name}' para o ativo {api_symbol} nao encontrada. Pulando.",
                            level='WARNING')
                        continue

                    strategy_logic = self.STRATEGIES[strategy_name]
                    timeframe = strategy_logic.get('timeframe', '15m')
                    min_candles = strategy_logic.get('min_candles', 200)

                    market_data = None
                    for exchange_name in data_source_exchanges:
                        data = self.client.get_historical_klines(ccxt_symbol, timeframe, limit=min_candles,
                                                                 exchange_name=exchange_name)
                        if data is not None and not data.empty:
                            market_data = data
                            break

                    if market_data is None or market_data.empty:
                        self.log(
                            f"Nao foi possivel obter dados para {ccxt_symbol} em nenhuma das exchanges: {data_source_exchanges}.",
                            level='ERROR')
                        continue

                    df = add_all_indicators(market_data)
                    last_candle, prev_candle = df.iloc[-2], df.iloc[-3]
                    side = None
                    direction_mode = setup.get('direction_mode', 'long_short')
                    if direction_mode in ['long_short', 'long_only'] and strategy_logic['long_entry'](last_candle,
                                                                                                      prev_candle):
                        side = "BUY"
                    elif direction_mode in ['long_short', 'short_only'] and strategy_logic['short_entry'](last_candle,
                                                                                                          prev_candle):
                        side = "SELL"

                    if side:
                        self.log(f"SINAL {side} DETECTADO para {api_symbol} (Estrategia: {strategy_name})")
                        if not self.market_info_cache.get(api_symbol): self.market_info_cache[
                            api_symbol] = self.client.get_market_info(api_symbol)
                        market_info = self.market_info_cache[api_symbol]
                        tick_size = float(market_info.get('tick_size', 0.01))
                        lot_size = float(market_info.get('lot_size', 0.01))
                        entry_price = last_candle['close']
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
                                f"ORDEM PREPARADA -> Ativo: {api_symbol}, Lado: {side}, Qtd: {amount_rounded_str}, TP: ${take_profit_price_rounded_str}, SL: ${stop_loss_price_rounded_str}")
                            self.log(log_details)
                            result = self.client.create_market_order(
                                symbol=api_symbol, side=side, amount=amount_rounded_str,
                                slippage=1.0,
                                take_profit_price=take_profit_price_rounded_str,
                                stop_loss_price=stop_loss_price_rounded_str,
                                tick_size=tick_size
                            )
                            self.log(f"Resultado do envio da ordem: {result}")
                            if result and result.get('data') and result['data'].get('order_id'):
                                order_data = result['data']
                                order_record = {"timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                                                "order_id": order_data.get('order_id'), "ativo": api_symbol,
                                                "lado": side, "quantidade": amount_rounded_str,
                                                "estrategia": strategy_name,
                                                "preco_entrada_aprox": f"{entry_price:.4f}",
                                                "stop_loss": stop_loss_price_rounded_str,
                                                "take_profit": take_profit_price_rounded_str,
                                                "risco_percentual": f"{risk_percentage * 100}%", "rrr": f"{rrr}:1"}
                                log_order_to_history(order_record)
                                self.log(f"Ordem {order_data.get('order_id')} salva no historico.json")
                        else:
                            self.log(f"Quantidade da ordem para {api_symbol} é zero.", level='WARNING')
                    else:
                        assets_without_signal.append(api_symbol)

                if assets_without_signal:
                    self.log(f"Nenhum sinal de entrada detectado para: {', '.join(assets_without_signal)}")

            except Exception as e:
                self.log(f"CRITICO no loop: {e}", level='ERROR')

            finally:
                elapsed_time = time.time() - loop_start_time
                sleep_duration = max(0, check_interval - elapsed_time)
                if self.stop_event.wait(timeout=sleep_duration):
                    break

        self.log("Bot foi parado.")

    def stop(self):
        self.stop_event.set()
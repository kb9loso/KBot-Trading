import threading

import pandas as pd
import time
import json
from threading import Event, Lock
from decimal import Decimal, ROUND_HALF_UP
import logging
from collections import defaultdict
import ast  # Necessário para parsear o tsl_value (string de dicionário) caso venha do banco de dados

from exchange_factory import get_client
from strategies import STRATEGIES, add_all_indicators, get_btc_trend_filter
from database import insert_successful_order
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from websocket_manager import WebSocketManager  # Para type hinting


def calculate_order_amount(capital, entry_price, sl_price, risk_percentage):
    """Calcula a quantidade (amount) da ordem com base no risco definido."""
    # (Código Original Mantido)
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
    def __init__(self, account_config: dict, global_config: dict, exchange_name: str, ws_manager: 'WebSocketManager'):
        self.account_config = account_config
        self.account_name = account_config.get('account_name', 'Desconhecida')
        self.exchange_name = exchange_name
        self.global_config = global_config
        self.ws_manager = ws_manager  # Armazena a instância do WS Manager
        self.logger = logging.getLogger(self.account_name)
        self.client = get_client(exchange_name, account_config)
        self._last_processed_timestamp = {}
        self.stop_event = Event()
        self.STRATEGIES = STRATEGIES
        self.market_info_cache = {}

        # NOVO: Cache para rastrear a idade da posição (aproximação para delay_bars)
        self._position_age_cache = {}  # {symbol: hold_bars_count}

        # Mapeia (symbol, timeframe) -> [lista de setups]
        self.setups_by_key = defaultdict(list)
        # Lock para garantir que on_new_candle não execute concorrentemente
        self.processing_lock = Lock()

        # *** ADICIONADO PARA OTIMIZAÇÃO DO BTC TREND ***
        self.btc_trend_direction = "long_short"  # Valor padrão
        self.btc_trend_lock = Lock()  # Lock para atualizar a tendência
        self.uses_btc_trend = False  # Flag para saber se precisa se registrar
        self._position_timer = None
        self._cancel_timer = None

        # *** ADICIONADO PARA CACHE DE CHAMADAS API DE CONTA (OTIMIZAÇÃO) ***
        # Armazena dados de conta (account_info, orders, positions) e timestamp da última busca bem-sucedida.
        self._api_cache = {'timestamp': 0, 'account_info': {}, 'all_open_orders': [], 'all_open_positions': []}

    def _cancel_open_limit_orders_periodically(self):
        if self.stop_event.is_set():
            return

        # *** CONFIGURAÇÃO DE SEGURANÇA ***
        # Define o tempo mínimo que uma ordem deve ter para ser considerada para cancelamento.
        MIN_ORDER_AGE_SECONDS = 180.0
        MIN_ORDER_AGE_MS = MIN_ORDER_AGE_SECONDS * 1000
        current_time_ms = int(time.time() * 1000)

        self.log(
            f"Verificando e cancelando ordens limite de ENTRADA abertas (> {MIN_ORDER_AGE_SECONDS / 60:.0f} min)...",
            level='INFO')
        try:
            all_open_orders = self.client.get_open_orders()
            orders_canceled_count = 0

            for order in all_open_orders:
                if order.get('order_type', '').lower() != 'limit':
                    continue
                order_time_ms = order.get('created_at')

                # *** FILTRO 2: Verifica se possui timestamp válido ***
                if not order_time_ms:
                    self.log(f"Ordem limite {order.get('order_id')} de {order.get('symbol')} sem timestamp. Ignorada.",
                             level='WARNING')
                    continue

                order_age_ms = current_time_ms - order_time_ms

                # Verifica:
                # 1. NÃO é 'reduce_only' (i.e., é uma ordem de entrada)?
                # 2. É mais antiga que o limite de segurança?
                is_stale_limit_entry_order = (
                        not order.get('reduce_only', False) and
                        order_age_ms >= MIN_ORDER_AGE_MS
                )

                if is_stale_limit_entry_order:
                    self.log(
                        f"Cancelando ordem limite de ENTRADA {order.get('order_id')} para {order.get('symbol')} (Idade: {order_age_ms / 1000:.0f}s)")

                    cancel_result = self.client.cancel_order(order.get('symbol'), order.get('order_id'))
                    self.log(f"Resultado do cancelamento: {cancel_result}", level='EXECUTE')
                    orders_canceled_count += 1
                elif order_age_ms < MIN_ORDER_AGE_SECONDS:
                    self.log(
                        f"Ordem limite {order.get('order_id')} de {order.get('symbol')} ignorada. Idade: {order_age_ms / 1000:.0f}s (Muito nova).",
                        level='DEBUG')

            self.log(
                f"Cancelamento periódico concluído. Total de ordens de entrada canceladas: {orders_canceled_count}.",
                level='INFO')

        except Exception as e:
            self.log(f"Erro no cancelamento periódico de ordens: {e}", level='ERROR')

        # Agenda a próxima execução (a cada 5 minutos - 300.0 segundos)
        if not self.stop_event.is_set():
            self._cancel_timer = threading.Timer(300, self._cancel_open_limit_orders_periodically)
            self._cancel_timer.start()

    def _log_open_positions_periodically(self):
        """Busca e loga as posições abertas no formato solicitado (Posições abertas: XXX, YYY)."""
        # Interrompe se o bot for parado
        if self.stop_event.is_set():
            return

        try:
            # 1. Busca as posições (requer uma chamada API)
            all_open_positions = self.client.get_open_positions()

            # 2. Processa e filtra os símbolos
            # Garante que só há um nome por ativo em aberto (ex: BTC-USDT)
            open_symbols = sorted(
                list(
                    {pos['symbol'].upper() for pos in all_open_positions if pos.get('symbol')}
                )
            )

            # 3. Loga no formato solicitado
            if open_symbols:
                symbols_list = ', '.join(open_symbols)
                log_message = f"Posições abertas: {symbols_list}"
                self.log(log_message, level='INFO')
            else:
                self.log("Posições abertas: Nenhuma", level='INFO')

        except Exception as e:
            self.log(f"Erro no log periódico de posições: {e}", level='ERROR')

        # 4. Agenda a próxima execução (a cada 300.0 segundos)
        if not self.stop_event.is_set():
            self._position_timer = threading.Timer(300.0, self._log_open_positions_periodically)
            self._position_timer.start()

    def log(self, message, level='INFO'):
        """Envia uma mensagem de log para o sistema de logging central."""
        level = level.upper()
        if level == 'EXECUTE':
            self.logger.execute(message)
        elif level == 'SUCCESS':
            self.logger.log(logging.SUCCESS, message)
        elif level == 'WARNING':
            self.logger.warning(message)
        elif level == 'ERROR':
            self.logger.error(message)
        elif level == 'DEBUG':
            self.logger.debug(message)
        else:
            self.logger.info(message)

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
        # --- Busca inicial de ordens e informações ---
        api_symbol = position['symbol'].upper()
        position_side = "long" if position.get('side') == 'bid' else "short"
        entry_price = float(position.get('entry_price'))

        # Incrementa a idade da posição no cache
        self._position_age_cache[api_symbol] = self._position_age_cache.get(api_symbol, 0) + 1  # ADDED

        last_candle = market_data.iloc[-2]  # Candle fechado
        prev_candle = market_data.iloc[-3]

        current_sl_order = next((o for o in all_open_orders if
                                 o.get('symbol', '').upper() == api_symbol and o.get('order_type', '').startswith(
                                     'stop')), None)
        current_tp_order = next((o for o in all_open_orders if
                                 o.get('symbol', '').upper() == api_symbol and o.get('order_type', '').startswith(
                                     'take_profit')), None)

        if not self.market_info_cache.get(api_symbol):
            try:
                self.market_info_cache[api_symbol] = self.client.get_market_info(api_symbol)
            except Exception as e:
                self.log(f"Erro ao buscar market_info ({api_symbol}): {e}", level='ERROR')
                return False  # Não pode continuar sem tick_size
        tick_size = float(self.market_info_cache[api_symbol].get('tick_size', 0.01))

        # --- Lógica para criar SL inicial (se ausente) ---
        if not current_sl_order:
            self.log(f"Posição em {api_symbol} sem SL. Criando SL inicial com base no setup.", level='WARNING')
            sl_config = setup['stop_loss_config']
            sl_distance = entry_price * sl_config['value'] if sl_config['type'] == 'percentage' else last_candle.get(
                'ATR', entry_price * 0.01) * sl_config.get('value', 1.0)
            if sl_distance <= 0:
                self.log(f"Distância SL inválida ({sl_distance:.4f}) para {api_symbol}. Não criando SL.", level="ERROR")
            else:
                initial_sl_price = entry_price - sl_distance if position_side == 'long' else entry_price + sl_distance
                sl_rounded_str = round_to_increment(initial_sl_price, tick_size)
                # Cria APENAS o SL
                result = self.client.set_position_tpsl(
                    symbol=api_symbol,
                    position_side=position.get('side'),
                    position_size=str(position.get('amount')),
                    tick_size=tick_size,
                    stop_loss_price=sl_rounded_str,
                    take_profit_price=None  # Garante que só cria SL
                )
                self.log(f"Resultado da criação do SL inicial: {result}", level='EXECUTE')
                if not (result and result.get('success')):
                    self.log(f"Falha ao criar SL inicial para {api_symbol}.", level='ERROR')
                    return False  # Aborta se falhou em criar SL essencial
                else:
                    # Atualiza a variável local para a lógica TSL
                    current_sl_order = {'symbol': api_symbol, 'order_type': 'stop',
                                        'stop_price': sl_rounded_str}

        # --- Lógica para criar TP inicial (se ausente e modo passivo e sem TSL) ---
        tsl_config = setup.get('trailing_stop_config', {})
        should_remove_tp_on_trail = tsl_config.get('remove_tp_on_trail', False)
        if not current_tp_order and setup.get('exit_mode', 'passivo') == 'passivo' and not should_remove_tp_on_trail:
            self.log(f"Posição {api_symbol} sem TP (modo passivo). Criando TP inicial.", level='WARNING')
            sl_config = setup['stop_loss_config']
            sl_distance = entry_price * sl_config['value'] if sl_config['type'] == 'percentage' else last_candle.get(
                'ATR', entry_price * 0.01) * sl_config.get('value', 1.0)

            if sl_distance <= 0:
                self.log(f"Distância SL (do setup) inválida ({sl_distance:.4f}) para {api_symbol}. Não criando TP.",
                         level="ERROR")
            else:
                rrr = setup.get('take_profit_rrr', 1.5)
                initial_tp_price = entry_price + (sl_distance * rrr) if position_side == 'long' else entry_price - (
                        sl_distance * rrr)
                tp_rounded_str = round_to_increment(initial_tp_price, tick_size)

                # Cria APENAS o TP
                payload_tp = {"symbol": api_symbol, "position_side": position.get('side'),
                              "position_size": str(position.get('amount')),
                              "tick_size": tick_size, "stop_loss_price": None, "take_profit_price": tp_rounded_str}
                try:
                    result_tp = self.client.set_position_tpsl(**payload_tp)
                    self.log(f"Resultado criação TP inicial {api_symbol}: {result_tp}", level='EXECUTE')
                    if not (result_tp and result_tp.get('success')):
                        self.log(f"Falha ao criar TP inicial {api_symbol}.", level='ERROR')
                except Exception as e:
                    self.log(f"Erro CRÍTICO ao criar TP inicial {api_symbol}: {e}", level="ERROR")

        # --- Lógica de Trailing Stop (TSL) ---
        tsl_config = setup.get('trailing_stop_config', {})
        tsl_type = tsl_config.get('type')
        tsl_value = tsl_config.get('tsl_value')

        if not tsl_type or tsl_type == 'none':
            return False

        if not current_sl_order:
            return False

        current_sl_price = float(current_sl_order['stop_price'])
        potential_new_sl = None

        # --- 1. VERIFICAÇÃO DE BREAK-EVEN / ZONA DE LUCRO ---
        is_in_profit_zone = (position_side == 'long' and last_candle['high'] > entry_price) or \
                            (position_side == 'short' and last_candle['low'] < entry_price)

        # O TSL SÓ DEVE AGIR SE JÁ ESTIVER NA ZONA DE LUCRO
        if not is_in_profit_zone:
            self.log(f"TSL {api_symbol}: Ignorado. Posição não atingiu o Break-Even (High/Low no último candle)",
                     level='DEBUG')
            return False

        # --- 2. CÁLCULO DO POTENTIAL_NEW_SL ---

        if tsl_type == 'fixed_pct':
            if tsl_value is None or tsl_value <= 0: return False
            if position_side == 'long':
                potential_new_sl = last_candle['close'] * (1 - tsl_value)
            else:
                potential_new_sl = last_candle['close'] * (1 + tsl_value)

        elif tsl_type == 'high_low_trail':
            if tsl_value is None or tsl_value <= 0: return False
            buffer_price = last_candle['close'] * tsl_value

            if position_side == 'long' and last_candle['close'] > prev_candle['close']:
                potential_new_sl = last_candle['low'] - buffer_price
            elif position_side == 'short' and last_candle['close'] < prev_candle['close']:
                potential_new_sl = last_candle['high'] + buffer_price

        elif tsl_type == 'ema_trail':
            # Lógica anterior (tsl_value = min_profit_pct; buffer fixo de 0.2%)
            ema_9_prev = prev_candle.get('EMA_9')
            if ema_9_prev is None:
                self.log(f"EMA 9 não disponível no candle anterior para {api_symbol}.", level='ERROR')
                return False
            min_profit_pct = tsl_value if tsl_value is not None else 0.0
            safety_buffer_pct = 0.002
            if position_side == 'long':
                if last_candle['high'] > entry_price * (1 + min_profit_pct):
                    potential_new_sl = ema_9_prev * (1 - safety_buffer_pct)
                else:
                    return False
            else:
                if last_candle['low'] < entry_price * (1 - min_profit_pct):
                    potential_new_sl = ema_9_prev * (1 + safety_buffer_pct)
                else:
                    return False

        elif tsl_type == 'breakeven':
            if (position_side == 'long' and current_sl_price >= entry_price) or \
                    (position_side == 'short' and current_sl_price <= entry_price): return False
            trigger_rrr = tsl_config.get('breakeven_trigger_rrr', 1.0)
            sl_config = setup['stop_loss_config']
            initial_sl_distance = entry_price * sl_config['value'] if sl_config['type'] == 'percentage' else \
                last_candle['ATR'] * sl_config.get('value', 1.0)
            breakeven_buffer = 0.15 * last_candle['ATR']
            if position_side == 'long':
                breakeven_trigger_price = entry_price + (initial_sl_distance * trigger_rrr)
                if last_candle['high'] >= breakeven_trigger_price: potential_new_sl = entry_price + breakeven_buffer
            else:
                breakeven_trigger_price = entry_price - (initial_sl_distance * trigger_rrr)
                if last_candle['low'] <= breakeven_trigger_price: potential_new_sl = entry_price - breakeven_buffer

        elif tsl_type == 'atr':
            atr_value = last_candle['ATR']
            current_price = last_candle['close']
            atr_multiple = tsl_value if tsl_value is not None else tsl_config.get('tsl_atr_multiple', 2.0)
            atr_pct = atr_value / current_price if current_price > 0 else 0
            if atr_pct > 0.015:
                atr_multiple *= 1.25
            elif atr_pct < 0.005:
                atr_multiple *= 0.9
            min_distance = max(atr_value * 2, current_price * 0.002)
            safety_buffer = 2.5 * atr_value
            high_water_mark = market_data['high'].iloc[-min(len(market_data), 5):].max()
            low_water_mark = market_data['low'].iloc[-min(len(market_data), 5):].min()
            if position_side == 'long':
                potential_new_sl = high_water_mark - (atr_multiple * atr_value)
                potential_new_sl = min(potential_new_sl, current_price - safety_buffer)
                if current_price - potential_new_sl < min_distance: potential_new_sl = current_price - min_distance
            else:
                potential_new_sl = low_water_mark + (atr_multiple * atr_value)
                potential_new_sl = max(potential_new_sl, current_price + safety_buffer)
                if potential_new_sl - current_price < min_distance: potential_new_sl = current_price + min_distance

        # --- NOVOS MODOS TSL IMPLEMENTADOS ---

        elif tsl_type == 'profit_lock':
            # === NOVO: Profit Lock Dinâmico (Dicionário) ===
            # Garantir que o valor seja um dicionário (pode vir como string do DB)
            if isinstance(tsl_value, str):
                try:
                    tsl_value = ast.literal_eval(tsl_value)
                except:
                    tsl_value = {}

            if not isinstance(tsl_value, dict) or tsl_value.get("min_buffer") is None:
                self.log(f"TSL {api_symbol}: profit_lock ignorado. Parâmetros inválidos.", level='WARNING')
            else:
                min_buffer = tsl_value["min_buffer"]
                scale_factor = tsl_value.get("scale_factor", 0.5)
                max_lock = tsl_value.get("max_lock", 0.8)

                current_close = last_candle['close']
                profit_pct = (current_close / entry_price - 1) if position_side == 'long' else (
                            1 - current_close / entry_price)

                if profit_pct > min_buffer:
                    locked_profit = (profit_pct - min_buffer) * scale_factor
                    locked_profit = min(locked_profit, profit_pct * max_lock)

                    if position_side == 'long':
                        potential_new_sl = entry_price * (1 + locked_profit)
                    else:
                        potential_new_sl = entry_price * (1 - locked_profit)

        elif tsl_type == 'atr_dynamic':
            # === NOVO: ATR Dinâmico (Float Multiplicador Base) ===
            atr_value = last_candle.get('ATR')
            if atr_value is None or atr_value <= 0:
                self.log(f"TSL {api_symbol}: atr_dynamic ignorado. ATR indisponível.", level='DEBUG')
            else:
                current_price = last_candle['close']
                atr_pct = (atr_value / current_price) if current_price > 0 else 0

                base_mult = tsl_value if tsl_value is not None else 2.0

                if atr_pct < 0.005:
                    atr_mult = base_mult * 1.5
                elif atr_pct > 0.015:
                    atr_mult = base_mult * 0.8
                else:
                    atr_mult = base_mult

                if position_side == 'long':
                    potential_new_sl = last_candle['low'] - (atr_value * atr_mult)
                else:
                    potential_new_sl = last_candle['high'] + (atr_value * atr_mult)

        elif tsl_type == 'candle_range_confirm':
            # === NOVO: Candle Range Confirmado (Float Buffer) ===
            if len(market_data) < 4:
                self.log(f"TSL {api_symbol}: candle_range_confirm ignorado. Dados insuficientes (Min 4 candles).",
                         level='DEBUG')
            else:
                # O candle -3 é o que fecha ANTES do candle -2 (last_candle)
                prev_candle_for_confirm = market_data.iloc[-3]
                buffer = tsl_value if tsl_value is not None else 0.0

                if position_side == 'long':
                    # Confirmação: Candle atual (-2) E o anterior (-3) são de alta (close > open)
                    is_confirmed_up = (last_candle['close'] > last_candle['open']) and \
                                      (prev_candle_for_confirm['close'] > prev_candle_for_confirm['open'])

                    if is_confirmed_up:
                        lowest_low = min(last_candle['low'], prev_candle_for_confirm['low'])
                        potential_new_sl = lowest_low * (1 - buffer)

                elif position_side == 'short':
                    is_confirmed_down = (last_candle['close'] < last_candle['open']) and \
                                        (prev_candle_for_confirm['close'] < prev_candle_for_confirm['open'])

                    if is_confirmed_down:
                        highest_high = max(last_candle['high'], prev_candle_for_confirm['high'])
                        potential_new_sl = highest_high * (1 + buffer)

        elif tsl_type == 'ema_trail_delay':
            # === NOVO: EMA Trail com Delay (Dicionário) ===
            # Garantir que o valor seja um dicionário
            if isinstance(tsl_value, str):
                try:
                    tsl_value = ast.literal_eval(tsl_value)
                except:
                    tsl_value = {}

            if not isinstance(tsl_value, dict) or tsl_value.get("min_profit_pct") is None:
                self.log(f"TSL {api_symbol}: ema_trail_delay ignorado. Parâmetros inválidos.", level='WARNING')
            else:
                min_profit = tsl_value["min_profit_pct"]
                delay_bars = tsl_value.get("delay_bars", 2)
                buffer = tsl_value.get("safety_buffer_pct", 0.002)

                position_age = self._position_age_cache.get(api_symbol, 0)

                current_close = last_candle['close']
                profit_pct = (current_close / entry_price - 1) if position_side == 'long' else (
                            1 - current_close / entry_price)

                if profit_pct >= min_profit:
                    if position_age >= delay_bars:
                        ema_9_prev = prev_candle.get('EMA_9')

                        if ema_9_prev is None:
                            self.log(f"TSL {api_symbol}: EMA 9 indisponível para ema_trail_delay.", level='ERROR')
                        else:
                            if position_side == 'long':
                                potential_new_sl = ema_9_prev * (1 - buffer)
                            elif position_side == 'short':
                                potential_new_sl = ema_9_prev * (1 + buffer)
                    else:
                        self.log(
                            f"TSL {api_symbol}: EMA Delay (Age:{position_age}/{delay_bars}). Profit OK. Delay Pendente.",
                            level='DEBUG')

        # --- FIM DOS NOVOS MODOS TSL ---

        if potential_new_sl is None:
            return False

        # --- 3. VERIFICAÇÃO DE MELHORIA E GATILHO DE MUDANÇA ---

        # Verifica se é uma melhoria (SL se movendo para mais longe do preço de entrada)
        is_improvement = (position_side == 'long' and potential_new_sl > current_sl_price) or \
                         (position_side == 'short' and potential_new_sl < current_sl_price)

        # Efetivamente, o TSL só deve ser ativado se for > que o preço de entrada (long)
        is_better_than_entry = (position_side == 'long' and potential_new_sl > entry_price) or \
                               (position_side == 'short' and potential_new_sl < entry_price)

        if is_improvement and is_better_than_entry:
            update_buffer = 1.0 * last_candle['ATR']
            if abs(potential_new_sl - current_sl_price) < update_buffer:
                self.log(f"TSL {api_symbol}: Mudança muito pequena (<\~1 ATR). Ignorando.", level='DEBUG')
                return False

            self.log(
                f"TRAILING STOP (UPDATE) para {api_symbol}: Movendo de ${current_sl_price:.4f} para ${potential_new_sl:.4f} (Modo: {tsl_type.upper()})",
                level='EXECUTE')
            new_sl_rounded_str = round_to_increment(potential_new_sl, tick_size)

            # Mantendo a lógica original de cancelar SL antigo primeiro
            cancel_sl_result = self.client.cancel_stop_order(api_symbol, current_sl_order['order_id'])
            self.log(f"Resultado do cancelamento do SL antigo (TSL): {cancel_sl_result}", level='EXECUTE')

            if not (cancel_sl_result and cancel_sl_result.get('success')):
                self.log(f"FALHA ao cancelar SL antigo para {api_symbol} (TSL). Abortando TSL.", level='ERROR')
                return False

            # Remove TP se necessário (lógica original mantida)
            if tsl_config.get('remove_tp_on_trail', False):
                tp_order_to_remove = next((o for o in all_open_orders if
                                           o.get('symbol', '').upper() == api_symbol and o.get('order_type',
                                                                                               '').startswith(
                                               'take_profit')), None)
                if tp_order_to_remove:
                    self.log(f"TSL {api_symbol}: Removendo TP (ID: {tp_order_to_remove.get('order_id')}).")
                    try:
                        cancel_tp_result = self.client.cancel_stop_order(api_symbol, tp_order_to_remove['order_id'])
                        self.log(f"Resultado da remoção do TP (TSL): {cancel_tp_result}", level='EXECUTE')
                        if not (cancel_tp_result and cancel_tp_result.get('success')):
                            self.log(f"Falha ao remover TP (TSL) para {api_symbol}.", level='ERROR')
                    except Exception as e:
                        self.log(f"Erro ao remover TP (TSL) para {api_symbol}: {e}", level='ERROR')

            # Cria o NOVO SL (sem TP)
            result_new_sl = self.client.set_position_tpsl(
                symbol=api_symbol,
                position_side=position.get('side'),
                position_size=str(position.get('amount')),
                tick_size=tick_size,
                stop_loss_price=new_sl_rounded_str,
                take_profit_price=None  # Garante que só (re)cria SL
            )
            self.log(f"Resultado da criação do novo SL (TSL): {result_new_sl}", level='EXECUTE')
            if result_new_sl and result_new_sl.get('success'):
                return True  # Indica que TSL moveu o SL
            else:
                self.log(f"Falha ao criar novo SL (TSL) para {api_symbol}.", level='ERROR')
                # Tentar recriar SL inicial (aqui o bot deveria ter uma rotina de recuperação mais robusta)
                return False

        return False

    # ... (Restante da classe - on_btc_trend_update, on_new_candle, run, stop - com ajustes de cache)

    def on_btc_trend_update(self, cache_key):
        """
        Callback acionado APENAS pelo candle de 1h do BTC.
        Calcula e armazena a tendência do BTC.
        """
        # cache_key é passado pelo listener, pode ser None se chamado manualmente
        with self.btc_trend_lock:
            try:
                self.log("Calculando filtro de tendência BTC (1h)...", level='INFO')
                new_trend = get_btc_trend_filter(self.global_config.get('data_source_exchanges', ['binance']))

                if new_trend != self.btc_trend_direction:
                    self.log(f"Filtro de Tendência BTC ATUALIZADO: {new_trend.upper()}", level='WARNING')
                else:
                    self.log(f"Filtro de Tendência BTC verificado: {new_trend.upper()} (sem mudança)", level='INFO')

                self.btc_trend_direction = new_trend

            except Exception as e:
                self.log(f"Erro ao calcular tendência BTC: {e}", level='ERROR')
                # Mantém a tendência anterior em caso de erro

    def on_new_candle(self, cache_key: tuple[str, str]):

        with self.processing_lock:
            try:
                # 1. Identificar setups para este candle
                setups_to_check = self.setups_by_key.get(cache_key)
                if not setups_to_check:
                    self.log(f"Callback {cache_key} recebido, mas nenhum setup mapeado (estranho?).", level='WARNING')
                    return  # Sai do 'with'

                ws_symbol, timeframe = cache_key
                self.log(f"Verificando candle {cache_key} ({len(setups_to_check)} setup(s))...", level='DEBUG')

                # 2. Obter dados (do cache, que acabou de ser atualizado)
                market_data = self.ws_manager.get_klines(ws_symbol, timeframe)

                strategy_name_example = setups_to_check[0].get('strategy_name')
                min_candles_req = self.STRATEGIES.get(strategy_name_example, {}).get('min_candles', 200)

                if market_data is None or len(market_data) < min_candles_req:
                    self.log(
                        f"Dados {cache_key} insuficientes no cache ({len(market_data) if market_data is not None else 0}/{min_candles_req}). Pulando verificação.",
                        level='WARNING')
                    return  # Sai do 'with'

                # 3. VERIFICAÇÃO DE TIMESTAMP (ESSENCIAL - DENTRO DO LOCK)
                last_closed_candle_timestamp = market_data.index[-2].value // 1_000_000  # MS UTC
                last_processed = self._last_processed_timestamp.get(cache_key)

                if last_processed is not None and last_closed_candle_timestamp <= last_processed:
                    self.log(
                        f"Candle {cache_key} ({last_closed_candle_timestamp}) já foi processado (pós-lock). Pulando.",
                        level='DEBUG')
                    return  # Sai do 'with'

                current_time_ms = int(time.time() * 1000)
                account_info, all_open_orders, all_open_positions = {}, [], []

                # Reutiliza o cache se foi atualizado há menos de 60 segundos (60000ms) e contém dados
                cache_expiry_ms = 60000
                if current_time_ms - self._api_cache['timestamp'] < cache_expiry_ms and self._api_cache['account_info']:
                    self.log("Reutilizando cache de dados da conta (API calls evitadas por 1 minuto).", level='DEBUG')
                    account_info = self._api_cache['account_info']
                    all_open_orders = self._api_cache['all_open_orders']
                    all_open_positions = self._api_cache['all_open_positions']
                else:
                    self.log(f"Buscando dados da conta via API (cache expirado/{cache_expiry_ms / 1000:.0f}s)...",
                             level='DEBUG')
                    try:
                        account_info = self.client.get_account_info()
                        if not account_info:
                            self.log("Não foi possível obter informações da conta no callback, pulando ciclo.",
                                     level='ERROR')
                            return

                        all_open_orders = self.client.get_open_orders()
                        all_open_positions = self.client.get_open_positions()

                        # Atualiza o cache (apenas se bem-sucedido)
                        self._api_cache['timestamp'] = current_time_ms
                        self._api_cache['account_info'] = account_info
                        self._api_cache['all_open_orders'] = all_open_orders
                        self._api_cache['all_open_positions'] = all_open_positions

                    except Exception as e:
                        self.log(f"Erro ao buscar dados da conta no callback: {e}", level='ERROR')
                        return

                # Pós-processamento dos dados (comum ao cache e à API)
                balance = float(account_info.get('balance', 0.0))
                open_positions_map = {pos['symbol'].upper(): pos for pos in all_open_positions}

                # 5. Calcular indicadores (UMA VEZ por evento)
                try:
                    df = add_all_indicators(market_data.copy())
                    last_candle, prev_candle = df.iloc[-2], df.iloc[-3]
                except Exception as e:
                    self.log(f"Erro indicadores {ws_symbol} ({timeframe}): {e}", level="ERROR")
                    return  # Sai do 'with'

                # 6. *** MODIFICADO: Apenas lê o valor da tendência ***
                btc_trend_direction = self.btc_trend_direction
                # Os logs de tendência agora são gerados pelo 'on_btc_trend_update'

                # Marca como processado ANTES de tentar ordens
                self._last_processed_timestamp[cache_key] = last_closed_candle_timestamp

                # 7. Iterar APENAS nos setups relevantes
                assets_sem_sinal = []
                for setup in setups_to_check:
                    if self.stop_event.is_set(): break  # Verifica se o bot foi parado

                    base_symbol = setup["base_currency"].upper()
                    api_symbol = base_symbol  # Assumindo que api_symbol é o base_symbol
                    strategy_name = setup["strategy_name"]
                    strategy_logic = self.STRATEGIES[strategy_name]

                    if api_symbol in open_positions_map:
                        position = open_positions_map[api_symbol]
                        position_side = "long" if position.get('side') == 'bid' else "short"

                        if self._manage_trailing_stop(setup, position, all_open_orders, df):
                            self.log(f"TSL gerenciado para {api_symbol}. Ressinc ordens.")
                            all_open_orders = self.client.get_open_orders()  # Re-busca ordens

                        exit_mode = setup.get('exit_mode', 'passivo')
                        if exit_mode == 'ativo' and strategy_logic.get('exit'):
                            if strategy_logic['exit'](last_candle, prev_candle, position_side):
                                self.log(f"SINAL DE SAÍDA ESTRATÉGICA DETECTADO {api_symbol}! Enviando ordem limite.",
                                         level='EXECUTE')
                                closing_side = "SELL" if position_side == 'long' else "BUY"
                                try:
                                    if not self.market_info_cache.get(api_symbol): self.market_info_cache[
                                        api_symbol] = self.client.get_market_info(api_symbol)
                                    market_info = self.market_info_cache[api_symbol]
                                    tick_size = float(market_info.get('tick_size', 0.01))
                                    orderbook = self.client.get_orderbook(api_symbol)
                                    if not orderbook: self.log(
                                        f"Não foi possível obter order book para fechar {api_symbol}.",
                                        level='ERROR'); continue
                                    price_buffer = 5 * tick_size
                                    if closing_side == "SELL":
                                        best_bid_price = float(
                                            orderbook['l'][0][0]['p'])
                                        limit_price = best_bid_price + price_buffer
                                    else:
                                        best_ask_price = float(
                                            orderbook['l'][1][0]['p'])
                                        limit_price = best_ask_price - price_buffer
                                    limit_price_str = round_to_increment(limit_price, tick_size)
                                    result = self.client.create_limit_order(symbol=api_symbol, side=closing_side,
                                                                            amount=str(position['amount']),
                                                                            price=limit_price_str, reduce_only=True,
                                                                            tick_size=tick_size)
                                    self.log(f"Resultado envio ordem fechamento {api_symbol}: {result}",
                                             level='EXECUTE')
                                except Exception as e:
                                    self.log(f"Erro ao enviar ordem limite fechamento {api_symbol}: {e}", level="ERROR")
                                continue

                        # Saída de Emergência (ORIGINAL MANTIDA)
                        entry_price = float(position.get('entry_price'))
                        sl_config = setup['stop_loss_config']
                        sl_distance = entry_price * sl_config['value'] if sl_config[
                                                                              'type'] == 'percentage' else last_candle.get(
                            'ATR', entry_price * 0.01) * sl_config.get('value',
                                                                       1.0)  # <<< ORIGINAL USAVA last_candle >>>
                        initial_stop_loss_price = entry_price - sl_distance if position_side == 'long' else entry_price + sl_distance
                        emergency_exit = False
                        closing_side = None
                        if position_side == 'long' and last_candle['low'] < initial_stop_loss_price:
                            emergency_exit = True
                            closing_side = "SELL"
                            self.log(
                                f"SAÍDA EMERGÊNCIA (< SL) {api_symbol}!", level='EXECUTE')
                        elif position_side == 'short' and last_candle['high'] > initial_stop_loss_price:
                            emergency_exit = True
                            closing_side = "BUY"
                            self.log(
                                f"SAÍDA EMERGÊNCIA (> SL) {api_symbol}!", level='EXECUTE')
                        if emergency_exit:
                            try:
                                if not self.market_info_cache.get(api_symbol): self.market_info_cache[
                                    api_symbol] = self.client.get_market_info(api_symbol)
                                tick_size = float(self.market_info_cache[api_symbol].get('tick_size', 0.01))
                                result = self.client.create_market_order(symbol=api_symbol, side=closing_side,
                                                                         amount=str(position['amount']),
                                                                         reduce_only=True, tick_size=tick_size)
                                self.log(f"Resultado fechamento emergência {api_symbol}: {result}", level='EXECUTE')
                            except Exception as e:
                                self.log(f"Erro ao fechar {api_symbol} (emergência): {e}", level='ERROR')

                            # Limpar cache de idade da posição ao fechar
                            self._position_age_cache.pop(api_symbol, None)  # ADDED

                            continue

                        continue  # Pula para o próximo setup

                    # --- Lógica de Entrada ---
                    side = None
                    direction_mode = setup.get('direction_mode', 'long_short')
                    # *** USA O VALOR ARMAZENADO 'btc_trend_direction' ***
                    can_go_long = (direction_mode in ['long_short', 'long_only']) or (
                            direction_mode == 'btc_trend' and btc_trend_direction != 'short_only')
                    can_go_short = (direction_mode in ['long_short', 'short_only']) or (
                            direction_mode == 'btc_trend' and btc_trend_direction != 'long_only')

                    if can_go_long and strategy_logic.get('long_entry') and strategy_logic['long_entry'](last_candle,
                                                                                                         prev_candle):
                        side = "BUY"
                    elif can_go_short and strategy_logic.get('short_entry') and strategy_logic['short_entry'](
                            last_candle, prev_candle):
                        side = "SELL"

                    if side:
                        self.log(f"SINAL {side} DETECTADO para {api_symbol} (Estrategia: {strategy_name})",
                                 level='EXECUTE')
                        self.log(f"Configurando alavancagem e modo de margem para {api_symbol}...")
                        try:
                            leverage_to_set = setup.get('leverage')
                            self.client.update_leverage(api_symbol, int(leverage_to_set))
                            self.client.update_margin_mode(api_symbol, is_isolated=True)
                            self.log(f"Configurações para {api_symbol} aplicadas: {leverage_to_set}x, ISOLATED.")
                        except Exception as e:
                            self.log(f"Falha ao configurar margem/alavancagem para {api_symbol}: {e}",
                                     level="ERROR")
                            continue

                        if not self.market_info_cache.get(api_symbol):
                            try:
                                self.market_info_cache[api_symbol] = self.client.get_market_info(api_symbol)
                            except Exception as e:
                                self.log(f"Erro info mercado ordem {api_symbol}: {e}", level="ERROR")
                                continue
                        market_info = self.market_info_cache[api_symbol]
                        tick_size = float(market_info.get('tick_size', 0.01))
                        lot_size = float(market_info.get('lot_size', 0.01))

                        try:
                            orderbook = self.client.get_orderbook(api_symbol)
                            if not orderbook: self.log(f"Não foi possível obter o order book para {api_symbol}.",
                                                       level='ERROR'); continue
                            if side == "BUY":
                                limit_price = float(orderbook['l'][0][0]['p']) - tick_size
                            else:
                                limit_price = float(orderbook['l'][1][0]['p']) + tick_size
                        except Exception as e:
                            self.log(f"Erro ao obter orderbook para {api_symbol}: {e}. Usando último fechado.",
                                     level="WARNING")
                            limit_price = last_candle['close']  # Fallback para último fechado

                        limit_price_str = round_to_increment(limit_price, tick_size)
                        entry_price = float(limit_price_str)  # Preço estimado

                        sl_config = setup['stop_loss_config']
                        sl_distance = entry_price * sl_config['value'] if sl_config[
                                                                              'type'] == 'percentage' else last_candle.get(
                            'ATR', entry_price * 0.01) * sl_config.get('value', 1.0)
                        if sl_distance <= 0: self.log(f"Distância SL inválida ({sl_distance:.4f}) {api_symbol}.",
                                                      level="ERROR"); continue
                        rrr = setup['take_profit_rrr']
                        risk_percentage = setup['risk_per_trade']

                        stop_loss_price = entry_price - sl_distance if side == "BUY" else entry_price + sl_distance
                        take_profit_price = entry_price + (sl_distance * rrr) if side == "BUY" else entry_price - (
                                sl_distance * rrr)
                        amount_raw = calculate_order_amount(balance, entry_price, stop_loss_price, risk_percentage)
                        amount_rounded_str = round_to_increment(amount_raw, lot_size)
                        final_amount = float(amount_rounded_str)

                        if final_amount <= 0:
                            self.log(f"Quantidade da ordem para {api_symbol} é zero ou negativa ({final_amount}).",
                                     level='WARNING')
                            continue

                        # Resetar a idade da posição ao abrir uma nova
                        self._position_age_cache[api_symbol] = 0  # ADDED

                        stop_loss_price_rounded_str = round_to_increment(stop_loss_price, tick_size)
                        take_profit_price_rounded_str = round_to_increment(take_profit_price, tick_size)

                        log_details = (
                            f"ORDEM LIMITE PREPARADA -> Ativo: {api_symbol}, Lado: {side}, Preço: ${limit_price_str}, "
                            f"Qtd: {amount_rounded_str}, TP: ${take_profit_price_rounded_str}, SL: ${stop_loss_price_rounded_str}")
                        self.log(log_details, level='EXECUTE')
                        payload_para_ordem = {"symbol": api_symbol, "side": side, "amount": amount_rounded_str,
                                              "price": limit_price_str,
                                              "take_profit_price": take_profit_price_rounded_str,
                                              "stop_loss_price": stop_loss_price_rounded_str, "tick_size": tick_size}
                        try:
                            result = self.client.create_limit_order(**payload_para_ordem)
                            self.log(f"Resultado do envio da ordem: {result}", level='EXECUTE')
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
                                    self.log(f"Ordem {order_id_value} salva no banco de dados de historico.")
                                else:
                                    self.log(f"Falha ao extrair ID da ordem da resposta: {order_data_response}",
                                             level='WARNING')
                            elif result and not result.get('success'):
                                self.log(f"Falha ao criar ordem: {result.get('error', 'Erro desconhecido')}",
                                         level='ERROR')
                        except Exception as e:
                            self.log(f"Erro CRÍTICO ao criar ordem para {api_symbol}: {e}", level="ERROR")

                    else:
                        assets_sem_sinal.append(api_symbol)

                if assets_sem_sinal:
                    unique_no_signal = sorted(list(set(assets_sem_sinal)))
                    if unique_no_signal:
                        self.log(f"Nenhum sinal de entrada para: {', '.join(unique_no_signal)}", level='INFO')

            except Exception as e:
                self.log(f"CRITICO no callback on_new_candle {cache_key}: {e}", level='ERROR')
            # O 'with' garante que o lock seja liberado aqui

    def run(self):
        self.log(f"Bot iniciado.", level='SUCCESS')
        num_setups = len(self.account_config.get('markets_to_trade', []))
        self.log(f"Monitorando {num_setups} setups.")

        try:
            # 3. Mapeia setups e registra listeners
            self.log("Registrando listeners de candles...")
            unique_cache_keys = set()
            btc_key_local = ('BTC-USDT', '1h')
            for setup in self.account_config.get('markets_to_trade', []):
                base_symbol = setup.get("base_currency")
                strategy_name = setup.get("strategy_name")

                # *** ADICIONADO: Verifica se o filtro btc está em uso ***
                if setup.get('direction_mode') == 'btc_trend':
                    self.uses_btc_trend = True

                if not (base_symbol and strategy_name and strategy_name in self.STRATEGIES):
                    continue

                strategy_logic = self.STRATEGIES[strategy_name]
                timeframe = strategy_logic.get('timeframe')

                if not timeframe:
                    self.log(f"Setup {base_symbol} (Estratégia: {strategy_name}) sem 'timeframe'. Pulando.",
                             level='WARNING')
                    continue

                ws_symbol = f"{base_symbol.upper()}-USDT"
                cache_key = (ws_symbol, timeframe)

                # Adiciona o setup à lista daquele cache_key
                self.setups_by_key[cache_key].append(setup)
                unique_cache_keys.add(cache_key)

            # Registra o callback UMA VEZ por cache_key (para os setups)
            for cache_key in unique_cache_keys:
                self.ws_manager.register_listener(cache_key, self.on_new_candle)

            self.log(f"Registrado para {len(unique_cache_keys)} pares de setup.")

            # *** ADICIONADO: Registra o listener do BTC 1h (se necessário) ***
            if self.uses_btc_trend:
                self.log("Registrando listener de tendência BTC (1h)...")
                self.ws_manager.register_listener(btc_key_local, self.on_btc_trend_update)
                # Chama uma vez no início para obter o valor inicial
                self.log("Obtendo valor inicial da tendência BTC...")
                self.on_btc_trend_update(None)

            self.log("Realizando verificação inicial dos sinais...")
            initial_check_keys = list(unique_cache_keys)  # Cria uma cópia da lista de chaves
            for check_key in initial_check_keys:
                if self.stop_event.is_set(): break  # Permite parar durante a verificação inicial
                self.log(f"Verificação inicial para {check_key}...", level='INFO')
                try:
                    # Chama on_new_candle para cada par/timeframe dos setups
                    # A função on_new_candle já tem o lock e a lógica de verificação de timestamp
                    self.on_new_candle(check_key)
                except Exception as e_initial:
                    self.log(f"Erro durante verificação inicial de {check_key}: {e_initial}", level="ERROR")

            self.log("Verificação inicial concluída. Aguardando novos candles...")

            self._log_open_positions_periodically()
            self._cancel_open_limit_orders_periodically()

            # 4. Mantém a thread viva esperando o evento de parada
            self.stop_event.wait()

        except Exception as e:
            self.log(f"CRITICO no setup do bot {self.account_name}: {e}", level='ERROR')
        finally:
            # 5. Limpeza ao parar
            self.log("Removendo listeners de candles...")
            for cache_key in self.setups_by_key.keys():
                self.ws_manager.unregister_listener(cache_key, self.on_new_candle)

            # *** ADICIONADO: Limpa o listener do BTC (se necessário) ***
            if self.uses_btc_trend:
                self.log("Removendo listener de tendência BTC (1h)...")
                btc_key = ('BTC-USDT', '1h')
                self.ws_manager.unregister_listener(btc_key, self.on_btc_trend_update)

            if self._position_timer and self._position_timer.is_alive():
                self._position_timer.cancel()

            if self._cancel_timer and self._cancel_timer.is_alive():
                self._cancel_timer.cancel()

            self.log("Bot foi parado.", level="WARNING")

    def stop(self):
        # Esta função permanece a mesma. Apenas seta o evento.
        self.stop_event.set()
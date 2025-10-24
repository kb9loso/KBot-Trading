# KBot-Trading/apex_client.py
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Any, Optional
import decimal

# Importa a classe base do nosso contrato
from exchange_interface import BaseExchangeClient

from apexomni.http_public import HttpPublic
from apexomni.http_private_sign import HttpPrivateSign
from apexomni.constants import APEX_OMNI_HTTP_MAIN, NETWORKID_MAIN

def round_to_increment(value: float, increment: float) -> str:
    """Arredonda um valor para o múltiplo mais próximo de um incremento (tick/lot size)."""
    if increment <= 0:
        return f"{value:.8f}"
    value_dec = Decimal(str(value))
    increment_dec = Decimal(str(increment))
    rounded_value = (value_dec / increment_dec).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * increment_dec
    return "{:f}".format(rounded_value.normalize())

class ApexClient(BaseExchangeClient):
    """
    Cliente para interagir com a API REST da Apex Omni Exchange,
    utilizando a biblioteca oficial apexomni para autenticação.
    Gerencia internamente um cliente para consultas (HttpPrivate_v3) e
    outro para assinaturas (HttpPrivateSign).
    """

    def __init__(self, api_key: str, api_secret: str, passphrase: str, zk_seeds: str, zk_l2key: str):
        try:
            # Cliente para chamadas públicas (não autenticadas)
            self.public_client = HttpPublic(APEX_OMNI_HTTP_MAIN)

            # Cliente para TODAS as chamadas autenticadas (GET e POST)
            self.client = HttpPrivateSign(
                APEX_OMNI_HTTP_MAIN,
                network_id=NETWORKID_MAIN,
                zk_seeds=zk_seeds,
                zk_l2Key=zk_l2key,
                api_key_credentials={'key': api_key, 'secret': api_secret, 'passphrase': passphrase}
            )
            print("Cliente Apex (HttpPrivateSign) inicializado com sucesso.")

            # Inicializa os atributos de cache
            self.symbols_info_cache = None
            self.last_cache_time = 0

        except Exception as e:
            print(f"ERRO CRÍTICO ao inicializar o cliente HttpPrivateSign da Apex: {e}")
            raise e

    # --- MÉTODOS DE MAPEAMENTO ---
    def _map_position_from_apex(self, apex_position: Dict) -> Dict:
        side_from_api = apex_position.get('side', '').upper()
        size = float(apex_position.get('size', 0.0))
        symbol = apex_position.get('symbol', '').split('-')[0]

        internal_side = 'bid' if side_from_api == 'LONG' else 'ask'

        return {
            'symbol': symbol,
            'side': internal_side,
            'amount': abs(size),
            'entry_price': float(apex_position.get('entryPrice', 0.0)),
            'pnl': float(apex_position.get('unrealizedPnl', 0.0)),
            'leverage': float(apex_position.get('leverage', 1.0)),
            'created_at': int(apex_position.get('createdAt', 0))
        }

    def _map_account_info_from_apex(self, apex_balance: Dict, apex_account_details: Dict) -> Dict:
        if not apex_balance or 'data' not in apex_balance: return {'balance': 0.0, 'account_equity': 0.0,
                                                                   'available_to_spend': 0.0}
        balance_data = apex_balance.get('data', {})
        equity = float(balance_data.get('totalEquityValue', 0.0))
        available = float(balance_data.get('totalValueWithoutDiscount', 0.0))
        return {'balance': equity, 'account_equity': equity, 'available_to_spend': available}

    def _map_trade_from_apex(self, apex_trade: Dict) -> Dict:
        entry_price_raw = apex_trade.get("averagePrice") or apex_trade.get("price")

        return {"history_id": apex_trade.get("id"),
                "order_id": apex_trade.get("orderId"),
                "client_order_id": apex_trade.get("clientOrderId"),
                "symbol": apex_trade.get("symbol", "").split('-')[0],
                "side": apex_trade.get("side", "").lower(),
                "amount": float(apex_trade.get("size") or 0.0),
                "price": float(apex_trade.get("price") or 0.0),
                "fee": float(apex_trade.get("fee") or 0.0),
                "pnl": 0.0,
                "entry_price": float(entry_price_raw or 0.0),
                "event_type": apex_trade.get("status", "").lower(),
                "created_at": apex_trade.get("createdAt"),
                "cause": apex_trade.get("cancelReason", "").lower()
                }

    def _map_order_from_apex(self, apex_order: Dict) -> Dict:
        return {"order_id": apex_order.get("id"),
                "client_order_id": apex_order.get("clientOrderId"),
                "symbol": apex_order.get("symbol", "").split('-')[0],
                "side": apex_order.get("side", "").lower(),
                "initial_price": float(apex_order.get("price") or 0.0),
                "average_filled_price": float(apex_order.get("latestMatchFillPrice") or 0.0),
                "amount": float(apex_order.get("size") or 0.0),
                "filled_amount": float(apex_order.get("cumSuccessFillSize") or 0.0),
                "order_status": apex_order.get("status", "").lower(),
                "order_type": apex_order.get("type", "").lower(),
                "stop_price": apex_order.get("triggerPrice"),
                "reduce_only": apex_order.get("reduceOnly"),
                "reason": apex_order.get("cancelReason", "").lower(),
                "created_at": apex_order.get("createdAt"),
                "updated_at": apex_order.get("updatedTime")}

    def _map_orderbook_from_apex(self, apex_orderbook: Dict) -> Dict:
        if not apex_orderbook or 'b' not in apex_orderbook or 'a' not in apex_orderbook: return None
        bids = [{'p': item[0], 'q': item[1]} for item in apex_orderbook['b']]
        asks = [{'p': item[0], 'q': item[1]} for item in apex_orderbook['a']]
        return {'l': [bids, asks]}

    def _map_ticker_from_apex(self, apex_ticker: Dict) -> Dict:
        return {'symbol': apex_ticker.get("symbol", "").replace("USDT", ""), 'mark': apex_ticker.get("markPrice"),
                'price': apex_ticker.get("lastPrice")}

    def _get_all_symbols_info(self, force_refresh: bool = False) -> Optional[Dict]:
        """
        Busca e armazena em cache as configurações de todos os símbolos da exchange.
        Este é um metodo auxiliar interno.
        """
        current_time = time.time()
        if not self.symbols_info_cache or force_refresh or (current_time - self.last_cache_time > 3600):
            try:
                response = self.client.configs_v3()
                if response and 'data' in response and 'contractConfig' in response['data']:
                    self.symbols_info_cache = {item['symbol']: item for item in response['data']['contractConfig']['perpetualContract']}
                    self.last_cache_time = current_time
            except Exception as e:
                print(f"Erro ao buscar configurações de símbolos da Apex: {e}")
                self.symbols_info_cache = None
        return self.symbols_info_cache

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        try:
            balance_response = self.client.get_account_balance_v3()
            account_response = self.client.get_account_v3()
            if balance_response and balance_response.get('data') and account_response:
                return self._map_account_info_from_apex(balance_response, account_response)
            return None
        except Exception as e:
            print(f"Erro no get_account_info da Apex: {e}")
            return None

    def get_open_positions(self, market: Optional[str] = None) -> List[Dict]:
        try:
            account_data = self.client.get_account_v3()
            if not (account_data and 'positions' in account_data): return []
            all_positions_raw = account_data['positions']
            all_open_positions = [self._map_position_from_apex(p) for p in all_positions_raw if
                                  float(p.get('size', "0")) != 0]
            if market:
                market_base_symbol = market.split('/')[0].upper()
                return [p for p in all_open_positions if p['symbol'].upper() == market_base_symbol]
            return all_open_positions
        except Exception as e:
            print(f"Erro no get_open_positions da Apex: {e}")
            return []

    def get_trade_history(self, start_time_ms: int, end_time_ms: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Busca o histórico de trades (preenchimentos) com suporte a paginação."""
        all_trades = []
        current_page = 0
        # A API rejeitou 1000, então usamos um limite menor e seguro por página. 100 é um padrão comum.
        page_limit = 100
        try:
            while True:
                response = self.client.fills_v3(
                    beginTimeInclusive=start_time_ms,
                    endTimeExclusive=end_time_ms,
                    limit=page_limit,
                    page=current_page
                )
                if (response and
                        isinstance(response.get('data'), dict) and
                        isinstance(response['data'].get('orders'), list)):
                    trades_page_data = response['data']['orders']
                    all_trades.extend(trades_page_data)
                    # Se a API retornou menos trades que o limite, significa que chegamos à última página.
                    if len(trades_page_data) < page_limit:
                        break

                    current_page += 1
                else:
                    # Se a resposta for inválida ou não contiver mais dados, encerra o loop.
                    break
                # --- FIM DA LÓGICA DE PAGINAÇÃO ---

            return [self._map_trade_from_apex(trade) for trade in all_trades]

        except Exception as e:
            print(f"Erro no get_trade_history da Apex: {e}")
            return []

    def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            # Usar orderType='HISTORY' não é necessário se não estivermos filtrando ativamente
            response = self.client.history_orders_v3(limit=limit, orderType='HISTORY')
            if (response and
                    isinstance(response.get('data'), dict) and
                    isinstance(response['data'].get('orders'), list)):
                orders_list = response['data']['orders']
                return [self._map_order_from_apex(order) for order in orders_list]
            return []
        except Exception as e:
            print(f"Erro no get_order_history da Apex: {e}")
            return []

    # def get_open_orders(self) -> List[Dict[str, Any]]:
    #     try:
    #         response = self.client.open_orders_v3()
    #         print(f"DEBUG: Resposta de open_orders_v3: {response}")
    #
    #         if response and isinstance(response.get('data'), list):
    #             orders_list = response['data']
    #             # Mapeia a lista completa de ordens para o formato padrão do bot
    #             return [self._map_order_from_apex(order) for order in orders_list]
    #
    #         return []
    #
    #     except Exception as e:
    #         print(f"Erro no get_open_orders da Apex: {e}")
    #         return []

    def get_open_orders(self) -> List[Dict[str, Any]]:
        all_open_orders_raw = []

        try:
            # 1. Busca as ordens ativas no livro (LIMIT)
            active_orders_response = self.client.open_orders_v3()
            if active_orders_response and isinstance(active_orders_response.get('data'), list):
                all_open_orders_raw.extend(active_orders_response['data'])

            # 2. Busca as ordens condicionais (STOP_MARKET, TAKE_PROFIT_MARKET)
            conditional_orders_response = self.client.history_orders_v3(orderType='CONDITION', limit=100)

            # --- INÍCIO DA CORREÇÃO ---
            # Acessa a lista de ordens no caminho correto: response -> data -> orders
            if (conditional_orders_response and
                    isinstance(conditional_orders_response.get('data'), dict) and
                    isinstance(conditional_orders_response['data'].get('orders'), list)):
                all_recent_conditional = conditional_orders_response['data']['orders']

                # Filtra para manter apenas as ordens que ainda não foram acionadas ('UNTRIGGERED')
                untriggered_orders = [
                    order for order in all_recent_conditional
                    if order.get('status') == 'UNTRIGGERED'
                ]
                all_open_orders_raw.extend(untriggered_orders)
            # --- FIM DA CORREÇÃO ---

            # 3. Mapeia a lista unificada de ordens para o formato padrão do bot
            if not all_open_orders_raw:
                return []
            return [self._map_order_from_apex(order) for order in all_open_orders_raw]

        except Exception as e:
            print(f"Erro no get_open_orders da Apex: {e}")
            return []

    def get_market_info(self, symbol: str) -> Dict:
        try:
            response = self.client.configs_v3()
            if not response:
                return {}

            # Navega de forma segura pela nova hierarquia do JSON
            data_base = response.get('data', response)
            contract_config = data_base.get('contractConfig', {})
            perpetual_contracts = contract_config.get('perpetualContract', [])

            if perpetual_contracts:
                # Constrói o símbolo no formato esperado em 'crossSymbolName' (ex: "SOLUSDT")
                symbol_to_find = f"{symbol.upper()}USDT"

                for contract_details in perpetual_contracts:
                    # Compara com a chave 'crossSymbolName'
                    if contract_details.get('crossSymbolName') == symbol_to_find:
                        return {
                            'symbol': contract_details.get('symbol'),
                            'tick_size': float(contract_details.get('tickSize')),
                            'lot_size': float(contract_details.get('stepSize'))
                        }
        except Exception as e:
            print(f"Erro no get_market_info da Apex: {e}")

        # Retorna um dicionário vazio se não encontrar o ativo ou se ocorrer um erro
        return {}


    def get_current_prices(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        all_prices = []
        if not symbols:
            all_symbols_config = self._get_all_symbols_info()
            if not all_symbols_config: return []
            symbols_to_fetch = [f"{s.replace('-', '')}" for s in all_symbols_config.keys()]
        else:
            symbols_to_fetch = [f"{s}USDT" for s in symbols]

        for symbol_for_api in symbols_to_fetch:
            try:
                response = self.public_client.ticker_v3(symbol=symbol_for_api)
                if response and 'data' in response and response['data']:
                    all_prices.append(self._map_ticker_from_apex(response['data'][0]))
            except Exception as e:
                print(f"Erro ao buscar ticker para o símbolo {symbol_for_api} na Apex: {e}")
        return all_prices

    def get_orderbook(self, symbol: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        try:
            symbol_for_api = symbol.upper() + 'USDT'
            response = self.public_client.depth_v3(symbol=symbol_for_api, limit=limit)
            if response and 'data' in response:
                return self._map_orderbook_from_apex(response['data'])
            return None
        except Exception as e:
            print(f"Erro no get_orderbook da Apex: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        try:
            return self.client.delete_order_v3(id=str(order_id))
        except Exception as e:
            print(f"Erro no cancel_order da Apex: {e}")
            return {"success": False, "error": str(e)}

    def cancel_stop_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        return self.cancel_order(symbol, order_id)

    def update_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        try:
            initial_margin_rate = 1 / int(leverage)
            symbol_for_api = f"{symbol.upper()}-USDT"
            return self.client.set_initial_margin_rate_v3(
                symbol=symbol_for_api,  # Usa o símbolo corrigido
                initialMarginRate=str(initial_margin_rate)
            )
        except Exception as e:
            print(f"Erro no update_leverage da Apex: {e}")
            return {"success": False, "error": str(e)}

    def create_order(self, order_type: str, symbol: str, side: str, amount: str, price: str = None,
                     reduce_only: bool = False, stop_loss_price: str = None, take_profit_price: str = None,
                     tick_size: float = 0.01, **kwargs) -> Dict[str, Any]:

        def _process_order_creation_response(api_response):
            if not isinstance(api_response, dict):
                return {"success": False, "error": "Resposta inválida da API"}
            order_data = api_response.get('data', api_response)
            if isinstance(order_data, dict) and 'id' in order_data:
                return {"success": True, "data": order_data}
            return {"success": False, "error": api_response.get('msg', 'Erro desconhecido na criação da ordem')}

        params = {
            "symbol": symbol.replace('/', '-') + "-USDT",
            "side": side.upper(),
            "type": order_type.upper(),
            "size": amount,
            "reduceOnly": reduce_only
        }
        if price:
            params["price"] = price

        if stop_loss_price and take_profit_price:
            params["isOpenTpslOrder"] = True
            params["isSetOpenSl"] = True
            params["slTriggerPrice"] = stop_loss_price
            params["slSide"] = "SELL" if side.upper() == "BUY" else "BUY"
            params["slSize"] = amount
            sl_slippage = decimal.Decimal("0.1") if params["slSide"] == "BUY" else decimal.Decimal("-0.1")
            sl_price_with_slippage = decimal.Decimal(stop_loss_price) * (decimal.Decimal("1") + sl_slippage)
            # --- CORREÇÃO APLICADA AQUI ---
            params["slPrice"] = round_to_increment(float(sl_price_with_slippage), tick_size)

            params["isSetOpenTp"] = True
            params["tpTriggerPrice"] = take_profit_price
            params["tpSide"] = "SELL" if side.upper() == "BUY" else "BUY"
            params["tpSize"] = amount
            tp_slippage = decimal.Decimal("0.1") if params["tpSide"] == "BUY" else decimal.Decimal("-0.1")
            tp_price_with_slippage = decimal.Decimal(take_profit_price) * (decimal.Decimal("1") + tp_slippage)
            # --- CORREÇÃO APLICADA AQUI ---
            params["tpPrice"] = round_to_increment(float(tp_price_with_slippage), tick_size)

        try:
            api_response = self.client.create_order_v3(**params)
            return _process_order_creation_response(api_response)
        except Exception as e:
            print(f"Erro no create_order da Apex: {e}")
            return {"success": False, "error": str(e)}

    def set_position_tpsl(self, symbol: str, position_side: str, position_size: str, tick_size: float,
                          stop_loss_price: str = None, take_profit_price: str = None) -> Dict:
        responses = {}
        order_side = "SELL" if position_side.upper() in ["BUY", "BID"] else "BUY"

        def process_response(api_response):
            if not isinstance(api_response, dict):
                return {"success": False, "error": "Resposta inválida da API"}
            if 'id' in api_response:
                return {"success": True, "data": api_response}
            order_data = api_response.get('data')
            if isinstance(order_data, dict) and 'id' in order_data:
                return {"success": True, "data": order_data}
            return {"success": False, "error": api_response.get('msg', 'Erro desconhecido da API')}

        # Lógica para criar as ordens (continua a mesma)
        if take_profit_price:
            try:
                tp_slippage = decimal.Decimal("-0.1") if order_side == "BUY" else decimal.Decimal("0.1")
                tp_price_with_slippage = decimal.Decimal(take_profit_price) * (decimal.Decimal("1") + tp_slippage)
                tp_price_rounded = round_to_increment(float(tp_price_with_slippage), tick_size)
                tp_api_response = self.client.create_order_v3(
                    symbol=symbol.replace('/', '-') + "-USDT", side=order_side, size=str(position_size),
                    price=tp_price_rounded, isPositionTpsl=True, reduceOnly=True,
                    triggerPrice=take_profit_price, triggerPriceType="INDEX", type="TAKE_PROFIT_MARKET"
                )
                responses['take_profit'] = process_response(tp_api_response)
            except Exception as e:
                responses['take_profit'] = {"success": False, "error": str(e)}

        if stop_loss_price:
            try:
                sl_slippage = decimal.Decimal("0.1") if order_side == "BUY" else decimal.Decimal("-0.1")
                sl_price_with_slippage = decimal.Decimal(stop_loss_price) * (decimal.Decimal("1") + sl_slippage)
                sl_price_rounded = round_to_increment(float(sl_price_with_slippage), tick_size)
                sl_api_response = self.client.create_order_v3(
                    symbol=symbol.replace('/', '-') + "-USDT", side=order_side, size=str(position_size),
                    price=sl_price_rounded, isPositionTpsl=True, reduceOnly=True,
                    triggerPrice=stop_loss_price, triggerPriceType="INDEX", type="STOP_MARKET"
                )
                responses['stop_loss'] = process_response(sl_api_response)
            except Exception as e:
                responses['stop_loss'] = {"success": False, "error": str(e)}

        # --- INÍCIO DA NOVA LÓGICA DE RETORNO ---
        # Consolida o resultado final em um único dicionário de status

        # Se a chamada foi apenas para o SL
        if stop_loss_price and not take_profit_price:
            return responses.get('stop_loss', {"success": False, "error": "Resposta de SL não encontrada"})

        # Se a chamada foi apenas para o TP
        if take_profit_price and not stop_loss_price:
            return responses.get('take_profit', {"success": False, "error": "Resposta de TP não encontrada"})

        # Se a chamada foi para ambos
        sl_success = responses.get('stop_loss', {}).get('success', False)
        tp_success = responses.get('take_profit', {}).get('success', False)

        if sl_success and tp_success:
            return {"success": True, "data": responses}
        else:
            # Retorna o primeiro erro que encontrar
            error_message = responses.get('stop_loss', {}).get('error') or responses.get('take_profit', {}).get('error')
            return {"success": False, "error": error_message or "Falha ao criar ordens de TP/SL"}
        # --- FIM DA NOVA LÓGICA DE RETORNO ---

    def create_limit_order(self, symbol: str, side: str, amount: str, price: str, reduce_only: bool = False,
                           stop_loss_price: str = None, take_profit_price: str = None, tick_size: float = 0.01) -> Dict[
        str, Any]:
        """Chama o metodo genérico 'create_order' com o tipo 'LIMIT' e todos os parâmetros."""
        return self.create_order(
            order_type="LIMIT",
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            reduce_only=reduce_only,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            tick_size=tick_size
        )

    # (Substitua esta função em apex_client.py)
    def create_market_order(self, symbol: str, side: str, amount: str, reduce_only: bool = False,
                            tick_size: float = 0.01, **kwargs) -> Dict[str, Any]:
        """Cria uma ordem a mercado. A Apex exige um 'price' para controle de derrapagem."""
        try:
            orderbook = self.get_orderbook(symbol.split('-')[0])
            if side.upper() == "BUY":
                price_raw = float(orderbook['l'][1][0]['p']) * 1.02  # Preço 5% acima do melhor ask
            else:  # SELL
                price_raw = float(orderbook['l'][0][0]['p']) * 0.98  # Preço 5% abaixo do melhor bid

            price = round_to_increment(price_raw, tick_size)

        except Exception:
            price = "0"  # Fallback em caso de erro

        return self.create_order(
            order_type="MARKET",
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            reduce_only=reduce_only,
            tick_size=tick_size
        )

    def update_margin_mode(self, symbol: str, is_isolated: bool) -> Dict[str, Any]:
        return {"success": True, "data": "Endpoint de modo de margem não aplicável para a Apex."}

    # (Substitua esta função dentro da classe ApexClient em apex_client.py)

    def get_pnl_history(self, start_time_ms: int, end_time_ms: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Busca o histórico de PNL (Lucro e Prejuízo) da conta, com suporte a paginação."""
        all_pnl_entries = []
        current_page = 0

        try:
            while True:
                response = self.client.historical_pnl_v3(
                    beginTimeInclusive=start_time_ms,
                    endTimeExclusive=end_time_ms,
                    limit=limit,
                    page=current_page  # Usa a página atual no loop
                )

                if (response and
                        isinstance(response.get('data'), dict) and
                        isinstance(response['data'].get('historicalPnl'), list)):

                    pnl_page_data = response['data']['historicalPnl']
                    all_pnl_entries.extend(pnl_page_data)
                    # Verifica se a página atual está vazia ou se já buscamos todos os registros
                    total_size = response['data'].get('totalSize', 0)
                    if not pnl_page_data or len(all_pnl_entries) >= total_size:
                        break

                    current_page += 1
                else:
                    break
            return all_pnl_entries

        except Exception as e:
            print(f"Erro no get_pnl_history da Apex: {e}")
            return []
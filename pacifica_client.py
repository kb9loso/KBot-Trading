# pacifica_client.py
import requests
import json
from typing import Dict, List, Any, Optional
import uuid
import time
import base58
from decimal import Decimal, ROUND_HALF_UP

from solders.keypair import Keypair

import ccxt
import pandas as pd


def _sort_json_keys(value):
    """Ordena recursivamente as chaves de um JSON para garantir consistência."""
    if isinstance(value, dict):
        return {key: _sort_json_keys(value[key]) for key in sorted(value.keys())}
    elif isinstance(value, list):
        return [_sort_json_keys(item) for item in value]
    else:
        return value


def _sign_message(header, payload, keypair):
    """Assina mensagem compacta com JSON ordenado recursivamente."""
    if not all(k in header for k in ("type", "timestamp", "expiry_window")):
        raise ValueError("Header deve conter type, timestamp e expiry_window")

    data = {**header, "data": payload}
    message = _sort_json_keys(data)

    message_str = json.dumps(message, separators=(",", ":")).encode("utf-8")
    signature = keypair.sign_message(message_str)

    return base58.b58encode(bytes(signature)).decode("ascii")

def round_to_increment(value: float, increment: float) -> str:
    """Arredonda um valor para o múltiplo mais próximo de um incremento (tick/lot size)."""
    if increment <= 0: return f"{value:.8f}"
    value_dec = Decimal(str(value))
    increment_dec = Decimal(str(increment))
    rounded_value = (value_dec / increment_dec).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * increment_dec
    return "{:f}".format(rounded_value.normalize())


class PacificaClient:
    BASE_URL = "https://api.pacifica.fi"

    def __init__(self, main_public_key: str, agent_private_key: str):
        self.main_public_key = main_public_key
        try:
            self.agent_keypair = Keypair.from_base58_string(agent_private_key)
            self.agent_public_key = str(self.agent_keypair.pubkey())
            print(f"Cliente iniciado para a conta principal: {self.main_public_key}")
            print(f"Usando agente de API: {self.agent_public_key}")
        except Exception as e:
            raise ValueError(f"Chave privada do agente inválida. Erro: {e}")

    def _sign_message_and_build_payload(self, signature_header: Dict, signature_payload: Dict) -> Dict:
        """
        Constrói a mensagem a ser assinada, assina com a chave do agente,
        e monta o payload final da requisição.
        """
        message_to_sign_str = json.dumps({**signature_header, "data": signature_payload}, sort_keys=True,
                                         separators=(',', ':'))
        signature_bytes = self.agent_keypair.sign_message(message_to_sign_str.encode("utf-8"))
        signature_str = base58.b58encode(bytes(signature_bytes)).decode("utf-8")

        return {
            "account": self.main_public_key,
            "agent_wallet": self.agent_public_key,
            "signature": signature_str,
            "timestamp": signature_header["timestamp"],
            "expiry_window": signature_header["expiry_window"],
            **signature_payload
        }

    def _make_request(self, method: str, path: str, params: Dict = None, data: Dict = None, headers: Dict = None):
        """Função genérica para fazer requisições, agora com retorno de erro detalhado."""
        if headers is None: headers = {'Content-Type': 'application/json'}
        url = self.BASE_URL + path
        try:
            response = requests.request(method, url, headers=headers, params=params, json=data, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Erro de requisição para {url}: {e}")
            if e.response is not None:
                print(f"Detalhes do erro (raw): {e.response.text}")
                try:
                    return e.response.json()
                except json.JSONDecodeError:
                    return {"success": False, "error": f"Erro {e.response.status_code}", "data": e.response.text}
            return {"success": False, "error": "Erro de conexão", "data": str(e)}

    @staticmethod
    def generate_base58_private_key():
        """Gera uma nova keypair e retorna a chave privada em formato base58."""
        kp = Keypair()
        secret_64 = bytes(kp)
        return base58.b58encode(secret_64).decode("ascii")

    def create_subaccount(self, new_sub_private_key_b58: str) -> Dict:
        path = '/api/v1/account/subaccount/create'
        sub_keypair = Keypair.from_base58_string(new_sub_private_key_b58)
        sub_public_key = str(sub_keypair.pubkey())
        timestamp = int(time.time() * 1000)
        expiry_window = 5000
        sub_header = {"type": "subaccount_initiate", "timestamp": timestamp, "expiry_window": expiry_window}
        sub_payload = {"account": self.main_public_key}
        sub_signature = _sign_message(sub_header, sub_payload, sub_keypair)
        main_header = {"type": "subaccount_confirm", "timestamp": timestamp, "expiry_window": expiry_window}
        main_payload = {"signature": sub_signature}
        main_signature = _sign_message(main_header, main_payload, self.agent_keypair)
        request_body = {
            "main_account": self.main_public_key, "subaccount": sub_public_key,
            "main_signature": main_signature, "sub_signature": sub_signature,
            "timestamp": timestamp, "expiry_window": expiry_window,
        }
        print(f"Enviando requisição para criar subconta para {sub_public_key}...")
        api_response = self._make_request("POST", path, data=request_body)
        if api_response and api_response.get('success'):
            return {"success": True, "subaccount_public_key": request_body['subaccount'],
                    "sub_signature": request_body['sub_signature']}
        else:
            return {"success": False, "error": api_response.get('error', 'Erro desconhecido ao criar subconta')}

    def update_leverage(self, symbol: str, leverage: int) -> Dict:
        path = '/api/v1/account/leverage'
        timestamp = int(time.time() * 1000)
        signature_header = {"type": "update_leverage", "timestamp": timestamp, "expiry_window": 30000}
        signature_payload = {"symbol": symbol, "leverage": leverage}
        final_payload = self._sign_message_and_build_payload(signature_header, signature_payload)
        print(f"ATUALIZANDO ALAVANCAGEM para {symbol} para {leverage}x...")
        return self._make_request('POST', path, data=final_payload)

    def update_margin_mode(self, symbol: str, is_isolated: bool) -> Dict:
        path = '/api/v1/account/margin'
        timestamp = int(time.time() * 1000)
        signature_header = {"type": "update_margin_mode", "timestamp": timestamp, "expiry_window": 30000}
        signature_payload = {"symbol": symbol, "is_isolated": is_isolated}
        final_payload = self._sign_message_and_build_payload(signature_header, signature_payload)
        mode = "ISOLATED" if is_isolated else "CROSS"
        print(f"ATUALIZANDO MODO DE MARGEM para {symbol} para {mode}...")
        return self._make_request('POST', path, data=final_payload)

    def create_market_order(self, symbol: str, side: str, amount: str, slippage: float,
                            take_profit_price: str, stop_loss_price: str, tick_size: float) -> Dict:
        path = '/api/v1/orders/create_market'
        timestamp = int(time.time() * 1000)
        signature_header = {"type": "create_market_order", "timestamp": timestamp, "expiry_window": 30000}

        is_buy_side = side.upper() == "BUY"

        sl_price_float = float(stop_loss_price)
        tp_price_float = float(take_profit_price)

        if is_buy_side:  # Posição LONG, ordens SL/TP são de VENDA
            sl_limit_price = sl_price_float * 0.99
            tp_limit_price = tp_price_float * 0.99
        else:  # Posição SHORT, ordens SL/TP são de COMPRA
            sl_limit_price = sl_price_float * 1.01
            tp_limit_price = tp_price_float * 1.01

        sl_limit_price_rounded = round_to_increment(sl_limit_price, tick_size)
        tp_limit_price_rounded = round_to_increment(tp_limit_price, tick_size)

        signature_payload = {
            "symbol": symbol, "amount": amount, "side": "bid" if is_buy_side else "ask",
            "slippage_percent": str(slippage), "reduce_only": False,
            "take_profit": {
                "stop_price": take_profit_price,
                "limit_price": tp_limit_price_rounded
            },
            "stop_loss": {
                "stop_price": stop_loss_price,
                "limit_price": sl_limit_price_rounded
            }
        }
        final_payload = self._sign_message_and_build_payload(signature_header, signature_payload)
        print(f"CRIANDO ORDEM ASSINADA PELO AGENTE para {symbol}...")
        return self._make_request('POST', path, data=final_payload)

    def close_market_order(self, symbol: str, side: str, amount: str) -> Dict:
        path = '/api/v1/orders/create_market'
        timestamp = int(time.time() * 1000)
        signature_header = {"type": "create_market_order", "timestamp": timestamp, "expiry_window": 30000}
        signature_payload = {
            "symbol": symbol, "amount": amount, "side": "bid" if side.upper() == "BUY" else "ask",
            "slippage_percent": "1.0", "reduce_only": True
        }
        final_payload = self._sign_message_and_build_payload(signature_header, signature_payload)
        print(f"CRIANDO ORDEM DE FECHAMENTO para {symbol}...")
        return self._make_request('POST', path, data=final_payload)

    def get_account_info(self) -> Dict:
        path = '/api/v1/account'
        params = {'account': self.main_public_key}
        response = self._make_request('GET', path, params=params)
        return response.get('data') if response else None

    def get_account_settings(self) -> List[Dict]:
        path = '/api/v1/account/settings'
        params = {'account': self.main_public_key}
        response_data = self._make_request('GET', path, params=params)
        if response_data and isinstance(response_data.get('data'), list):
            return response_data['data']
        return []

    def get_open_positions(self, market: Optional[str] = None) -> List[Dict]:
        path = '/api/v1/positions'
        params = {'account': self.main_public_key}
        response_data = self._make_request('GET', path, params=params)
        if response_data and isinstance(response_data.get('data'), list):
            positions_list = response_data['data']
            all_open_positions = [pos for pos in positions_list if 'amount' in pos and float(pos['amount']) > 0]
            if market:
                market_base_symbol = market.split('/')[0].upper()
                return [pos for pos in all_open_positions if
                        'symbol' in pos and pos['symbol'].upper() == market_base_symbol]
            return all_open_positions
        return []

    def get_trade_history(self, start_time_ms: int, end_time_ms: int) -> List[Dict]:
        path = '/api/v1/positions/history'
        params = {'account': self.main_public_key, 'start_time': start_time_ms, 'end_time': end_time_ms, 'limit': 1000}
        response_data = self._make_request('GET', path, params=params)
        if response_data and 'data' in response_data and isinstance(response_data['data'], list):
            return response_data['data']
        if response_data and isinstance(response_data, list):
            return response_data
        return []

    def get_market_info(self, symbol: str) -> Dict:
        path = '/api/v1/info'
        try:
            markets_data = self._make_request('GET', path)
            data = markets_data.get('data', markets_data) if isinstance(markets_data, dict) else markets_data
            if isinstance(data, list):
                for market in data:
                    if market.get('symbol') == symbol:
                        return market
            return {}
        except Exception as e:
            print(f"Erro ao buscar informações do mercado: {e}")
            return {}

    def get_current_prices(self) -> List[Dict]:
        path = '/api/v1/info/prices'
        response_data = self._make_request('GET', path)
        if response_data:
            data = response_data.get('data')
            if isinstance(data, list) and all(isinstance(x, dict) for x in data):
                return data
        return []

    def get_open_orders(self) -> List[Dict]:
        print("Buscando todas as ordens abertas...")
        path = '/api/v1/orders'
        params = {'account': self.main_public_key}
        response_data = self._make_request('GET', path, params=params)
        if response_data and isinstance(response_data, list):
            return response_data
        if response_data and isinstance(response_data.get('data'), list):
            return response_data['data']
        return []

    def cancel_stop_order(self, symbol: str, order_id: int) -> Dict:
        path = '/api/v1/orders/stop/cancel'
        timestamp = int(time.time() * 1000)
        signature_header = {"type": "cancel_stop_order", "timestamp": timestamp, "expiry_window": 30000}
        signature_payload = {"symbol": symbol, "order_id": order_id}
        final_payload = self._sign_message_and_build_payload(signature_header, signature_payload)
        print(f"CANCELANDO ORDEM {order_id} para {symbol}...")
        return self._make_request('POST', path, data=final_payload)

    def set_position_tpsl(self, symbol: str, side: str, tick_size: float, stop_loss_price: str = None,
                          take_profit_price: str = None) -> Dict:
        path = '/api/v1/positions/tpsl'
        timestamp = int(time.time() * 1000)
        signature_header = {"type": "set_position_tpsl", "timestamp": timestamp, "expiry_window": 30000}

        is_position_long = side == "bid"
        closing_side = "ask" if is_position_long else "bid"

        signature_payload = {"symbol": symbol, "side": closing_side}

        if stop_loss_price:
            sl_price_float = float(stop_loss_price)
            sl_limit_price = sl_price_float * 0.99 if is_position_long else sl_price_float * 1.01
            signature_payload["stop_loss"] = {
                "stop_price": stop_loss_price,
                "limit_price": round_to_increment(sl_limit_price, tick_size),
                "client_order_id": str(uuid.uuid4())
            }
        if take_profit_price:
            tp_price_float = float(take_profit_price)
            tp_limit_price = tp_price_float * 0.99 if is_position_long else tp_price_float * 1.01
            signature_payload["take_profit"] = {
                "stop_price": take_profit_price,
                "limit_price": round_to_increment(tp_limit_price, tick_size),
                "client_order_id": str(uuid.uuid4())
            }

        final_payload = self._sign_message_and_build_payload(signature_header, signature_payload)
        print(f"ATUALIZANDO TP/SL para {symbol}...")
        return self._make_request('POST', path, data=final_payload)

    def create_stop_order(self, symbol: str, side: str, amount: str, stop_price: str, tick_size: float) -> Dict:
        path = '/api/v1/orders/stop/create'
        timestamp = int(time.time() * 1000)
        signature_header = {"type": "create_stop_order", "timestamp": timestamp, "expiry_window": 30000}

        is_buy_side = side.upper() == "BUY"
        stop_price_float = float(stop_price)
        limit_price = stop_price_float * 1.01 if is_buy_side else stop_price_float * 0.99

        signature_payload = {
            "symbol": symbol, "side": "bid" if is_buy_side else "ask",
            "reduce_only": True, "amount": amount,
            "stop_order": {
                "stop_price": stop_price,
                "limit_price": round_to_increment(limit_price, tick_size)
            }
        }
        final_payload = self._sign_message_and_build_payload(signature_header, signature_payload)
        print(f"CRIANDO NOVA ORDEM STOP para {symbol} em ${stop_price}...")
        return self._make_request('POST', path, data=final_payload)

    def get_historical_klines(self, market: str, timeframe: str, limit: int = 200,
                              exchange_name: str = 'binance') -> Any:
        try:
            exchange = getattr(ccxt, exchange_name)()
            klines = exchange.fetch_ohlcv(market, timeframe, limit=limit)
            if not klines:
                print(f"Nenhum dado de klines retornado por {exchange_name} para {market}.")
                return None
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            return None
# KBot-Trading/exchange_interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseExchangeClient(ABC):
    """
    Classe base abstrata (Interface) que define os métodos essenciais
    que qualquer cliente de exchange deve implementar para ser compatível com o bot.
    """

    @abstractmethod
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Busca as informações gerais da conta (saldo, equity, etc.)."""
        pass

    @abstractmethod
    def get_open_positions(self, market: str = None) -> List[Dict[str, Any]]:
        """Busca todas as posições abertas ou para um mercado específico."""
        pass

    @abstractmethod
    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """Busca detalhes de um mercado (tick size, lot size, etc.)."""
        pass

    @abstractmethod
    def get_orderbook(self, symbol: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """Busca o livro de ordens para um símbolo."""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Busca todas as ordens que estão atualmente abertas."""
        pass

    @abstractmethod
    def create_limit_order(self, symbol: str, side: str, amount: str, price: str, reduce_only: bool = False,
                           take_profit_price: str = None, stop_loss_price: str = None, tick_size: float = 0.01) -> Dict[
        str, Any]:
        """Cria uma nova ordem a limite com parâmetros de TP/SL."""
        pass

    @abstractmethod
    def create_market_order(self, symbol: str, side: str, amount: str, reduce_only: bool = False,
                            tick_size: float = 0.01) -> Dict[str, Any]:
        """Cria uma nova ordem a mercado."""
        pass

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancela uma ordem aberta pelo seu ID."""
        pass

    @abstractmethod
    def cancel_stop_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancela uma ordem de stop (SL/TP) pelo seu ID."""
        pass

    @abstractmethod
    def update_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Atualiza a alavancagem para um determinado símbolo."""
        pass

    @abstractmethod
    def update_margin_mode(self, symbol: str, is_isolated: bool) -> Dict[str, Any]:
        """Atualiza o modo de margem para um símbolo (isolada ou cruzada)."""
        pass

    @abstractmethod
    def get_trade_history(self, start_time_ms: int, end_time_ms: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Busca o histórico de trades (preenchimentos) dentro de um período."""
        pass

    @abstractmethod
    def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Busca o histórico de todas as ordens (abertas, fechadas, canceladas)."""
        pass

    @abstractmethod
    def set_position_tpsl(self, symbol: str, position_side: str, position_size: str, tick_size: float,
                          stop_loss_price: str = None, take_profit_price: str = None) -> Dict[str, Any]:
        """
        Define ou atualiza o Stop Loss e/ou Take Profit para uma posição existente.
        """
        pass

    @abstractmethod
    def get_current_prices(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """
        Busca os preços atuais (tickers). Se uma lista de símbolos for fornecida,
        busca apenas os preços para esses símbolos.
        """
        pass

    @abstractmethod
    def get_pnl_history(self, start_time_ms: int, end_time_ms: int, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Busca o histórico de PNL (Lucro e Prejuízo) realizado.
        Específico para exchanges que fornecem este endpoint.
        """
        pass
# KBot-Trading/exchange_factory.py
from typing import Dict, Any

# Importa a classe base do novo arquivo de interface
from exchange_interface import BaseExchangeClient

# Importa as classes de cliente específicas
from pacifica_client import PacificaClient
from apex_client import ApexClient

def get_client(exchange_name: str, account_config: Dict[str, Any]) -> BaseExchangeClient:
    """
    Factory function que instancia e retorna o cliente de exchange apropriado.

    :param exchange_name: O nome da exchange (ex: 'pacifica', 'apex').
    :param account_config: O dicionário de configuração da conta contendo as credenciais.
    :return: Uma instância de um cliente de exchange que herda de BaseExchangeClient.
    """
    if exchange_name.lower() == 'pacifica':
        return PacificaClient(
            main_public_key=account_config['main_public_key'],
            agent_private_key=account_config['agent_private_key']
        )
    elif exchange_name.lower() == 'apex':
        return ApexClient(
            api_key=account_config['api_key'],
            api_secret=account_config['api_secret'],
            passphrase=account_config['passphrase'],
            zk_seeds=account_config['zk_seeds'],
            zk_l2key=account_config['zk_l2key']
        )
    else:
        raise ValueError(f"A exchange '{exchange_name}' não é suportada.")
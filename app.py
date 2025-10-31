import subprocess
import numpy as np
import requests
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_socketio import SocketIO, emit
from collections import deque
import json
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import sys
from logging.handlers import RotatingFileHandler

from bot_logic import TradingBot
from strategies import STRATEGIES
from exchange_factory import get_client
from backtester import run_full_backtest
from database import init_db, sync_order_history, sync_trade_history, sync_pnl_history, query_daily_pnl_all_accounts
from notifications import check_and_send_close_alerts, check_and_send_open_alerts, check_and_send_close_alerts_apex

import nest_asyncio
from websocket_manager import WebSocketManager

app = Flask(__name__)
app.secret_key = 'supersecretkey'
socketio = SocketIO(app, async_mode='threading')
nest_asyncio.apply()
ws_manager = WebSocketManager()  # Instância global mantida

populate_lock = threading.Lock()
# Set para rastrear quais (symbol, timeframe) já tiveram o histórico carregado nesta sessão
populated_caches = set()

LOG_HISTORY_LIMIT = 500
log_history = deque(maxlen=LOG_HISTORY_LIMIT)  # Armazena os últimos N logs

logging.SUCCESS = 25
logging.addLevelName(logging.SUCCESS, 'SUCCESS')
logging.EXECUTE = 26
logging.addLevelName(logging.EXECUTE, 'EXECUTE')


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_object = {
            'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
            'account': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            'is_html': getattr(record, 'is_html', False)
        }
        if record.exc_info:
            log_object['message'] += '\n' + self.formatException(record.exc_info)
        return log_object


class SocketIOHandler(logging.Handler):
    def emit(self, record):
        try:
            log_entry = self.format(record)
            # --- INÍCIO DA MODIFICAÇÃO (Adiciona ao histórico e emite) ---
            log_history.append(log_entry)  # Adiciona ao deque (automático remove o mais antigo se cheio)
            socketio.emit('new_log', log_entry)
            # --- FIM DA MODIFICAÇÃO ---
        except Exception:
            self.handleError(record)


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Previne handlers duplicados se a função for chamada mais de uma vez
    if logger.hasHandlers():
        logger.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S'))

    socket_handler = SocketIOHandler()
    socket_handler.setFormatter(JSONFormatter())

    file_handler = RotatingFileHandler('kbot_logs.txt', maxBytes=1024 * 1024 * 5, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(
        logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S'))

    logger.addHandler(stdout_handler)
    logger.addHandler(socket_handler)
    logger.addHandler(file_handler)

    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('socketio').setLevel(logging.WARNING)
    logging.getLogger('engineio').setLevel(logging.WARNING)

    def execute(self, message, *args, **kws):
        if self.isEnabledFor(logging.EXECUTE):
            self._log(logging.EXECUTE, message, args, **kws)

    logging.Logger.execute = execute


log = logging.getLogger('Sistema')


# MODIFICAÇÃO: Garantir que o BTC/1h seja carregado se o filtro estiver em uso
def get_active_symbol_timeframes(config: dict) -> dict[tuple[str, str], int]:
    """
    Extrai todos os pares (symbol, timeframe) únicos e o mínimo
    de candles necessários para cada um, dos setups ativos.

    Retorna:
        dict: {(ws_symbol, timeframe): min_candles, ...}
    """
    active_pairs_map = {}
    btc_trend_in_use = False  # <<< ADICIONADO

    exchanges_config = config.get('exchanges', {})
    for ex_data in exchanges_config.values():
        for account in ex_data.get('accounts', []):
            for setup in account.get('markets_to_trade', []):
                symbol = setup.get('base_currency')
                strategy_name = setup.get('strategy_name')

                # <<< ADICIONADO: Verifica se o filtro BTC está em uso
                if setup.get('direction_mode') == 'btc_trend':
                    btc_trend_in_use = True

                if symbol and strategy_name and strategy_name in STRATEGIES:
                    strategy_config = STRATEGIES[strategy_name]
                    timeframe = strategy_config.get('timeframe')
                    min_candles_strategy = strategy_config.get('min_candles', 200)

                    if timeframe and timeframe in WebSocketManager.SUPPORTED_TIMEFRAMES:
                        ccxt_symbol = f"{symbol.upper()}/USDT"
                        ws_symbol = ccxt_symbol.replace('/', '-')

                        cache_key = (ws_symbol, timeframe)

                        current_min = active_pairs_map.get(cache_key, 0)
                        active_pairs_map[cache_key] = max(current_min, min_candles_strategy)

                    elif timeframe:
                        log.warning(
                            f"Timeframe '{timeframe}' da estratégia '{strategy_name}' não é suportado pelo WebSocketManager.")
                elif not symbol:
                    log.warning(f"Setup na conta {account.get('account_name')} sem 'base_currency'.")
                elif not strategy_name:
                    log.warning(f"Setup para {symbol} na conta {account.get('account_name')} sem 'strategy_name'.")
                elif strategy_name not in STRATEGIES:
                    log.warning(f"Estratégia '{strategy_name}' não encontrada no arquivo strategies.py.")

    # <<< ADICIONADO: Força a inclusão do BTC/1h se o filtro estiver em uso
    if btc_trend_in_use:
        log.info("Filtro de tendência BTC em uso. Adicionando 'BTC-USDT'/'1h' ao WS Manager.")
        btc_key = ('BTC-USDT', '1h')
        current_min_btc = active_pairs_map.get(btc_key, 0)
        # O filtro de tendência (get_btc_trend_filter) precisa de pelo menos 50 velas de 1h
        active_pairs_map[btc_key] = max(current_min_btc, 100)  # 100 por segurança

    log.info(f"Pares/Timeframes ativos identificados para WebSocket: {active_pairs_map or 'Nenhum'}")
    return active_pairs_map


@socketio.on('connect')
def handle_connect():
    """Envia o histórico de logs para o cliente que acabou de conectar."""
    log.debug(f"Cliente conectado: {request.sid}. Enviando histórico de logs.")
    emit('initial_logs', list(log_history))  # Converte deque para lista antes de enviar


@socketio.on('disconnect')
def handle_disconnect():
    log.debug(f"Cliente desconectado: {request.sid}")


# --- FIM DA MODIFICAÇÃO ---


# Variáveis globais (sem bot_logs)
account_info_cache = {}
dashboard_metrics_cache = {}
dashboard_metrics_timestamp = {}  # Timestamp do último cálculo das métricas históricas
trade_history_30d_cache = {}  # <<< NOVO: Cache de trades dos últimos 30 dias (bruto)
trade_history_30d_timestamp = {}  # <<< NOVO: Timestamp do cache de trades
open_positions_cache = {}
weekly_volume_cache = {}
last_refresh_error = {}
backtest_result_cache = {}
running_bots = {}


def get_version_info():
    """Busca a versão local (commit hash) e compara com a remota no GitHub."""
    try:
        git_dir = subprocess.run(['git', 'rev-parse', '--git-dir'], capture_output=True, text=True, check=True)
        if ".git" not in git_dir.stdout:
            raise FileNotFoundError("Diretório .git não encontrado.")
        local_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        repo_url = "https://api.github.com/repos/kb9loso/KBot-Trading/commits/main"
        response = requests.get(repo_url, timeout=5)
        response.raise_for_status()
        remote_hash = response.json()['sha']
        return {
            "local_short": local_hash[:7],
            "is_latest": local_hash == remote_hash,
            "repo_url": "https://github.com/kb9loso/KBot-Trading"
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"error": "Não é um repositório Git."}
    except Exception as e:
        log.warning(f"Falha ao verificar versão: {e}")  # Loga o erro
        return {"error": f"Falha ao verificar versão: {e}"}


def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        log.error("Arquivo config.json não encontrado!")
        return {}  # Retorna um dict vazio para evitar erros
    except json.JSONDecodeError:
        log.error("Erro ao decodificar config.json! Verifique a sintaxe.")
        return {}


def save_config(config):
    try:
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        log.error(f"Erro ao salvar config.json: {e}")


def sync_all_trade_histories(start_time_ms: int):
    """Função auxiliar para sincronizar o histórico de trades de TODAS as contas."""
    log.info(f"AGENDADOR: Iniciando sincronização de TRADES")
    config = load_config()
    for exchange_name, exchange_data in config.get('exchanges', {}).items():
        for account_config in exchange_data.get('accounts', []):
            try:
                client = get_client(exchange_name, account_config)
                sync_trade_history(client, exchange_name, account_config['account_name'], start_time_ms)
            except Exception as e:
                logging.getLogger(account_config['account_name']).error(
                    f"ERRO na sincronização agendada de trades: {e}")


def sync_apex_pnl_task():
    """Tarefa agendada para sincronizar o PNL histórico de todas as contas Apex."""
    log.info(f"AGENDADOR: Iniciando sincronização de PNL da Apex")
    config = load_config()
    apex_exchange_data = config.get('exchanges', {}).get('apex', {})
    if not apex_exchange_data.get('accounts'):
        return
    start_time_ms = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
    for account_config in apex_exchange_data.get('accounts', []):
        try:
            client = get_client('apex', account_config)
            sync_pnl_history(client, account_config['account_name'], start_time_ms)
        except Exception as e:
            logging.getLogger(account_config['account_name']).error(f"ERRO na sincronização agendada de PNL: {e}")


def hourly_alert_task():
    """Tarefa principal que roda de hora em hora."""
    log.info("AGENDADOR: Iniciando tarefas de notificação por hora.")
    config = load_config()
    if not config.get('telegram', {}).get('enabled', False):
        log.debug("Agendador: Notificações Telegram desabilitadas, pulando tarefa horária.")
        return
    start_time_ms = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
    sync_all_trade_histories(start_time_ms)
    check_and_send_open_alerts()
    check_and_send_close_alerts()
    check_and_send_close_alerts_apex()


def full_daily_sync_task():
    log.info(f"AGENDADOR DIÁRIO: Iniciando tarefa de sincronização COMPLETA")
    config = load_config()
    daily_start_time_ms = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
    for exchange_name, exchange_data in config.get('exchanges', {}).items():
        for account_config in exchange_data.get('accounts', []):
            account_logger = logging.getLogger(account_config['account_name'])
            try:
                account_logger.info(f"Sincronização diária completa INICIADA.")
                client = get_client(exchange_name, account_config)
                # sync_order_history(client, exchange_name, account_config['account_name'])
                sync_trade_history(client, exchange_name, account_config['account_name'], daily_start_time_ms)
                # Força o recálculo do dashboard após a sync completa
                global dashboard_metrics_timestamp
                dashboard_metrics_timestamp[account_config['account_name']] = 0
            except Exception as e:
                account_logger.error(f"ERRO durante a sincronização diária: {e}")
    log.info("AGENDADOR DIÁRIO: Tarefa de sincronização completa concluída.")


scheduler = BackgroundScheduler(daemon=True)
# Adiciona um ID para evitar jobs duplicados em reloads
scheduler.add_job(hourly_alert_task, 'interval', hours=1, next_run_time=datetime.now() + timedelta(seconds=30),
                  id='hourly_alerts')
scheduler.add_job(full_daily_sync_task, 'interval', hours=24, next_run_time=datetime.now() + timedelta(minutes=5),
                  id='daily_sync')
scheduler.add_job(sync_apex_pnl_task, 'interval', hours=1, next_run_time=datetime.now() + timedelta(minutes=1),
                  id='apex_pnl_sync')

try:
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
except Exception as e:
    log.critical(f"Falha ao iniciar o agendador: {e}", exc_info=True)


@app.route('/manage_account', methods=['POST'])
def manage_account():
    config = load_config()
    exchanges_config = config.get('exchanges', {})
    selected_exchange = request.form['exchange_name']
    exchange_details = exchanges_config.get(selected_exchange, {})

    if not exchange_details.get('multi_account_allowed', False) and len(exchange_details.get('accounts', [])) > 0:
        original_name = request.form.get('original_account_name')
        if not original_name or exchange_details['accounts'][0]['account_name'] != original_name:
            flash(f"A exchange '{selected_exchange}' não permite múltiplas contas.", "error")
            return redirect(url_for('index'))

    interval = int(request.form.get('check_interval', 180))
    account_details = {
        "account_name": request.form['account_name'],
        "check_interval_seconds": max(interval, 120)
    }

    if selected_exchange == 'pacifica':
        account_details["main_public_key"] = request.form.get('main_public_key')
        account_details["agent_private_key"] = request.form.get('agent_private_key')
    elif selected_exchange == 'apex':
        account_details["api_key"] = request.form.get('api_key')
        account_details["api_secret"] = request.form.get('api_secret')
        account_details["passphrase"] = request.form.get('passphrase')
        account_details["zk_seeds"] = request.form.get('zk_seeds')
        account_details["zk_l2key"] = request.form.get('zk_l2key')

    original_name = request.form.get('original_account_name')
    accounts_in_exchange = exchange_details.get('accounts', [])

    if original_name:  # Editando
        for i, acc in enumerate(accounts_in_exchange):
            if acc['account_name'] == original_name:
                # Mantém credenciais secretas se não forem fornecidas novamente
                if selected_exchange == 'pacifica' and not account_details.get("agent_private_key"):
                    account_details["agent_private_key"] = acc.get("agent_private_key")
                elif selected_exchange == 'apex':
                    if not account_details.get("api_secret"): account_details["api_secret"] = acc.get("api_secret")
                    if not account_details.get("passphrase"): account_details["passphrase"] = acc.get("passphrase")
                    if not account_details.get("zk_seeds"): account_details["zk_seeds"] = acc.get("zk_seeds")
                    if not account_details.get("zk_l2key"): account_details["zk_l2key"] = acc.get("zk_l2key")

                # Preserva configurações que não estão no formulário
                account_details['markets_to_trade'] = acc.get('markets_to_trade', [])
                account_details['notifications_enabled'] = acc.get('notifications_enabled', False)
                accounts_in_exchange[i] = account_details
                log.info(f"Conta '{original_name}' atualizada para '{account_details['account_name']}'.")
                break
    else:  # Adicionando
        account_details['markets_to_trade'] = []
        account_details['notifications_enabled'] = False
        accounts_in_exchange.append(account_details)
        log.info(f"Conta '{account_details['account_name']}' adicionada.")

    exchanges_config.setdefault(selected_exchange, {})['accounts'] = accounts_in_exchange
    config['exchanges'] = exchanges_config
    save_config(config)
    return redirect(url_for('index'))


@app.route('/delete_account/<exchange_name>/<account_name>', methods=['POST'])
def delete_account(exchange_name, account_name):
    config = load_config()
    if exchange_name in config.get('exchanges', {}):
        config['exchanges'][exchange_name]['accounts'] = [
            acc for acc in config['exchanges'][exchange_name].get('accounts', [])
            if acc.get('account_name') != account_name
        ]
        save_config(config)
        log.warning(f"Conta '{account_name}' deletada.")
    else:
        log.warning(f"Tentativa de deletar conta '{account_name}' da exchange inexistente '{exchange_name}'.")
    return redirect(url_for('index'))


def calculate_dashboard_metrics(trades: list) -> dict:
    default_metrics = {"total_trades": 0, "total_pnl": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0, "total_fees": 0}
    if not trades:
        return default_metrics
    try:
        df = pd.DataFrame(trades)
        if df.empty: return default_metrics

        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
        df['fee'] = pd.to_numeric(df['fee'], errors='coerce').fillna(0)

        # Garante que 'side' existe e é string
        if 'side' not in df.columns: return default_metrics
        df['side'] = df['side'].astype(str)

        closed_trades_df = df[df['side'].str.startswith('close_', na=False)].copy()
        if closed_trades_df.empty:
            return default_metrics

        if 'order_id' in closed_trades_df.columns:
            operations_df = closed_trades_df.groupby('order_id').agg(
                pnl=('pnl', 'sum'),
                fee=('fee', 'sum')
            ).reset_index()
        else:  # Se não houver order_id, trata cada trade como uma operação
            operations_df = closed_trades_df[['pnl', 'fee']]

        if operations_df.empty:
            return default_metrics

        winning_trades_pnl = operations_df[operations_df['pnl'] > 0]['pnl'].tolist()
        losing_trades_pnl = operations_df[operations_df['pnl'] <= 0]['pnl'].tolist()

        total_fees = operations_df['fee'].sum()
        total_trades = len(operations_df)
        total_pnl = operations_df['pnl'].sum()
        win_rate = (len(winning_trades_pnl) / total_trades) * 100 if total_trades > 0 else 0
        avg_win = sum(winning_trades_pnl) / len(winning_trades_pnl) if winning_trades_pnl else 0
        avg_loss = sum(losing_trades_pnl) / len(losing_trades_pnl) if losing_trades_pnl else 0

        return {
            "total_pnl": total_pnl, "total_trades": total_trades, "win_rate": win_rate,
            "avg_win": avg_win, "avg_loss": avg_loss, "total_fees": total_fees
        }
    except Exception as e:
        log.error(f"Erro ao calcular métricas do dashboard: {e}", exc_info=True)
        return default_metrics


def _adapt_pnl_history_for_dashboard(pnl_history: list) -> list:
    """Adapta o formato do histórico de PNL da Apex para o formato esperado pela função de cálculo de métricas."""
    adapted_trades = []
    for pnl_entry in pnl_history:
        adapted_trades.append({
            'pnl': float(pnl_entry.get('totalPnl') or 0.0),
            'side': 'close_' + pnl_entry.get('side').lower(),
            'fee': float(pnl_entry.get('fee') or 0.0),
            'order_id': pnl_entry.get('id')  # Usa tradeId como agrupador se disponível
        })
    return adapted_trades


def fetch_and_cache_api_data(account_config: dict, exchange_name: str):
    """Função genérica para buscar dados da API usando a factory. Otimizada para evitar buscas históricas pesadas."""
    global account_info_cache, dashboard_metrics_cache, open_positions_cache, last_refresh_error, weekly_volume_cache, dashboard_metrics_timestamp, trade_history_30d_cache, trade_history_30d_timestamp
    account_name = account_config['account_name']
    account_logger = logging.getLogger(account_name)
    account_logger.debug("Iniciando busca de dados da API...")
    leverage_map = {
        market['base_currency'].upper(): market.get('leverage', 1)
        for market in account_config.get('markets_to_trade', [])
    }
    try:
        client = get_client(exchange_name, account_config)
        account_info = client.get_account_info()
        if not account_info:
            raise Exception("Falha ao buscar informações da conta (retorno vazio).")
        account_info_cache[account_name] = account_info
        account_logger.debug("Informações da conta obtidas.")

        current_time_s = int(time.time())
        CACHE_FRESHNESS_S = 3600  # 1 hora

        # --- CÁLCULO DE TEMPOS (30d) ---
        start_dt_30d = datetime.now() - timedelta(days=30)
        # Define o início do dia (00:00:00) 30 dias atrás para garantir o período completo
        start_of_day_30d = start_dt_30d.replace(hour=0, minute=0, second=0, microsecond=0)
        start_time_30d_ms = int(start_of_day_30d.timestamp() * 1000)
        end_time_30d_ms = int(time.time() * 1000)  # Usado para o final do período (agora)
        # --- FIM DO CÁLCULO DE TEMPOS ---

        # 0. Cache de Histórico de Trades (30 dias) - OTIMIZAÇÃO DE BUSCA PARA VOLUMES
        is_history_cache_fresh = current_time_s - trade_history_30d_timestamp.get(account_name, 0) <= CACHE_FRESHNESS_S

        if not is_history_cache_fresh and exchange_name != 'apex':  # Apex não usa esse cache de trades
            account_logger.info("Cache de trades (30d) expirado/ausente. Buscando via API...")
            try:
                # Busca o histórico dos últimos 30 dias (início do dia 30 dias atrás)
                trades_30d = client.get_trade_history(start_time_30d_ms, end_time_30d_ms)
                trade_history_30d_cache[account_name] = trades_30d
                trade_history_30d_timestamp[account_name] = current_time_s
                account_logger.debug(f"Cache de trades (30d) populado com {len(trades_30d)} trades.")
            except Exception as e:
                account_logger.error(f"Erro ao buscar histórico de trades (30d): {e}")
                trade_history_30d_cache[account_name] = []
        elif exchange_name == 'apex':
            # Garante que o cache para Apex esteja vazio
            trade_history_30d_cache[account_name] = []

        # 1. Volume Semanal (Busca Leve) - USA O CACHE DE 30 DIAS OU FALLBACK
        start_time_7d_ms = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
        trades_for_7d_volume = []

        if trade_history_30d_cache.get(account_name):
            account_logger.debug("Calculando volume (7d) a partir do cache (30d).")
            # Filtra o cache de 30 dias pelo timestamp de 7 dias
            trades_for_7d_volume = [
                t for t in trade_history_30d_cache[account_name]
                if t.get('created_at') and t['created_at'] >= start_time_7d_ms
            ]
        elif exchange_name != 'apex':
            # Fallback: Se o cache falhou na Pacífica, faz a chamada de 7 dias
            account_logger.warning("Cache de trades indisponível. Buscando trades (7d) via API (fallback).")
            trades_for_7d_volume = client.get_trade_history(start_time_7d_ms, end_time_30d_ms)
        else:
            # Apex: Não usa cache de trades, faz a chamada de 7 dias
            trades_for_7d_volume = client.get_trade_history(start_time_7d_ms, end_time_30d_ms)

        total_volume = 0
        if trades_for_7d_volume:
            for trade in trades_for_7d_volume:
                amount = float(trade.get('amount', 0.0))
                trade_price = float(trade.get('price', 0.0))
                total_volume += amount * trade_price
        weekly_volume_cache[account_name] = total_volume
        account_logger.debug(f"Volume semanal calculado: {total_volume}")

        # 2. Métricas do Dashboard (30 dias) - Busca Pesada, OTIMIZADA POR TEMPO DE CACHE
        is_metrics_cache_fresh = current_time_s - dashboard_metrics_timestamp.get(account_name, 0) <= CACHE_FRESHNESS_S

        if not is_metrics_cache_fresh:
            account_logger.info("Métricas do dashboard (30d) expiradas ou ausentes. Recalculando...")

            # --- CÓDIGO DA BUSCA PESADA ---
            if exchange_name == 'apex':
                # Apex usa endpoint de PNL (get_pnl_history)
                # AQUI FOI CORRIGIDO PARA USAR start_time_30d_ms
                pnl_history = client.get_pnl_history(start_time_30d_ms, end_time_30d_ms)
                trades_for_metrics = _adapt_pnl_history_for_dashboard(pnl_history)
            else:
                # Pacífica/Outras usam o histórico de trades de 30 dias que foi cacheado acima
                if trade_history_30d_cache.get(account_name):
                    trades_for_metrics = trade_history_30d_cache[account_name]
                else:
                    # Se o cache falhou para trades (30d), faz a chamada de 30 dias novamente (último recurso)
                    account_logger.warning("Cache de trades (30d) indisponível para métricas. Buscando via API.")
                    # AQUI FOI CORRIGIDO PARA USAR start_time_30d_ms
                    trades_for_metrics = client.get_trade_history(start_time_30d_ms, end_time_30d_ms)

            dashboard_metrics_cache[account_name] = calculate_dashboard_metrics(trades_for_metrics)
            dashboard_metrics_timestamp[account_name] = current_time_s  # Atualiza timestamp
            account_logger.debug("Métricas do dashboard calculadas e cacheadas.")
        else:
            account_logger.debug(f"Métricas do dashboard (30d) frescas. Pulando cálculo.")
            if account_name not in dashboard_metrics_cache:
                dashboard_metrics_cache[account_name] = calculate_dashboard_metrics([])

        # 3. Posições Abertas (Busca Leve)
        open_positions_list = client.get_open_positions()
        open_positions_details = []
        if open_positions_list:
            symbols_to_fetch = list(
                set(pos['symbol'] for pos in open_positions_list if pos.get('symbol')))  # Evita duplicados e None
            all_prices = client.get_current_prices(symbols=symbols_to_fetch)
            all_open_orders = client.get_open_orders()
            prices_map = {p['symbol'].upper().split('-')[0]: float(p.get('mark', p.get('price', 0))) for p in all_prices
                          if p.get('symbol') and (p.get('mark') or p.get('price'))}

            for pos in open_positions_list:
                symbol, entry_price = pos.get('symbol', 'N/A').upper(), float(pos.get('entry_price', 0))
                amount = float(pos.get('amount', 0))
                if symbol == 'N/A' or entry_price == 0: continue  # Pula posição inválida

                leverage = leverage_map.get(symbol.split('-')[0], 1)  # Usa base currency para leverage
                current_price = prices_map.get(symbol.split('-')[0], 0.0)

                pnl_usd = 0.0
                if current_price > 0:  # Evita divisão por zero ou cálculo inválido
                    if pos.get('side') == 'bid':  # Long
                        pnl_usd = (current_price - entry_price) * amount
                    else:  # Short
                        pnl_usd = (entry_price - current_price) * amount

                price_change_pct = ((current_price - entry_price) / entry_price) * 100 if pos.get(
                    'side') == 'bid' else ((entry_price - current_price) / entry_price) * 100 if entry_price > 0 else 0
                pnl_pct = price_change_pct * leverage

                sl_order = next((o for o in all_open_orders if
                                 o.get('symbol', '').upper() == symbol and
                                 o.get('order_type', '').startswith('stop')), None)
                tp_order = next((o for o in all_open_orders if
                                 o.get('symbol', '').upper() == symbol and
                                 o.get('order_type', '').startswith('take_profit')), None)

                pos.update({
                    'account_name': account_name,
                    'current_price': current_price,
                    'sl_price': float(sl_order['stop_price']) if sl_order and sl_order.get('stop_price') else 'N/A',
                    'tp_price': float(tp_order['stop_price']) if tp_order and tp_order.get('stop_price') else 'N/A',
                    'pnl_percentage': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'creation_date_utc': pd.to_datetime(pos.get('created_at'), unit='ms').strftime(
                        '%Y-%m-%d %H:%M:%S') if pos.get('created_at') else 'N/A'
                })
                open_positions_details.append(pos)
        open_positions_cache[account_name] = open_positions_details
        account_logger.debug(f"{len(open_positions_details)} posições abertas processadas.")
        last_refresh_error.pop(account_name, None)  # Limpa erro anterior se sucesso
    except Exception as e:
        error_msg = f"Erro ao buscar dados da API: {e}"
        account_logger.error(error_msg, exc_info=True)
        # Limpa cache de dados leves para essa conta em caso de erro
        account_info_cache[account_name] = None
        open_positions_cache[account_name] = []
        weekly_volume_cache[account_name] = 0
        last_refresh_error[account_name] = error_msg


@app.route('/')
def index():
    config = load_config()
    exchanges = config.get('exchanges', {})
    exchanges_allowed = config.get('exchanges_allowed', [])
    strategy_names = list(STRATEGIES.keys())
    last_backtest_symbol = request.args.get('last_backtest_symbol')
    version_info = get_version_info()
    telegram_config = config.get('telegram', {})
    all_accounts = []

    for ex_name, ex_data in exchanges.items():
        for acc in ex_data.get('accounts', []):
            acc['exchange_name'] = ex_name
            all_accounts.append(acc)

    # Força a busca se o cache estiver vazio para alguma conta
    for acc in all_accounts:
        if acc['account_name'] not in account_info_cache:
            fetch_and_cache_api_data(acc, acc['exchange_name'])

    selected_account_name = request.args.get('account', 'all')
    accounts_to_process = []

    if all_accounts:
        if selected_account_name == 'all' or not any(
                acc['account_name'] == selected_account_name for acc in all_accounts):
            accounts_to_process = all_accounts
        else:
            account_config = next((acc for acc in all_accounts if acc['account_name'] == selected_account_name), None)
            if account_config:
                accounts_to_process.append(account_config)

    account_info_to_display = {}
    dashboard_to_display = {}
    volume_to_display = 0
    setups_to_display = []
    open_positions_to_display = []

    if accounts_to_process:
        total_equity = sum(
            account_info_cache.get(acc['account_name'], {}).get('account_equity', 0) for acc in accounts_to_process if
            acc['account_name'] in account_info_cache and account_info_cache[acc['account_name']])
        total_available = sum(
            account_info_cache.get(acc['account_name'], {}).get('available_to_spend', 0) for acc in accounts_to_process
            if acc['account_name'] in account_info_cache and account_info_cache[acc['account_name']])
        account_info_to_display = {'account_equity': total_equity, 'available_to_spend': total_available}
        volume_to_display = sum(weekly_volume_cache.get(acc['account_name'], 0) for acc in accounts_to_process)

        cached_dashboards = [dashboard_metrics_cache.get(acc['account_name']) for acc in accounts_to_process if
                             dashboard_metrics_cache.get(acc['account_name'])]
        if cached_dashboards:
            total_trades_agg = sum(d['total_trades'] for d in cached_dashboards)
            if total_trades_agg > 0:
                dashboard_to_display = {
                    "total_pnl": sum(d['total_pnl'] for d in cached_dashboards),
                    "total_trades": total_trades_agg,
                    "win_rate": np.average([d['win_rate'] for d in cached_dashboards],
                                           weights=[d['total_trades'] for d in cached_dashboards]),
                    "avg_win": np.average([d['avg_win'] for d in cached_dashboards if d['avg_win'] > 0],
                                          # Evita média com zeros
                                          weights=[d['total_trades'] for d in cached_dashboards if
                                                   d['avg_win'] > 0]) if any(
                        d['avg_win'] > 0 for d in cached_dashboards) else 0,
                    "avg_loss": np.average([d['avg_loss'] for d in cached_dashboards if d['avg_loss'] < 0],
                                           # Evita média com zeros
                                           weights=[d['total_trades'] for d in cached_dashboards if
                                                    d['avg_loss'] < 0]) if any(
                        d['avg_loss'] < 0 for d in cached_dashboards) else 0,
                    "total_fees": sum(d['total_fees'] for d in cached_dashboards)
                }
            else:
                dashboard_to_display = calculate_dashboard_metrics([])  # Usa o default

        for acc in accounts_to_process:
            for market in acc.get('markets_to_trade', []):
                market['account_name'] = acc['account_name']
                setups_to_display.append(market)
            open_positions_to_display.extend(open_positions_cache.get(acc['account_name'], []))

    open_positions_by_account = {}
    for acc_name, positions in open_positions_cache.items():
        if positions:
            open_positions_by_account[acc_name] = {pos['symbol'].upper() for pos in positions}

    # ADICIONADO: Parâmetros TSL lidos do config.json
    profit_lock_params = config.get('profit_lock_params', [])
    ema_trail_delay_params = config.get('ema_trail_delay_params', [])

    return render_template(
        'index.html',
        accounts=all_accounts,
        exchanges_data=exchanges,
        exchanges_allowed=exchanges_allowed,
        running_bots=running_bots.keys(),
        account_info=account_info_to_display,
        volume_7d=volume_to_display,
        dashboard=dashboard_to_display,
        setups_to_display=setups_to_display,
        open_positions_details=open_positions_to_display,
        selected_account_filter=selected_account_name,
        open_positions_by_account=open_positions_by_account,
        last_updated=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        backtest_result=backtest_result_cache,
        last_backtest_symbol=last_backtest_symbol,
        version_info=version_info,
        telegram_config=telegram_config,
        strategy_names=strategy_names,
        profit_lock_params=profit_lock_params,  # ADDED
        ema_trail_delay_params=ema_trail_delay_params  # ADDED
    )


@app.route('/refresh_all', methods=['POST'])
def refresh_all():
    """Limpa apenas os caches de dados em tempo real para forçar uma atualização leve."""
    global account_info_cache, open_positions_cache, weekly_volume_cache, last_refresh_error
    log.info("Forçando a atualização LEVE de todos os dados das contas (exceto métricas históricas)...")
    account_info_cache.clear()
    open_positions_cache.clear()
    weekly_volume_cache.clear()
    last_refresh_error.clear()
    # dashboard_metrics_cache e dashboard_metrics_timestamp NÃO são limpos, mantendo o histórico pesado cacheado.
    # trade_history_30d_cache e trade_history_30d_timestamp TAMBÉM NÃO SÃO LIMPOS
    return redirect(url_for('index'))


@app.route('/start_all', methods=['POST'])
def start_all():
    config = load_config()
    log.info("Iniciando todos os bots...")
    for ex_name, ex_data in config.get('exchanges', {}).items():
        for account in ex_data.get('accounts', []):
            start_bot_logic(account['account_name'], ex_name)
    return redirect(url_for('index'))


@app.route('/stop_all', methods=['POST'])
def stop_all():
    log.warning("Parando todos os bots...")
    for account_name in list(running_bots.keys()):
        stop_bot_logic(account_name)
    return redirect(url_for('index'))


# MODIFICAÇÃO 2: Simplifica drasticamente o start_bot_logic
def start_bot_logic(account_name, exchange_name):
    """Inicia a thread do bot, popula o cache SOB DEMANDA (se novo) e subscreve aos klines."""
    global running_bots, populated_caches, populate_lock  # Removido populate_cache_thread
    account_logger = logging.getLogger(account_name)

    if account_name in running_bots:
        account_logger.warning("Bot já está em execução.")
        return

    config = load_config()
    account_config = None
    found_exchange = None
    # ... (lógica para encontrar account_config mantida) ...
    for ex_n, ex_d in config.get('exchanges', {}).items():
        for acc in ex_d.get('accounts', []):
            if acc['account_name'] == account_name:
                account_config = acc
                found_exchange = ex_n
                break
        if account_config: break

    if not account_config or not found_exchange:
        log.error(f"Configuração não encontrada para a conta '{account_name}'. Bot não iniciado.")
        return
    exchange_name = found_exchange

    # --- Lógica de Inicialização Sob Demanda do WS Manager ---
    # REMOVIDO: Bloco 'first_bot_starting' (WS já está rodando)

    # Identifica os pares/timeframes necessários para ESTE bot
    pairs_needed_by_bot = set()
    min_candles_map = {}
    for setup in account_config.get('markets_to_trade', []):
        symbol = setup.get('base_currency')
        strategy_name = setup.get('strategy_name')
        if symbol and strategy_name and strategy_name in STRATEGIES:
            timeframe = STRATEGIES[strategy_name].get('timeframe')
            min_candles = STRATEGIES[strategy_name].get('min_candles', 200)
            if timeframe:
                ws_symbol = f"{symbol.upper()}-USDT"
                cache_key = (ws_symbol, timeframe)
                pairs_needed_by_bot.add(cache_key)
                min_candles_map[cache_key] = max(min_candles_map.get(cache_key, 0), min_candles)

    # Popula o cache SÍNCRONAMENTE (Mantido para setups adicionados APÓS startup)
    population_successful = True
    for symbol, timeframe in pairs_needed_by_bot:
        cache_key = (symbol, timeframe)
        if cache_key not in populated_caches:
            with populate_lock:
                if cache_key not in populated_caches:
                    log.info(f"Cache para {cache_key} não populado (novo setup?). Iniciando busca síncrona...")
                    min_candles_needed = min_candles_map[cache_key]
                    try:
                        # Chama a função síncrona diretamente
                        ws_manager.populate_initial_cache(symbol, timeframe, min_candles_needed)
                        with ws_manager._lock:
                            if cache_key in ws_manager._cache and len(
                                    ws_manager._cache[cache_key]) >= min_candles_needed:
                                log.info(f"Cache para {cache_key} populado com sucesso (on-demand).")
                                populated_caches.add(cache_key)
                            else:
                                log.error(f"Falha ao popular cache (on-demand) para {cache_key}.")
                                population_successful = False
                    except Exception as pop_err:
                        log.error(f"Erro CRÍTICO ao popular cache (on-demand) para {cache_key}: {pop_err}",
                                  exc_info=True)
                        population_successful = False
        # else: log.debug(f"Cache {cache_key} já populado (ignorado on-demand).")

    # Sincronização de histórico REST (mantida)
    try:
        account_logger.info("Sincronizando histórico REST (trades)...")
        client = get_client(exchange_name, account_config)
        # Define um start_time razoável, pode ser ajustado
        start_time_ms = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
        sync_trade_history(client, exchange_name, account_name, start_time_ms)
        account_logger.info("Sincronização REST (trades) concluída.")
    except Exception as e:
        account_logger.error(f"ERRO ao sincronizar histórico REST ao iniciar: {e}", exc_info=True)
        # Decide se continua mesmo com erro de sync. Por ora, continuamos.

    # Inicia a instância e thread do Bot
    try:
        bot_instance_local = TradingBot(account_config, config, exchange_name, ws_manager)
        bot_thread_local = threading.Thread(target=bot_instance_local.run, name=f"BotThread-{account_name}",
                                            daemon=True)
        bot_thread_local.start()
        running_bots[account_name] = {"instance": bot_instance_local, "thread": bot_thread_local}

        # Subscreve aos pares deste bot no WS Manager
        # (Isto agora irá acionar o ws_manager para iniciar o stream se não estiver ativo)
        for symbol, timeframe in pairs_needed_by_bot:
            ws_manager.subscribe(symbol, timeframe)

    except Exception as e:
        account_logger.critical(f"Falha CRÍTICA ao iniciar thread do bot {account_name}: {e}", exc_info=True)
        # Tenta limpar se a entrada foi adicionada a running_bots
        if account_name in running_bots:
            del running_bots[account_name]
        # Considerar cancelar subscrições feitas no loop acima se falhar aqui?
        for symbol, timeframe in pairs_needed_by_bot:
            ws_manager.unsubscribe(symbol, timeframe)  # Decrementa contador


# MODIFICAÇÃO 3: Simplifica drasticamente o stop_bot_logic
def stop_bot_logic(account_name):
    global running_bots, populated_caches  # populated_caches mantido
    account_logger = logging.getLogger(account_name)
    if account_name in running_bots:
        bot_to_stop_data = running_bots.pop(account_name)  # Remove da lista primeiro
        if bot_to_stop_data and bot_to_stop_data["instance"]:
            account_logger.warning(f"Parando bot...")
            instance = bot_to_stop_data["instance"]
            thread = bot_to_stop_data["thread"]

            # Cancela subscrições WS deste bot
            account_config = instance.account_config
            for setup in account_config.get('markets_to_trade', []):
                symbol = setup.get('base_currency')
                strategy_name = setup.get('strategy_name')
                if symbol and strategy_name and strategy_name in STRATEGIES:
                    timeframe = STRATEGIES[strategy_name].get('timeframe')
                    if timeframe:
                        ws_symbol = f"{symbol.upper()}-USDT"
                        ws_manager.unsubscribe(ws_symbol, timeframe)

            # Para a instância/thread do bot
            instance.stop()
            if thread and thread.is_alive():
                thread.join(timeout=5)  # Espera um pouco


        else:
            account_logger.error("Registro do bot encontrado, mas instância inválida.")
    else:
        account_logger.warning("Bot já estava parado.")


@app.route('/start/<exchange_name>/<account_name>', methods=['POST'])
def start_bot(exchange_name, account_name):
    start_bot_logic(account_name, exchange_name)
    return redirect(url_for('index'))


@app.route('/stop/<account_name>', methods=['POST'])
def stop_bot(account_name):
    stop_bot_logic(account_name)
    return redirect(url_for('index'))


@app.route('/delete_market/<account_name>/<market_id>', methods=['POST'])
def delete_market(account_name, market_id):
    config = load_config()
    account_found = False
    for ex_data in config.get('exchanges', {}).values():
        for account in ex_data.get('accounts', []):
            if account.get('account_name') == account_name:
                initial_len = len(account.get('markets_to_trade', []))
                account['markets_to_trade'] = [m for m in account.get('markets_to_trade', []) if
                                               m.get('id') != market_id]
                if len(account['markets_to_trade']) < initial_len:
                    save_config(config)
                    logging.getLogger(account_name).warning(f"Setup de mercado (ID: {market_id}) removido.")
                else:
                    logging.getLogger(account_name).warning(
                        f"Setup de mercado (ID: {market_id}) não encontrado para remoção.")
                account_found = True
                break
        if account_found: break
    if not account_found:
        log.warning(f"Conta '{account_name}' não encontrada ao tentar remover setup {market_id}.")

    return redirect(url_for('index'))


@app.route('/save_telegram_settings', methods=['POST'])
def save_telegram_settings():
    config = load_config()
    token = request.form.get('telegram_bot_token', '').strip()
    chat_id = request.form.get('telegram_chat_id', '').strip()
    if 'telegram' not in config:
        config['telegram'] = {}
    config['telegram']['bot_token'] = token
    config['telegram']['chat_id'] = chat_id
    config['telegram']['enabled'] = bool(token and chat_id)
    config['telegram']['notify_on_open'] = request.form.get('notify_on_open') == 'true'
    config['telegram']['notify_on_close'] = request.form.get('notify_on_close') == 'true'
    save_config(config)
    flash("Configurações do Telegram salvas com sucesso!", "success")
    log.info("Configurações do Telegram salvas.")
    return redirect(url_for('index'))


@app.route('/toggle_notifications/<exchange_name>/<account_name>', methods=['POST'])
def toggle_notifications(exchange_name, account_name):
    config = load_config()
    telegram_config = config.get('telegram', {})
    if not telegram_config.get('enabled'):
        flash("Preencha as configurações de notificação do Telegram primeiro.", "error")
        return redirect(url_for('index'))
    account_found = False
    try:
        # Acessa a lista de contas da exchange específica
        accounts_list = config['exchanges'][exchange_name]['accounts']
        for i, account in enumerate(accounts_list):
            if account['account_name'] == account_name:
                current_status = account.get('notifications_enabled', False)
                new_status = not current_status
                accounts_list[i]['notifications_enabled'] = new_status  # Modifica na lista original
                account_found = True
                logging.getLogger(account_name).info(f"Notificações {'ativadas' if new_status else 'desativadas'}.")
                break
    except KeyError:
        flash(f"Configuração para a exchange '{exchange_name}' não encontrada.", "error")
        log.error(f"Exchange '{exchange_name}' não encontrada ao tentar alternar notificações para '{account_name}'.")
        return redirect(url_for('index'))

    if account_found:
        save_config(config)
    else:
        flash(f"Conta '{account_name}' não encontrada na exchange '{exchange_name}'.", "error")
        log.error(
            f"Conta '{account_name}' não encontrada na exchange '{exchange_name}' ao tentar alternar notificações.")

    return redirect(url_for('index'))


@app.route('/process_setup_form', methods=['POST'])
def process_setup_form():
    global backtest_result_cache
    action = request.form.get('action')
    base_currency_form = request.form.get('base_currency', '').strip().upper()
    account_name_form = request.form.get('account_name')

    # Validação básica
    if not account_name_form:
        flash("Nome da conta é obrigatório.", "error")
        return redirect(url_for('index'))
    if not base_currency_form:
        flash("Nome do ativo é obrigatório.", "error")
        return redirect(url_for('index'))

    account_logger = logging.getLogger(account_name_form)
    backtest_logger = logging.getLogger("Backtester")

    if action == 'add_market':
        try:
            config = load_config()
            new_market_setup = {
                "id": str(int(time.time() * 1000)),
                "base_currency": base_currency_form,
                "quote_currency": "USDT",  # Fixo por enquanto
                "leverage": int(request.form['leverage']),
                "strategy_name": request.form['strategy_name'],
                "risk_per_trade": float(request.form['risk']) / 100.0,
                "stop_loss_config": {"type": "percentage", "value": float(request.form['sl_value']) / 100.0},
                # Assumindo % por enquanto
                "take_profit_rrr": float(request.form['rrr']),
                "exit_mode": request.form.get('exit_mode', 'passivo'),
                "direction_mode": request.form.get('direction_mode', 'long_short')
            }

            # Validações de valores
            if not (0 < new_market_setup['leverage'] <= 100): raise ValueError("Alavancagem inválida.")
            if not (0 < new_market_setup['risk_per_trade'] <= 0.1): raise ValueError(
                "Risco por trade inválido (0.01% a 10%).")
            if not (0 < new_market_setup['stop_loss_config']['value'] <= 0.5): raise ValueError(
                "Stop loss inválido (0.01% a 50%).")
            if not (0.1 <= new_market_setup['take_profit_rrr'] <= 20): raise ValueError(
                "Take Profit RRR inválido (0.1 a 20).")

            tsl_type = request.form.get('tsl_type')
            if tsl_type and tsl_type != 'none':
                tsl_config = {"type": tsl_type}

                # --- LÓGICA DE TRATAMENTO DO NOVO VALOR TSL ---
                tsl_value_input = request.form.get('tsl_value')
                tsl_value_converted = None

                # Variáveis para os novos dicionários - usando .get(..., '') para strings vazias
                tsl_profit_lock_index_str = request.form.get('tsl_profit_lock_index', '').strip()
                tsl_ema_delay_index_str = request.form.get('tsl_ema_delay_index', '').strip()

                # Tipos que usam valor direto (float)
                if tsl_type in ['atr', 'fixed_pct', 'high_low_trail', 'ema_trail', 'atr_dynamic',
                                'candle_range_confirm']:
                    if not tsl_value_input:
                        raise ValueError(f"O valor de Trailing Stop para o modo '{tsl_type.upper()}' é obrigatório.")

                    tsl_value = float(tsl_value_input)
                    if tsl_type in ['atr', 'atr_dynamic']:  # ATR Multiplier is used directly
                        if not (0.1 <= tsl_value <= 10): raise ValueError("Múltiplo ATR inválido (0.1 a 10).")
                        tsl_value_converted = tsl_value
                    elif tsl_type in ['fixed_pct', 'high_low_trail', 'ema_trail', 'candle_range_confirm']:
                        # PCT Value needs conversion
                        if not (0.01 <= tsl_value <= 10): raise ValueError(
                            "Valor Percentual TSL inválido (0.01% a 10%).")
                        tsl_value_converted = tsl_value / 100.0

                    tsl_config['tsl_value'] = tsl_value_converted

                    # Lógica para Dicionários (profit_lock)
                elif tsl_type == 'profit_lock':
                    profit_lock_params = config.get('profit_lock_params', [])
                    if tsl_profit_lock_index_str.isdigit():
                        index = int(tsl_profit_lock_index_str)
                        if 0 <= index < len(profit_lock_params):
                            tsl_config['tsl_value'] = profit_lock_params[index]
                        else:
                            raise ValueError("Índice de Parâmetro Profit Lock inválido.")
                    else:
                        raise ValueError("Parâmetro Profit Lock não selecionado.")

                        # Lógica para Dicionários (ema_trail_delay)
                elif tsl_type == 'ema_trail_delay':
                    ema_trail_delay_params = config.get('ema_trail_delay_params', [])
                    if tsl_ema_delay_index_str.isdigit():
                        index = int(tsl_ema_delay_index_str)
                        if index < len(ema_trail_delay_params):
                            tsl_config['tsl_value'] = ema_trail_delay_params[index]
                        else:
                            raise ValueError("Índice de Parâmetro EMA Trail Delay inválido.")
                    else:
                        raise ValueError("Parâmetro EMA Trail Delay não selecionado.")

                elif tsl_type == 'breakeven':
                    breakeven_trigger = float(request.form.get('tsl_breakeven_trigger', 1.0))
                    if not (0.1 <= breakeven_trigger <= 10): raise ValueError(
                        "Gatilho Breakeven RRR inválido (0.1 a 10).")
                    tsl_config['breakeven_trigger_rrr'] = breakeven_trigger

                tsl_config['remove_tp_on_trail'] = request.form.get(
                    'remove_tp_on_trail') == 'true'  # Converte para bool
                new_market_setup['trailing_stop_config'] = tsl_config
            # ... (rest of the function)

            account_found = False
            for ex_data in config.get('exchanges', {}).values():
                for account in ex_data.get('accounts', []):
                    if account.get('account_name') == account_name_form:
                        markets = account.get('markets_to_trade', [])
                        # Edita se já existe base_currency, senão adiciona
                        existing_idx = next(
                            (i for i, m in enumerate(markets) if m.get('base_currency') == base_currency_form),
                            -1)
                        if existing_idx != -1:
                            # Mantém o ID original ao editar
                            new_market_setup['id'] = markets[existing_idx].get('id', new_market_setup['id'])
                            markets[existing_idx] = new_market_setup
                            account_logger.log(logging.SUCCESS,
                                               f"Setup para {base_currency_form} ATUALIZADO.")  # Usando nível SUCCESS
                        else:
                            markets.append(new_market_setup)
                            account_logger.log(logging.SUCCESS,
                                               f"Setup para {base_currency_form} ADICIONADO.")  # Usando nível SUCCESS
                        account['markets_to_trade'] = markets
                        account_found = True
                        break
                if account_found:
                    break

            if account_found:
                save_config(config)
            else:
                flash(f"Conta '{account_name_form}' não encontrada para adicionar setup.", "error")
                log.error(f"Conta '{account_name_form}' não encontrada ao tentar adicionar/atualizar setup.")

        except ValueError as ve:
            flash(f"Erro nos dados do formulário: {ve}", "error")
            log.warning(f"Erro de validação ao processar setup para {account_name_form}/{base_currency_form}: {ve}")
        except Exception as e:
            flash(f"Erro inesperado ao salvar setup: {e}", "error")
            log.error(f"Erro inesperado ao processar setup para {account_name_form}/{base_currency_form}: {e}",
                      exc_info=True)

        return redirect(url_for('index'))

    elif action == 'run_backtest':
        try:
            leverage = int(request.form.get('leverage'))
            if not (1 <= leverage <= 100): raise ValueError("Alavancagem inválida para backtest.")
            symbol = f"{base_currency_form}/USDT"  # Assume USDT
            backtest_result_cache.pop(symbol, None)  # Limpa cache anterior
            backtest_logger.info(f"Iniciando backtest para {symbol} (Alav: {leverage}x)...")

            all_results = run_full_backtest(symbol, leverage)

            if all_results:
                backtest_result_cache[symbol] = all_results[0]  # Salva apenas o melhor resultado no cache
                df_results = pd.DataFrame(all_results)

                # Formatação e Renomeação (igual ao anterior)
                percent_cols = ['Retorno %', 'Taxa de Acerto %', 'Long Return %', 'Short Return %', 'Long Win Rate %',
                                'Short Win Rate %']
                for col in percent_cols:
                    if col in df_results.columns: df_results[col] = df_results[col].map('{:+.2f}%'.format)
                rename_map = {'Stop Loss %': 'SL %', 'Take Profit RRR': 'TP RRR', 'Total de Trades': 'Trades',
                              'Retorno %': 'Retorno %', 'Taxa de Acerto %': 'Acerto %', 'Long Trades': 'L. Trades',
                              'Long Return %': 'L. Retorno %', 'Long Win Rate %': 'L. Acerto %',
                              'Short Trades': 'S. Trades', 'Short Return %': 'S. Retorno %',
                              'Short Win Rate %': 'S. Acerto %'}
                df_results.rename(columns=rename_map, inplace=True)
                cols_to_log = ['Estratégia', 'SL %', 'TP RRR', 'Trades', 'Retorno %', 'Acerto %', 'L. Trades',
                               'L. Retorno %', 'L. Acerto %', 'S. Trades', 'S. Retorno %', 'S. Acerto %']
                existing_cols = [col for col in cols_to_log if col in df_results.columns]
                log_body = df_results[existing_cols].to_string(index=False)
                log_header = f"===== MELHORES RESULTADOS PARA {symbol} (Alav: {leverage}x) ====="
                full_log_message = f"<pre>{log_header}\n{log_body}\n{'=' * len(log_header)}</pre>"

                # Envia o log como HTML usando o 'extra' dict
                backtest_logger.info(full_log_message, extra={'is_html': True})
            else:
                backtest_logger.warning(f"Backtest para {symbol} não produziu resultados.")

        except ValueError as ve:
            flash(f"Erro nos dados para backtest: {ve}", "error")
            backtest_logger.warning(f"Erro de validação ao iniciar backtest para {base_currency_form}: {ve}")
        except Exception as e:
            flash(f"Erro inesperado durante o backtest: {e}", "error")
            backtest_logger.error(f"Erro inesperado no backtest para {base_currency_form}: {e}", exc_info=True)

        return redirect(url_for('index',
                                last_backtest_symbol=symbol if 'symbol' in locals() else base_currency_form))  # Garante que o símbolo vá para a URL


@app.route('/api/pnl_chart_data')
def get_pnl_chart_data():
    # --- INÍCIO DA MODIFICAÇÃO (Buscar 30 dias) ---
    log.info(f"Buscando dados do gráfico PNL (30 dias) via query unificada...")

    end_time = datetime.now()
    # Define o início da busca para 30 dias atrás
    start_time_30d = end_time - timedelta(days=30)
    start_time_30d_ms = int(start_time_30d.timestamp() * 1000)
    # --- FIM DA MODIFICAÇÃO ---

    try:
        # Chama a função do database.py que retorna todos os dados necessários dos últimos 30 dias
        all_pnl_data = query_daily_pnl_all_accounts(start_time_30d_ms)  # Passa o timestamp de 30 dias

        if not all_pnl_data:
            log.warning("Nenhum dado PNL encontrado pela query unificada nos últimos 30 dias.")
            # Retorna dados vazios para o gráfico exibir 'sem dados' em vez de erro 404
            return jsonify({"labels": [], "datasets": []})
            # return jsonify({"error": "Nenhum dado PNL encontrado nos últimos 30 dias."}), 404 # Alternativa

        pnl_df = pd.DataFrame(all_pnl_data)
        pnl_df['date'] = pd.to_datetime(pnl_df['date'])
        pnl_df['daily_pnl'] = pd.to_numeric(pnl_df['daily_pnl'], errors='coerce').fillna(0)

        # --- INÍCIO DA MODIFICAÇÃO (Processamento para 30 dias) ---
        # Cria o range de datas completo dos últimos 30 dias
        date_range_30d = pd.date_range(start=start_time_30d.date(), end=end_time.date(), freq='D')

        chart_labels = [date.strftime('%Y-%m-%d') for date in date_range_30d]
        chart_datasets = []

        account_names_in_data = sorted(pnl_df['account_name'].unique())

        # Processa cada conta encontrada nos dados
        for account_name in account_names_in_data:
            account_df = pnl_df[pnl_df['account_name'] == account_name].set_index('date')

            # Cria a série temporal de 30 dias, preenchendo dias faltantes com 0
            pnl_series_30d = account_df['daily_pnl'].reindex(date_range_30d, fill_value=0)

            # Calcula o cumulativo diretamente sobre os 30 dias
            cumulative_pnl_30d = pnl_series_30d.cumsum()

            # Prepara os dados para o dataset desta conta (não precisa mais ser relativo)
            account_data = [round(pnl, 2) for pnl in cumulative_pnl_30d.values]
            # --- FIM DA MODIFICAÇÃO ---

            chart_datasets.append({
                "label": account_name,
                "data": account_data,
                "tension": 0.1,
                "pointRadius": 2,
                "pointHoverRadius": 5
            })

        final_chart_data = {
            "labels": chart_labels,
            "datasets": chart_datasets
        }

        return jsonify(final_chart_data)

    except Exception as e:
        log.error(f"Erro ao processar dados PNL da query unificada: {e}", exc_info=True)
        return jsonify({"error": "Erro interno ao processar dados do gráfico."}), 500


def _background_populate_caches(pairs_to_populate_dict: dict):
    """Função para ser executada em uma thread separada para popular caches."""
    log.info("Thread de população de cache iniciada.")

    # Separa o par BTC/1h dos outros
    btc_key = ('BTC-USDT', '1h')
    btc_min_candles = pairs_to_populate_dict.pop(btc_key, None)
    other_pairs_to_populate = pairs_to_populate_dict

    # Popula BTC/1h primeiro (se necessário)
    if btc_min_candles is not None:
        log.info(f"Populando cache prioritário {btc_key} em background...")
        try:
            ws_manager.populate_initial_cache(btc_key[0], btc_key[1], btc_min_candles)
            with ws_manager._lock:
                if btc_key in ws_manager._cache and len(ws_manager._cache[btc_key]) >= btc_min_candles:
                    with populate_lock:
                        populated_caches.add(btc_key)
                    log.info(f"Cache {btc_key} populado com sucesso (background).")
                else:
                    log.error(f"Cache {btc_key} (background) não atingiu o mínimo de candles.")
        except Exception as e:
            log.critical(f"Falha grave durante a população (background) do cache {btc_key}: {e}", exc_info=True)

    # Popula os demais pares
    log.info(f"Populando cache dos demais {len(other_pairs_to_populate)} pares em background...")
    try:
        populated_keys_others = ws_manager.populate_all_initial_caches(other_pairs_to_populate)
        with populate_lock:
            populated_caches.update(populated_keys_others)
        log.info(
            f"População de cache (demais pares, background) concluída. {len(populated_keys_others)}/{len(other_pairs_to_populate)} bem-sucedidos.")
    except Exception as e:
        log.critical(f"Falha grave durante a população do cache (demais pares, background): {e}", exc_info=True)

    log.info("Thread de população de cache finalizada.")


# MODIFICAÇÃO 4: Lógica de inicialização principal
if __name__ == '__main__':
    setup_logging()
    try:
        init_db()
        log.info("=============================================")
        log.info("Iniciando KBot-Trading WebApp...")

        # --- LÓGICA DE STARTUP MODIFICADA ---
        log.info("Carregando configuração...")
        config = load_config()

        log.info("Identificando pares/timeframes ativos...")
        pairs_to_populate = get_active_symbol_timeframes(config)  # Mantém a lógica que pode adicionar BTC/1h

        if pairs_to_populate:
            log.info(f"Definindo subscrições iniciais ({len(pairs_to_populate)}) no WS Manager...")
            ws_manager.set_initial_subscriptions(set(pairs_to_populate.keys()))

            log.info("Iniciando WebSocket Manager (thread asyncio)...")
            ws_manager.start()
            # NÃO esperamos mais o WS Manager estar pronto aqui

            # *** INICIA A POPULAÇÃO DO CACHE EM BACKGROUND ***
            log.info("Iniciando população de cache em background...")
            cache_thread = threading.Thread(
                target=_background_populate_caches,
                args=(pairs_to_populate.copy(),),  # Passa uma cópia do dicionário
                daemon=True,
                name="CachePopulationThread"
            )
            cache_thread.start()
            # *** NÃO ESPERAMOS A THREAD TERMINAR ***

        else:
            log.warning("Nenhum setup ativo encontrado. WS Manager iniciado, mas inativo (sem caches).")
            # Inicia o WS Manager mesmo sem pares, caso sejam adicionados depois
            ws_manager.start()

            # *** INICIA O SERVIDOR WEB IMEDIATAMENTE ***
        log.info("=============================================")
        log.info("Iniciando servidor web Flask (SocketIO)... Acesso liberado!")
        # A população do cache continua em background
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True, use_reloader=False)

    except KeyboardInterrupt:
        log.warning("Ctrl+C recebido. Encerrando...")
    except Exception as e:
        log.critical(f"Falha CRÍTICA: {e}", exc_info=True)
    finally:
        log.warning("Encerrando aplicação...")
        # (Lógica de shutdown mantida)
        if ws_manager.is_running():
            log.warning("Encerrando WebSocket Manager...")
            ws_manager.stop()
        log.warning("Encerrando Agendador...")
        if scheduler.running:
            scheduler.shutdown()
        log.info("Aplicação KBot-Trading encerrada.")
# app.py
import subprocess

import numpy as np
import requests
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import json
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import atexit
from apscheduler.schedulers.background import BackgroundScheduler

from bot_logic import TradingBot
from strategies import STRATEGIES
from exchange_factory import get_client
from backtester import run_full_backtest
from database import init_db, sync_order_history, sync_trade_history, sync_pnl_history
from notifications import check_and_send_close_alerts, check_and_send_open_alerts, check_and_send_close_alerts_apex

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Variáveis globais
bot_logs = [{"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "account": "Sistema", "level": "INFO",
             "message": "Bot está parado. Clique em 'Iniciar' para começar."}]
account_info_cache = {}
dashboard_metrics_cache = {}
open_positions_cache = {}
weekly_volume_cache = {}
last_refresh_error = {}
backtest_result_cache = {}
running_bots = {}


def get_version_info():
    """Busca a versão local (commit hash) e compara com a remota no GitHub."""
    try:
        # Garante que a pasta .git existe para não dar erro
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
        return {"error": f"Falha ao verificar versão: {e}"}


def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)


def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)


def sync_all_trade_histories(start_time_ms: int):
    """Função auxiliar para sincronizar o histórico de trades de TODAS as contas."""
    print(f"AGENDADOR: Iniciando sincronização de TRADES - {datetime.now()}")
    config = load_config()
    for exchange_name, exchange_data in config.get('exchanges', {}).items():
        for account_config in exchange_data.get('accounts', []):
            try:
                client = get_client(exchange_name, account_config)
                sync_trade_history(client, exchange_name, account_config['account_name'], start_time_ms)
            except Exception as e:
                print(f"ERRO na sincronização de trades para {account_config['account_name']}: {e}")


def sync_apex_pnl_task():
    """Tarefa agendada para sincronizar o PNL histórico de todas as contas Apex."""
    print(f"AGENDADOR: Iniciando sincronização de PNL da Apex - {datetime.now()}")
    config = load_config()

    apex_exchange_data = config.get('exchanges', {}).get('apex', {})
    if not apex_exchange_data.get('accounts'):
        return

    # --- INÍCIO DA CORREÇÃO ---
    # Período de busca alterado para os últimos 7 dias
    start_time_ms = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
    # --- FIM DA CORREÇÃO ---

    for account_config in apex_exchange_data.get('accounts', []):
        try:
            client = get_client('apex', account_config)
            sync_pnl_history(client, account_config['account_name'], start_time_ms)
        except Exception as e:
            print(f"ERRO na sincronização de PNL para {account_config['account_name']}: {e}")

def hourly_alert_task():
    """Tarefa principal que roda de hora em hora."""
    config = load_config()
    if not config.get('telegram', {}).get('enabled', False):
        return

    start_time_ms = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
    sync_all_trade_histories(start_time_ms)
    check_and_send_open_alerts()
    check_and_send_close_alerts()
    check_and_send_close_alerts_apex()


def full_daily_sync_task():
    """Tarefa diária que faz uma sincronização completa (trades e ordens) para TODAS as contas."""
    print(f"AGENDADOR DIÁRIO: Iniciando tarefa de sincronização COMPLETA - {datetime.now()}")
    config = load_config()
    daily_start_time_ms = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)

    for exchange_name, exchange_data in config.get('exchanges', {}).items():
        for account_config in exchange_data.get('accounts', []):
            try:
                print(f"Sincronização diária completa para a conta: {account_config['account_name']}")
                client = get_client(exchange_name, account_config)
                sync_order_history(client, exchange_name, account_config['account_name'])
                sync_trade_history(client, exchange_name, account_config['account_name'], daily_start_time_ms)
            except Exception as e:
                print(f"ERRO durante a sincronização diária da conta {account_config['account_name']}: {e}")
    print("AGENDADOR DIÁRIO: Tarefa de sincronização completa concluída.")


scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(hourly_alert_task, 'interval', hours=1, next_run_time=datetime.now() + timedelta(seconds=30))
scheduler.add_job(full_daily_sync_task, 'interval', hours=24, next_run_time=datetime.now() + timedelta(minutes=5))
scheduler.add_job(sync_apex_pnl_task, 'interval', hours=1, next_run_time=datetime.now() + timedelta(minutes=1))

scheduler.start()
atexit.register(lambda: scheduler.shutdown())


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

    # --- INÍCIO DA ALTERAÇÃO ---
    # Adiciona as credenciais específicas de cada exchange
    if selected_exchange == 'pacifica':
        account_details["main_public_key"] = request.form.get('main_public_key')
        account_details["agent_private_key"] = request.form.get('agent_private_key')
    elif selected_exchange == 'apex':
        account_details["api_key"] = request.form.get('api_key')
        account_details["api_secret"] = request.form.get('api_secret')
        account_details["passphrase"] = request.form.get('passphrase')
        account_details["zk_seeds"] = request.form.get('zk_seeds')
        account_details["zk_l2key"] = request.form.get('zk_l2key')
    # --- FIM DA ALTERAÇÃO ---

    original_name = request.form.get('original_account_name')
    accounts_in_exchange = exchange_details.get('accounts', [])

    if original_name:
        for i, acc in enumerate(accounts_in_exchange):
            if acc['account_name'] == original_name:
                # Ao editar, mantém chaves secretas se não forem fornecidas novamente
                if selected_exchange == 'pacifica' and not account_details["agent_private_key"]:
                    account_details["agent_private_key"] = acc.get("agent_private_key")
                elif selected_exchange == 'apex':
                    if not account_details["api_secret"]: account_details["api_secret"] = acc.get("api_secret")
                    if not account_details["passphrase"]: account_details["passphrase"] = acc.get("passphrase")
                    if not account_details["zk_seeds"]: account_details["zk_seeds"] = acc.get("zk_seeds")
                    if not account_details["zk_l2key"]: account_details["zk_l2key"] = acc.get("zk_l2key")

                account_details['markets_to_trade'] = acc.get('markets_to_trade', [])
                account_details['notifications_enabled'] = acc.get('notifications_enabled', False)
                accounts_in_exchange[i] = account_details
                break
    else:
        account_details['markets_to_trade'] = []
        account_details['notifications_enabled'] = False
        accounts_in_exchange.append(account_details)

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
    return redirect(url_for('index'))


def calculate_dashboard_metrics(trades: list) -> dict:
    if not trades:
        return {"total_trades": 0, "total_pnl": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0, "total_fees": 0}

    df = pd.DataFrame(trades)

    # Garante que as colunas numéricas tenham o tipo correto, tratando possíveis erros
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
    df['fee'] = pd.to_numeric(df['fee'], errors='coerce').fillna(0)

    # 1. Filtra apenas para trades de fechamento
    closed_trades_df = df[df['side'].str.startswith('close_', na=False)].copy()

    if closed_trades_df.empty:
        return {"total_trades": 0, "total_pnl": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0, "total_fees": 0}

    if 'order_id' in closed_trades_df.columns:
        operations_df = closed_trades_df.groupby('order_id').agg(
            pnl=('pnl', 'sum'),
            fee=('fee', 'sum')
        ).reset_index()
    else:
        operations_df = closed_trades_df

    if operations_df.empty:
        return {"total_trades": 0, "total_pnl": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0, "total_fees": 0}

    winning_trades_pnl = operations_df[operations_df['pnl'] > 0]['pnl'].tolist()
    losing_trades_pnl = operations_df[operations_df['pnl'] <= 0]['pnl'].tolist()

    total_fees = operations_df['fee'].sum()
    total_trades = len(operations_df)
    total_pnl = operations_df['pnl'].sum()
    win_rate = (len(winning_trades_pnl) / total_trades) * 100 if total_trades > 0 else 0
    avg_win = sum(winning_trades_pnl) / len(winning_trades_pnl) if winning_trades_pnl else 0
    avg_loss = sum(losing_trades_pnl) / len(losing_trades_pnl) if losing_trades_pnl else 0

    return {
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_fees": total_fees
    }


def _adapt_pnl_history_for_dashboard(pnl_history: list) -> list:
    """Adapta o formato do histórico de PNL da Apex para o formato esperado pela função de cálculo de métricas."""
    adapted_trades = []
    for pnl_entry in pnl_history:
        # A função calculate_dashboard_metrics precisa de um campo 'pnl' e um campo 'side' que comece com 'close_'
        if pnl_entry.get('type') == 'CLOSE_POSITION':
            adapted_trades.append({
                'pnl': float(pnl_entry.get('totalPnl') or 0.0),
                'side': 'close_position', # Suficiente para a lógica 'startswith("close_")'
                'fee': float(pnl_entry.get('fee') or 0.0)
            })
    return adapted_trades


def fetch_and_cache_api_data(account_config: dict, exchange_name: str):
    """Função genérica para buscar dados da API usando a factory."""
    global account_info_cache, dashboard_metrics_cache, open_positions_cache, last_refresh_error, weekly_volume_cache
    account_name = account_config['account_name']
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

        start_time_7d = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
        end_time_7d = int(time.time() * 1000)

        trade_history_7d = client.get_trade_history(start_time_7d, end_time_7d)

        total_volume = 0
        if trade_history_7d:
            for trade in trade_history_7d:
                # Calcula o volume do trade: quantidade * preço de entrada
                amount = float(trade.get('amount', 0.0))
                trade_price = float(trade.get('price', 0.0))
                total_volume += amount * trade_price

        weekly_volume_cache[account_name] = total_volume

        end_time_30d = int(time.time() * 1000)
        start_time_30d = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)

        if exchange_name == 'apex':
            pnl_history = client.get_pnl_history(start_time_30d, end_time_30d)
            trades_for_metrics = _adapt_pnl_history_for_dashboard(pnl_history)
        else:
            trades_for_metrics = client.get_trade_history(start_time_30d, end_time_30d)

        dashboard_metrics_cache[account_name] = calculate_dashboard_metrics(trades_for_metrics)

        open_positions_list = client.get_open_positions()
        open_positions_details = []
        if open_positions_list:
            symbols_to_fetch = [pos['symbol'] for pos in open_positions_list]
            all_prices = client.get_current_prices(symbols=symbols_to_fetch)
            all_open_orders = client.get_open_orders()
            prices_map = {p['symbol'].upper().split('-')[0]: float(p.get('mark', p.get('price', 0))) for p in all_prices
                          if p.get('mark') or p.get('price')}

            for pos in open_positions_list:
                symbol, entry_price = pos['symbol'].upper(), float(pos['entry_price'])
                leverage = leverage_map.get(symbol, 1)
                current_price = prices_map.get(symbol, 0.0)

                sl_order = next((o for o in all_open_orders if
                                 o.get('symbol', '').upper().split('-')[0] == symbol and
                                 o.get('order_type', '').startswith('stop')), None)
                tp_order = next((o for o in all_open_orders if
                                 o.get('symbol', '').upper().split('-')[0] == symbol and
                                 o.get('order_type', '').startswith('take_profit')), None)

                price_change_pct = ((current_price - entry_price) / entry_price) * 100 if pos.get(
                    'side') == 'bid' else ((entry_price - current_price) / entry_price) * 100 if entry_price > 0 else 0
                pnl_pct = price_change_pct * leverage

                pos.update({
                    'account_name': account_name,
                    'current_price': current_price,
                    'sl_price': float(sl_order['stop_price']) if sl_order and sl_order.get('stop_price') else 'N/A',
                    'tp_price': float(tp_order['stop_price']) if tp_order and tp_order.get('stop_price') else 'N/A',
                    'pnl_percentage': pnl_pct,
                    'creation_date_utc': pd.to_datetime(pos['created_at'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')
                })
                open_positions_details.append(pos)
        open_positions_cache[account_name] = open_positions_details
        last_refresh_error.pop(account_name, None)
    except Exception as e:
        error_msg = f"Erro ao buscar dados para a conta {account_name}: {e}"
        print(f"ERRO FATAL ao buscar dados da API: {e}")
        last_refresh_error[account_name] = error_msg
        account_info_cache[account_name] = None
        dashboard_metrics_cache[account_name] = None
        open_positions_cache[account_name] = []
        weekly_volume_cache[account_name] = 0


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

    # --- INÍCIO DA CORREÇÃO ---
    # A busca de dados agora é feita apenas uma vez, dentro do fetch_and_cache
    for acc in all_accounts:
        if acc['account_name'] not in account_info_cache:  # Evita recarregar a cada refresh
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

    # Agrega os dados a partir do CACHE, em vez de fazer novas chamadas de API
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

        # Agrega as métricas do dashboard a partir do cache
        cached_dashboards = [dashboard_metrics_cache.get(acc['account_name']) for acc in accounts_to_process if
                             dashboard_metrics_cache.get(acc['account_name'])]
        if cached_dashboards:
            dashboard_to_display = {
                "total_pnl": sum(d['total_pnl'] for d in cached_dashboards),
                "total_trades": sum(d['total_trades'] for d in cached_dashboards),
                "win_rate": np.average([d['win_rate'] for d in cached_dashboards],
                                       weights=[d['total_trades'] for d in cached_dashboards]) if sum(
                    d['total_trades'] for d in cached_dashboards) > 0 else 0,
                "avg_win": np.average([d['avg_win'] for d in cached_dashboards],
                                      weights=[d['win_rate'] for d in cached_dashboards]) if sum(
                    d['win_rate'] for d in cached_dashboards) > 0 else 0,
                "avg_loss": np.average([d['avg_loss'] for d in cached_dashboards],
                                       weights=[100 - d['win_rate'] for d in cached_dashboards]) if sum(
                    100 - d['win_rate'] for d in cached_dashboards) > 0 else 0,
                "total_fees": sum(d['total_fees'] for d in cached_dashboards)
            }

        for acc in accounts_to_process:
            for market in acc.get('markets_to_trade', []):
                market['account_name'] = acc['account_name']
                setups_to_display.append(market)
            open_positions_to_display.extend(open_positions_cache.get(acc['account_name'], []))

    open_position_symbols = {pos['symbol'].upper() for pos in open_positions_to_display}

    selected_level = request.args.get('log_level', 'all')
    selected_account_log = request.args.get('log_account', 'all')
    filtered_logs = [log for log in bot_logs if (selected_level == 'all' or log.get('level') == selected_level) and (
            selected_account_log == 'all' or log.get('account') == selected_account_log)]
    log_levels = sorted(list(set(log.get('level') for log in bot_logs)))
    log_accounts = sorted(list(set(log.get('account') for log in bot_logs if log.get('account'))))

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
        open_position_symbols=open_position_symbols,
        last_updated=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        backtest_result=backtest_result_cache,
        last_backtest_symbol=last_backtest_symbol,
        logs=filtered_logs,
        log_levels=log_levels,
        log_accounts=log_accounts,
        selected_level=selected_level,
        selected_account=selected_account_log,
        version_info=version_info,
        telegram_config=telegram_config,
        strategy_names=strategy_names
    )


@app.route('/api/get_logs')
def get_logs():
    return jsonify(logs=bot_logs)


@app.route('/refresh_all', methods=['POST'])
def refresh_all():
    """Limpa todos os caches de dados para forçar uma atualização completa."""
    global account_info_cache, dashboard_metrics_cache, open_positions_cache, weekly_volume_cache, last_refresh_error
    account_info_cache.clear()
    dashboard_metrics_cache.clear()
    open_positions_cache.clear()
    weekly_volume_cache.clear()
    last_refresh_error.clear()

    bot_logs.append({
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "account": "Sistema",
        "level": "INFO",
        "message": "Forçando a atualização de todos os dados das contas..."
    })
    return redirect(url_for('index'))


@app.route('/start_all', methods=['POST'])
def start_all():
    config = load_config()
    for ex_name, ex_data in config.get('exchanges', {}).items():
        for account in ex_data.get('accounts', []):
            start_bot_logic(account['account_name'], ex_name)
    return redirect(url_for('index'))


@app.route('/stop_all', methods=['POST'])
def stop_all():
    for account_name in list(running_bots.keys()):
        stop_bot_logic(account_name)
    return redirect(url_for('index'))


def start_bot_logic(account_name, exchange_name):
    """Inicia a thread do bot para uma conta específica."""
    global running_bots, bot_logs
    if account_name not in running_bots:
        config = load_config()
        account_config = next(
            (acc for acc in config['exchanges'][exchange_name]['accounts'] if acc['account_name'] == account_name),
            None)
        if account_config:
            try:
                print(f"Sincronizando histórico para a conta {account_name} antes de iniciar o bot...")
                # --- ALTERAÇÃO PRINCIPAL AQUI ---
                # Substitui a chamada hardcoded pela factory
                client = get_client(exchange_name, account_config)
                # --- FIM DA ALTERAÇÃO ---

                sync_order_history(client, exchange_name, account_name)

                start_time_ms = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
                sync_trade_history(client, exchange_name, account_name, start_time_ms)
                print(f"Sincronização para {account_name} concluída.")
            except Exception as e:
                error_msg = f"ERRO ao sincronizar o histórico para a conta {account_name} ao iniciar: {e}"
                print(error_msg)
                bot_logs.append(
                    {"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "account": "Sistema", "level": "ERROR",
                     "message": error_msg})

            bot_instance_local = TradingBot(account_config, config, bot_logs, exchange_name)
            bot_thread_local = threading.Thread(target=bot_instance_local.run, daemon=True)
            bot_thread_local.start()
            running_bots[account_name] = {"instance": bot_instance_local, "thread": bot_thread_local}



def stop_bot_logic(account_name):
    global running_bots
    if account_name in running_bots:
        bot_to_stop = running_bots.get(account_name)
        if bot_to_stop:
            bot_to_stop["instance"].stop()
            del running_bots[account_name]
            bot_logs.append(
                {"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "account": "Sistema", "level": "INFO",
                 "message": f"Bot para a conta '{account_name}' parado."})


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
    for ex_data in config.get('exchanges', {}).values():
        for account in ex_data.get('accounts', []):
            if account.get('account_name') == account_name:
                account['markets_to_trade'] = [m for m in account.get('markets_to_trade', []) if
                                               m.get('id') != market_id]
                save_config(config)
                return redirect(url_for('index'))
    return redirect(url_for('index'))


@app.route('/save_telegram_settings', methods=['POST'])
def save_telegram_settings():
    config = load_config()
    token = request.form.get('telegram_bot_token')
    chat_id = request.form.get('telegram_chat_id')

    if 'telegram' not in config:
        config['telegram'] = {}

    config['telegram']['bot_token'] = token
    config['telegram']['chat_id'] = chat_id
    config['telegram']['enabled'] = bool(token and chat_id)
    config['telegram']['notify_on_open'] = request.form.get('notify_on_open') == 'true'
    config['telegram']['notify_on_close'] = request.form.get('notify_on_close') == 'true'

    save_config(config)
    flash("Configurações do Telegram salvas com sucesso!", "success")
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

        # Itera com um índice para garantir a modificação no objeto original
        for i, account in enumerate(accounts_list):
            if account['account_name'] == account_name:
                # Modifica o valor diretamente na lista usando o índice
                current_status = account.get('notifications_enabled', False)
                accounts_list[i]['notifications_enabled'] = not current_status
                account_found = True
                break
    except KeyError:
        # Caso a exchange ou a lista de contas não exista
        flash(f"Configuração para a exchange '{exchange_name}' não encontrada.", "error")
        return redirect(url_for('index'))

    if account_found:
        save_config(config)
    else:
        flash(f"Conta '{account_name}' não encontrada na exchange '{exchange_name}'.", "error")

    return redirect(url_for('index'))


@app.route('/process_setup_form', methods=['POST'])
def process_setup_form():
    global backtest_result_cache, bot_logs
    action = request.form.get('action')
    base_currency_form = request.form.get('base_currency').upper()
    account_name_form = request.form.get('account_name')

    if action == 'add_market':
        config = load_config()
        new_market_setup = {
            "id": str(int(time.time() * 1000)), "base_currency": base_currency_form, "quote_currency": "USDT",
            "leverage": int(request.form['leverage']), "strategy_name": request.form['strategy_name'],
            "risk_per_trade": float(request.form['risk']) / 100.0,
            "stop_loss_config": {"type": "percentage", "value": float(request.form['sl_value']) / 100.0},
            "take_profit_rrr": float(request.form['rrr']), "exit_mode": request.form.get('exit_mode', 'passivo'),
            "direction_mode": request.form.get('direction_mode', 'long_short')
        }
        tsl_type = request.form.get('tsl_type')
        if tsl_type and tsl_type != 'none':
            tsl_config = {"type": tsl_type}
            if tsl_type == 'breakeven':
                tsl_config['breakeven_trigger_rrr'] = float(request.form.get('tsl_breakeven_trigger', 1.0))
            elif tsl_type == 'atr':
                tsl_config['atr_multiple'] = float(request.form.get('tsl_atr_multiple', 2.0))
            tsl_config['remove_tp_on_trail'] = bool(request.form.get('remove_tp_on_trail'))
            new_market_setup['trailing_stop_config'] = tsl_config

        account_found = False
        for ex_data in config.get('exchanges', {}).values():
            for account in ex_data.get('accounts', []):
                if account.get('account_name') == account_name_form:
                    markets = account.get('markets_to_trade', [])
                    existing_idx = next(
                        (i for i, m in enumerate(markets) if m.get('base_currency') == base_currency_form),
                        -1)
                    if existing_idx != -1:
                        markets[existing_idx] = new_market_setup
                    else:
                        markets.append(new_market_setup)
                    account['markets_to_trade'] = markets
                    account_found = True
                    break
            if account_found:
                break
        save_config(config)
        return redirect(url_for('index'))

    elif action == 'run_backtest':
        leverage = int(request.form.get('leverage'))
        if not base_currency_form: return redirect(url_for('index'))
        symbol = f"{base_currency_form}/USDT"
        backtest_result_cache.pop(symbol, None)
        all_results = run_full_backtest(symbol, leverage)
        if all_results:
            backtest_result_cache[symbol] = all_results[0]
            df_results = pd.DataFrame(all_results)
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
            full_log_message = f"{log_header}\n{log_body}\n{'=' * len(log_header)}"
            bot_logs.append(
                {"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "account": "Backtester", "level": "INFO",
                 "message": f"<pre>{full_log_message}</pre>"})
        else:
            bot_logs.append(
                {"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "account": "Backtester", "level": "WARNING",
                 "message": f"Backtest para {symbol} não produziu resultados."})
        return redirect(url_for('index', last_backtest_symbol=symbol))


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
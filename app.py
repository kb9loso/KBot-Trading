# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import json
import threading
import time
from datetime import datetime, timedelta
import pandas as pd

from bot_logic import TradingBot
from strategies import STRATEGIES
from pacifica_client import PacificaClient
from backtester import run_full_backtest

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Variáveis globais
bot_logs = [{"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "account": "Sistema", "level": "INFO",
             "message": "Bot está parado. Clique em 'Iniciar' para começar."}]
account_info_cache = {}
dashboard_metrics_cache = {}
open_positions_cache = {}
last_refresh_error = {}
backtest_result_cache = {}
running_bots = {}


def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)


def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)


@app.route('/manage_account', methods=['POST'])
def manage_account():
    config = load_config()
    exchanges_config = config.get('exchanges', {})
    selected_exchange = request.form['exchange_name']

    # Validação de multi-conta
    exchange_details = exchanges_config.get(selected_exchange, {})
    if not exchange_details.get('multi_account_allowed', False) and len(exchange_details.get('accounts', [])) > 0:
        original_name = request.form.get('original_account_name')
        # Permite editar, mas não adicionar uma nova se já existir uma
        if not original_name or exchange_details['accounts'][0]['account_name'] != original_name:
            flash(f"A exchange '{selected_exchange}' não permite múltiplas contas.", "error")
            return redirect(url_for('index'))

    interval = int(request.form.get('check_interval', 180))
    account_details = {
        "account_name": request.form['account_name'],
        "main_public_key": request.form['main_public_key'],
        "agent_private_key": request.form['agent_private_key'],
        "check_interval_seconds": max(interval, 120)
    }

    original_name = request.form.get('original_account_name')
    accounts_in_exchange = exchange_details.get('accounts', [])

    if original_name:
        for i, acc in enumerate(accounts_in_exchange):
            if acc['account_name'] == original_name:
                account_details['markets_to_trade'] = acc.get('markets_to_trade', [])
                accounts_in_exchange[i] = account_details
                break
    else:
        account_details['markets_to_trade'] = []
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
    if not trades: return None
    closed_trades = [t for t in trades if t.get('side', '').startswith('close_') and t.get('pnl') is not None]
    if not closed_trades: return {"total_trades": 0, "total_pnl": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0,
                                  "total_fees": 0}
    winning_trades_pnl = [float(t['pnl']) for t in closed_trades if float(t['pnl']) > 0]
    losing_trades_pnl = [float(t['pnl']) for t in closed_trades if float(t['pnl']) <= 0]
    total_fees = sum(float(t.get('fee', 0.0)) for t in closed_trades)
    total_trades = len(closed_trades)
    total_pnl = sum(winning_trades_pnl) + sum(losing_trades_pnl)
    win_rate = (len(winning_trades_pnl) / total_trades) * 100 if total_trades > 0 else 0
    avg_win = sum(winning_trades_pnl) / len(winning_trades_pnl) if winning_trades_pnl else 0
    avg_loss = sum(losing_trades_pnl) / len(losing_trades_pnl) if losing_trades_pnl else 0
    return {"total_pnl": total_pnl, "total_trades": total_trades, "win_rate": win_rate, "avg_win": avg_win,
            "avg_loss": avg_loss, "total_fees": total_fees}


def fetch_and_cache_api_data(account_config: dict, exchange_name: str):
    global account_info_cache, dashboard_metrics_cache, open_positions_cache, last_refresh_error
    account_name = account_config['account_name']

    try:
        client = PacificaClient(main_public_key=account_config['main_public_key'],
                                agent_private_key=account_config['agent_private_key'])
        account_info = client.get_account_info()

        if not account_info:
            raise Exception("Falha ao buscar informações da conta.")

        account_info_cache[account_name] = account_info

        end_time = int(time.time() * 1000)
        start_time = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
        trade_history = client.get_trade_history(start_time, end_time)
        dashboard_metrics_cache[account_name] = calculate_dashboard_metrics(
            trade_history) if trade_history is not None else None

        open_positions_list = client.get_open_positions()
        open_positions_details = []
        if open_positions_list:
            all_prices = client.get_current_prices()
            all_open_orders = client.get_open_orders()
            prices_map = {p['symbol'].upper(): float(p['mark']) for p in all_prices if 'mark' in p}
            for pos in open_positions_list:
                symbol, entry_price = pos['symbol'].upper(), float(pos['entry_price'])
                current_price = prices_map.get(symbol, 0.0)
                sl_order = next((o for o in all_open_orders if
                                 o.get('symbol', '').upper() == symbol and o.get('order_type', '').startswith(
                                     'stop_loss')), None)
                tp_order = next((o for o in all_open_orders if
                                 o.get('symbol', '').upper() == symbol and o.get('order_type', '').startswith(
                                     'take_profit')), None)
                pnl_pct = ((current_price - entry_price) / entry_price) * 100 if pos.get('side') == 'bid' else ((
                                                                                                                        entry_price - current_price) / entry_price) * 100 if entry_price > 0 else 0
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
        last_refresh_error[account_name] = None
    except Exception as e:
        error_msg = f"Erro ao buscar dados para a conta {account_name}: {e}"
        print(f"ERRO FATAL ao buscar dados da API: {e}")
        last_refresh_error[account_name] = error_msg
        account_info_cache[account_name] = None
        dashboard_metrics_cache[account_name] = None
        open_positions_cache[account_name] = []


@app.route('/')
def index():
    config = load_config()
    exchanges = config.get('exchanges', {})
    exchanges_allowed = config.get('exchanges_allowed', [])
    strategy_names = list(STRATEGIES.keys())
    last_backtest_symbol = request.args.get('last_backtest_symbol')

    all_accounts = []
    for ex_name, ex_data in exchanges.items():
        for acc in ex_data.get('accounts', []):
            acc['exchange_name'] = ex_name
            all_accounts.append(acc)

    # Filtros de Log
    selected_level = request.args.get('log_level', 'all')
    selected_account = request.args.get('log_account', 'all')

    filtered_logs = bot_logs
    if selected_level != 'all':
        filtered_logs = [log for log in filtered_logs if log.get('level') == selected_level]
    if selected_account != 'all':
        filtered_logs = [log for log in filtered_logs if log.get('account') == selected_account]

    log_levels = sorted(list(set(log.get('level') for log in bot_logs)))
    log_accounts = sorted(list(set(log.get('account') for log in bot_logs if log.get('account'))))

    # Dados da conta a ser exibida (a primeira por padrão)
    displayed_account = all_accounts[0] if all_accounts else None

    # Coletando todos os setups e posições
    all_setups = []
    all_open_positions = []
    open_position_symbols = set()

    if all_accounts:
        for acc in all_accounts:
            fetch_and_cache_api_data(acc, acc['exchange_name'])

            for market in acc.get('markets_to_trade', []):
                market['account_name'] = acc['account_name']
                all_setups.append(market)

            positions = open_positions_cache.get(acc['account_name'], [])
            all_open_positions.extend(positions)
            open_position_symbols.update({pos['symbol'].upper() for pos in positions})

    return render_template(
        'index.html',
        accounts=all_accounts,
        exchanges_data=exchanges,
        exchanges_allowed=exchanges_allowed,
        running_bots=running_bots.keys(),
        displayed_account=displayed_account,
        setups_to_display=all_setups,
        account_info=account_info_cache.get(displayed_account['account_name']) if displayed_account else None,
        dashboard=dashboard_metrics_cache.get(displayed_account['account_name']) if displayed_account else None,
        error_message=last_refresh_error.get(
            displayed_account['account_name']) if displayed_account else "Nenhuma conta configurada.",
        strategy_names=strategy_names,
        last_updated=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        backtest_result=backtest_result_cache,
        last_backtest_symbol=last_backtest_symbol,
        open_position_symbols=open_position_symbols,
        open_positions_details=all_open_positions,
        logs=filtered_logs,
        log_levels=log_levels,
        log_accounts=log_accounts,
        selected_level=selected_level,
        selected_account=selected_account
    )


@app.route('/api/get_logs')
def get_logs():
    return jsonify(logs=bot_logs)


@app.route('/refresh_all', methods=['POST'])
def refresh_all():
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
    global running_bots, bot_logs
    if account_name not in running_bots:
        config = load_config()
        account_config = next(
            (acc for acc in config['exchanges'][exchange_name]['accounts'] if acc['account_name'] == account_name),
            None)
        if account_config:
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
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
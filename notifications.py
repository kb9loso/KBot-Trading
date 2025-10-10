# KBot-Trading/notifications.py
import requests
import json
import re
from database import get_db_connection


def escape_markdown(text: str) -> str:
    """Escapa caracteres especiais para o modo MarkdownV2 do Telegram."""
    if not isinstance(text, str):
        text = str(text)
    escape_chars = r'\_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)


def send_telegram_alert(message: str, bot_token: str, chat_id: str):
    """Envia uma mensagem de alerta para o Telegram."""
    if not bot_token or not chat_id:
        print("AVISO: Token do bot ou Chat ID do Telegram não configurados para envio.")
        return
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "MarkdownV2"}
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"ERRO ao enviar alerta do Telegram: {response.text}")
    except Exception as e:
        print(f"ERRO inesperado na função de alerta do Telegram: {e}")


def send_start_notification(account_name: str):
    """Envia uma notificação de inicialização do bot."""
    safe_account_name = escape_markdown(account_name)
    message = (
        f"✅ *Bot Iniciado*\n\n"
        f"A conta *{safe_account_name}* começou a ser monitorada com sucesso\\."
    )
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        telegram_config = config.get('telegram', {})
        if telegram_config.get('enabled'):
            send_telegram_alert(message, telegram_config.get('bot_token'), telegram_config.get('chat_id'))
    except Exception:
        pass


def get_notification_settings():
    """Lê o config e retorna as configurações de notificação e um mapa de permissões por conta."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)

        telegram_config = config.get('telegram', {})

        account_permissions = {}
        for exchange_data in config.get('exchanges', {}).values():
            for account in exchange_data.get('accounts', []):
                account_permissions[account['account_name']] = account.get('notifications_enabled', False)

        return telegram_config, account_permissions
    except Exception as e:
        print(f"ERRO ao ler o arquivo de configuração para notificações: {e}")
        return None, None


def check_and_send_open_alerts():
    """Verifica o DB por posições abertas e envia alertas se a conta tiver permissão."""
    telegram_config, account_permissions = get_notification_settings()
    if not (telegram_config and telegram_config.get('notify_on_open') and any(account_permissions.values())):
        return

    print("Verificando por posições abertas para alertar...")
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            order_id, account_name, exchange, symbol, side, SUM(amount) as total_amount, MAX(DATETIME(created_at, 'localtime')) as created_at
        FROM trade_history
        WHERE (side = 'open_long' OR side = 'open_short') AND alert_sent = 0 AND exchange != 'apex'
        GROUP BY order_id, account_name, exchange, symbol, side
    """)
    orders_to_alert = cursor.fetchall()

    if not orders_to_alert:
        print("Nenhuma nova posição aberta para alertar.")
        conn.close()
        return

    alerts_sent_count = 0
    for order in orders_to_alert:
        order_id, account_name, exchange, symbol, side, total_amount, created_at = order

        if account_permissions.get(account_name):
            trade_type = "Long" if side == 'open_long' else "Short"
            message = (
                f"🚀 *Nova Posição Aberta*\n\n"
                f"👤 *Conta:* {escape_markdown(account_name)}\n"
                f"📦 *Exchange:* {escape_markdown(exchange.capitalize())}\n"
                f"📈 *Ativo:* {escape_markdown(symbol)}\n"
                f"🧭 *Tipo:* {escape_markdown(trade_type)}\n"
                f"💰 *Quantidade:* {escape_markdown(f'{total_amount:.8f}')}\n"
                f"📅 *Data:* {escape_markdown(created_at)}\n"
                f"🆔 *Order ID:* `{order_id}`"
            )
            print(f"Enviando alerta de abertura para a ordem {order_id} da conta {account_name}...")
            send_telegram_alert(message, telegram_config.get('bot_token'), telegram_config.get('chat_id'))
            alerts_sent_count += 1
            cursor.execute("UPDATE trade_history SET alert_sent = 1 WHERE order_id = ?", (order_id,))

    conn.commit()
    conn.close()
    if alerts_sent_count > 0:
        print(f"{alerts_sent_count} alertas de posições abertas foram enviados.")


def check_and_send_close_alerts():
    """Verifica o DB por posições fechadas e envia alertas se a conta tiver permissão."""
    telegram_config, account_permissions = get_notification_settings()
    if not (telegram_config and telegram_config.get('notify_on_close') and any(account_permissions.values())):
        return

    print("Verificando por posições fechadas para alertar...")
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            order_id, account_name, exchange, symbol, side, SUM(pnl) as total_pnl, MAX(DATETIME(created_at, 'localtime')) as created_at
        FROM trade_history
        WHERE (side = 'close_long' OR side = 'close_short') AND alert_sent = 0 AND datetime(created_at) >= datetime('now', '-1 day') AND exchange != 'apex'
        GROUP BY order_id, account_name, exchange, symbol, side
    """)
    orders_to_alert = cursor.fetchall()

    if not orders_to_alert:
        print("Nenhuma nova posição fechada para alertar.")
        conn.close()
        return

    alerts_sent_count = 0
    for order in orders_to_alert:
        order_id, account_name, exchange, symbol, side, total_pnl, created_at = order

        if order_id is None: continue

        if account_permissions.get(account_name):
            pnl_value = float(total_pnl)
            status_icon = "✅" if pnl_value >= 0 else "❌"
            trade_type = "Long" if side == 'close_long' else "Short"
            pnl_formatted = f"{pnl_value:+.2f}"
            message = (
                f"{status_icon} *Posição Fechada*\n\n"
                f"👤 *Conta:* {escape_markdown(account_name)}\n"
                f"📦 *Exchange:* {escape_markdown(exchange.capitalize())}\n"
                f"📈 *Ativo:* {escape_markdown(symbol)}\n"
                f"🧭 *Tipo:* {escape_markdown(trade_type)}\n"
                f"💰 *PNL Total:* `${escape_markdown(pnl_formatted)}`\n"
                f"📅 *Data:* {escape_markdown(created_at)}\n"
                f"🆔 *Order ID:* `{order_id}`"
            )
            print(f"Enviando alerta consolidado para a ordem {order_id} da conta {account_name}...")
            send_telegram_alert(message, telegram_config.get('bot_token'), telegram_config.get('chat_id'))
            alerts_sent_count += 1
            cursor.execute("UPDATE trade_history SET alert_sent = 1 WHERE order_id = ?", (order_id,))

    conn.commit()
    conn.close()
    if alerts_sent_count > 0:
        print(f"{alerts_sent_count} alertas de ordens fechadas foram enviados.")


def check_and_send_close_alerts_apex():
    """Verifica a tabela pnl_history_apex por posições fechadas e envia alertas."""
    telegram_config, account_permissions = get_notification_settings()

    if not (telegram_config and telegram_config.get('notify_on_close')):
        return
    if not any(account_permissions.values()):
        return

    print("Verificando por posições fechadas da APEX para alertar...")
    conn = get_db_connection()
    cursor = conn.cursor()

    # Query na nova tabela para registros onde o alerta ainda não foi enviado
    cursor.execute("""
        SELECT
            id, account_name, symbol, side, totalPnl, createdAt
        FROM
            pnl_history_apex
        WHERE
            alert_sent = 0
            AND type = 'CLOSE_POSITION'
    """)
    records_to_alert = cursor.fetchall()

    if not records_to_alert:
        print("Nenhuma nova posição fechada da APEX para alertar.")
        conn.close()
        return

    alerts_sent_count = 0
    for record in records_to_alert:
        record_id, account_name, symbol, side, total_pnl, created_at = record

        # Verifica a permissão da conta específica
        if account_permissions.get(account_name):
            pnl_value = float(total_pnl)
            status_icon = "✅" if pnl_value >= 0 else "❌"
            trade_type = "Long" if side == 'LONG' else "Short"
            pnl_formatted = f"{pnl_value:+.2f}"

            message = (
                f"{status_icon} *Posição Fechada (Apex)*\n\n"
                f"👤 *Conta:* {escape_markdown(account_name)}\n"
                f"📈 *Ativo:* {escape_markdown(symbol)}\n"
                f"🧭 *Tipo:* {escape_markdown(trade_type)}\n"
                f"💰 *PNL Total:* `${escape_markdown(pnl_formatted)}`\n"
                f"📅 *Data:* {escape_markdown(created_at)}\n"
                f"🆔 *Record ID:* `{record_id}`"
            )
            print(f"Enviando alerta para o registro {record_id} da conta {account_name}...")
            send_telegram_alert(message, telegram_config.get('bot_token'), telegram_config.get('chat_id'))
            alerts_sent_count += 1

            # Marca o registro como alertado no banco de dados
            cursor.execute("UPDATE pnl_history_apex SET alert_sent = 1 WHERE id = ?", (record_id,))

    conn.commit()
    conn.close()
    if alerts_sent_count > 0:
        print(f"{alerts_sent_count} alertas de posições fechadas da Apex foram enviados.")
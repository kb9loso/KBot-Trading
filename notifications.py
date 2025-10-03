# KBot-Trading/notifications.py
import requests
import json
# Precisamos de acesso à base de dados para fazer a verificação
from database import get_db_connection


def send_telegram_alert(message: str):
    """Envia uma mensagem de alerta para o Telegram."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)

        telegram_config = config.get('telegram', {})
        if not telegram_config.get('enabled'):
            return

        bot_token = telegram_config.get('bot_token')
        chat_id = telegram_config.get('chat_id')

        if not bot_token or not chat_id:
            print("AVISO: Token do bot ou Chat ID do Telegram não configurados.")
            return

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "MarkdownV2"}

        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"ERRO ao enviar alerta do Telegram: {response.text}")

    except Exception as e:
        print(f"ERRO inesperado na função de alerta do Telegram: {e}")


def send_start_notification(account_name: str):
    """Envia uma notificação de inicialização do bot."""
    message = (
        f"✅ *Bot Iniciado*\n\n"
        f"A conta *{account_name}* começou a ser monitorada com sucesso\\."
    )
    send_telegram_alert(message)


def check_and_send_close_alerts():
    """
    Verifica o DB por posições fechadas (agrupadas por order_id)
    que ainda não foram alertadas e envia um alerta consolidado.
    """
    print("Verificando por posições fechadas para alertar...")

    # Carrega a configuração para verificar quais contas têm notificações ativadas
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)

        accounts_with_notifications = {
            acc['account_name'] for ex in config.get('exchanges', {}).values()
            for acc in ex.get('accounts', []) if acc.get('notifications_enabled')
        }
    except Exception as e:
        print(f"ERRO ao ler a configuração para verificar notificações: {e}")
        return

    if not accounts_with_notifications:
        print("Nenhuma conta com notificações ativadas.")
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    # Query que agrupa todos os trades de fecho por ordem e soma o PNL
    cursor.execute("""
            SELECT
                order_id, account_name, exchange, symbol, side, SUM(pnl) as total_pnl
            FROM
                trade_history
            WHERE
                (side = 'close_long' OR side = 'close_short')
                AND alert_sent = 0
                AND datetime(created_at) >= datetime('now', '-1 day')
            GROUP BY
                order_id, account_name, exchange, symbol, side
        """)
    orders_to_alert = cursor.fetchall()

    if not orders_to_alert:
        print("Nenhuma nova posição fechada para alertar.")
        conn.close()
        return

    alerts_sent_count = 0
    for order in orders_to_alert:
        order_id, account_name, exchange, symbol, side, total_pnl = order

        if order_id is None or account_name not in accounts_with_notifications:
            continue

        pnl_value = float(total_pnl)
        status_icon = "✅" if pnl_value >= 0 else "❌"
        trade_type = "Long" if side == 'close_long' else "Short"

        pnl_formatted = f"{pnl_value:+.2f}".replace('.', ',')
        symbol_formatted = symbol.replace('-', '\\-')

        message = (
            f"{status_icon} *Posição Fechada*\n\n"
            f"👤 *Conta:* {account_name}\n"
            f"📦 *Exchange:* {exchange.capitalize()}\n"
            f"📈 *Ativo:* {symbol_formatted}\n"
            f"🧭 *Tipo:* {trade_type}\n"
            f"💰 *PNL Total:* `${pnl_formatted}`\n"
            f"🆔 *Order ID:* `{order_id}`"
        )

        print(f"Enviando alerta consolidado para a ordem {order_id} da conta {account_name}...")
        send_telegram_alert(message)
        alerts_sent_count += 1

        # Marca TODOS os trades associados a esta order_id como alertados
        cursor.execute("UPDATE trade_history SET alert_sent = 1 WHERE order_id = ?", (order_id,))

    conn.commit()
    conn.close()
    if alerts_sent_count > 0:
        print(f"{alerts_sent_count} alertas de ordens fechadas foram enviados.")
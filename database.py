# KBot-Trading/database.py
import sqlite3
import json
from datetime import datetime, timedelta

DATABASE_FILE = 'trading_history.db'


def get_db_connection():
    conn = sqlite3.connect(DATABASE_FILE, timeout=10)
    return conn


def init_db():
    """Inicializa o banco de dados e cria todas as tabelas se não existirem."""
    print("Verificando e inicializando o banco de dados...")
    conn = get_db_connection()
    cursor = conn.cursor()

    # ... (tabela successful_orders) ...
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS successful_orders (
            order_id INTEGER PRIMARY KEY,
            timestamp_utc TEXT NOT NULL,
            account_name TEXT NOT NULL,
            exchange TEXT NOT NULL,
            ativo TEXT NOT NULL,
            lado TEXT NOT NULL,
            quantidade REAL NOT NULL,
            strategy_name TEXT NOT NULL,
            leverage INTEGER NOT NULL,
            risk_per_trade REAL NOT NULL,
            take_profit_rrr REAL NOT NULL,
            direction_mode TEXT,
            exit_mode TEXT,
            stop_loss_config TEXT,
            trailing_stop_config TEXT,
            preco_entrada_aprox REAL NOT NULL,
            stop_loss_price REAL NOT NULL,
            take_profit_price REAL NOT NULL,
            full_payload_json TEXT
        );
    ''')

    cursor.execute('''
            CREATE TABLE IF NOT EXISTS order_history (
                order_id INTEGER PRIMARY KEY,
                exchange TEXT NOT NULL,
                account_name TEXT NOT NULL,
                client_order_id TEXT,
                symbol TEXT,
                side TEXT,
                initial_price REAL,
                average_filled_price REAL,
                amount REAL,
                filled_amount REAL,
                order_status TEXT,
                order_type TEXT,
                stop_price REAL,
                stop_parent_order_id INTEGER,
                reduce_only INTEGER,
                reason TEXT,
                created_at TEXT, 
                updated_at TEXT
            );
        ''')

    cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                history_id INTEGER PRIMARY KEY,
                exchange TEXT NOT NULL,
                account_name TEXT NOT NULL,
                order_id INTEGER,
                client_order_id TEXT,
                symbol TEXT,
                amount REAL,
                price REAL,
                entry_price REAL,
                fee REAL,
                pnl REAL,
                event_type TEXT,
                side TEXT,
                created_at TEXT,
                cause TEXT,
                alert_sent INTEGER DEFAULT 0
            );
        ''')
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS pnl_history_apex (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_name TEXT NOT NULL,
                symbol TEXT,
                size REAL,
                exitPrice REAL,
                price REAL,
                side TEXT,
                totalPnl REAL,
                createdAt TEXT,
                type TEXT,
                isLiquidate BOOLEAN,
                isDeleverage BOOLEAN,
                fee REAL,
                closeSharedOpenFee REAL,
                liquidateFee REAL,
                exitType TEXT,
                closeSharedFundingFee REAL,
                closeSharedOpenValue REAL,
                alert_sent INTEGER DEFAULT 0,
                UNIQUE(account_name, symbol, createdAt)
            );
        ''')
    conn.commit()
    conn.close()
    print("Banco de dados pronto.")


def _ms_to_utc_string(ms):
    """Converte timestamp em milissegundos para uma string de data/hora UTC, se não for nulo."""
    if ms is None:
        return None
    # Usa utcfromtimestamp para garantir que a conversão é sempre em UTC, sem influência do fuso local.
    return datetime.utcfromtimestamp(ms / 1000.0).strftime('%Y-%m-%d %H:%M:%S')


def sync_order_history(client, exchange_name: str, account_name: str):
    """Busca o histórico de ordens da API e atualiza o banco de dados."""
    print(f"Iniciando sincronização do histórico de ORDENS para a conta {account_name}...")
    orders = client.get_order_history()
    if not orders:
        print("Nenhuma ordem encontrada ou erro na API.")
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    for order in orders:
        cursor.execute('''
            INSERT OR IGNORE INTO order_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            order.get('order_id'), exchange_name, account_name, order.get('client_order_id'), order.get('symbol'),
            order.get('side'), float(order.get('initial_price', 0)), float(order.get('average_filled_price', 0)),
            float(order.get('amount', 0)), float(order.get('filled_amount', 0)), order.get('order_status'),
            order.get('order_type'), order.get('stop_price'), order.get('stop_parent_order_id'),
            1 if order.get('reduce_only') else 0, order.get('reason'),
            _ms_to_utc_string(order.get('created_at')),
            _ms_to_utc_string(order.get('updated_at'))
        ))

    conn.commit()
    conn.close()
    print(f"Sincronização de {len(orders)} ordens concluída.")


def sync_trade_history(client, exchange_name: str, account_name: str, start_time_ms: int):
    print(f"Iniciando sincronização de TRADES para a conta {account_name}...")
    end_time = int(datetime.now().timestamp() * 1000)
    trades = client.get_trade_history(start_time_ms, end_time)
    if not trades:
        print(f"Nenhum trade novo encontrado para a conta {account_name}.")
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    for trade in trades:
        # A query agora insere/atualiza mantendo o valor antigo de alert_sent se o registo já existir
        cursor.execute('''
        INSERT OR IGNORE INTO trade_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.get('history_id'), exchange_name, account_name, trade.get('order_id'), trade.get('client_order_id'),
            trade.get('symbol'), float(trade.get('amount', 0)), float(trade.get('price', 0)),
            float(trade.get('entry_price', 0)), float(trade.get('fee', 0)), float(trade.get('pnl', 0)),
            trade.get('event_type'), trade.get('side'), _ms_to_utc_string(trade.get('created_at')),
            trade.get('cause'), 0
        ))

    conn.commit()
    conn.close()
    print(f"Sincronização de {len(trades)} trades para a conta {account_name} concluída.")



def insert_successful_order(order_details: dict):
    """Insere uma nova ordem bem-sucedida no banco de dados de forma transacional."""
    sql = '''
        INSERT OR IGNORE INTO successful_orders (
            order_id, timestamp_utc, account_name, exchange, ativo, lado, quantidade,
            strategy_name, leverage, risk_per_trade, take_profit_rrr, direction_mode, exit_mode,
            stop_loss_config, trailing_stop_config,
            preco_entrada_aprox, stop_loss_price, take_profit_price,
            full_payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    params = (
        order_details.get('order_id'),
        order_details.get('timestamp_utc'),
        order_details.get('account_name'),
        order_details.get('exchange'),
        order_details.get('ativo'),
        order_details.get('lado'),
        float(order_details.get('quantidade', 0)),
        order_details.get('strategy_name'),
        order_details.get('leverage'),
        order_details.get('risk_per_trade'),
        order_details.get('take_profit_rrr'),
        order_details.get('direction_mode'),
        order_details.get('exit_mode'),
        json.dumps(order_details.get('stop_loss_config')),
        json.dumps(order_details.get('trailing_stop_config')),
        float(order_details.get('preco_entrada_aprox', 0)),
        float(order_details.get('stop_loss_price', 0)),
        float(order_details.get('take_profit_price', 0)),
        json.dumps(order_details.get('full_payload'))
    )

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
    except sqlite3.Error as e:
        print(f"ERRO de SQLite ao inserir ordem: {e}")
    except Exception as e:
        print(f"ERRO inesperado ao inserir ordem no DB: {e}")


def sync_pnl_history(client, account_name: str, start_time_ms: int):
    """Busca o histórico de PNL da Apex e insere na tabela pnl_history_apex."""
    print(f"Iniciando sincronização de PNL HISTÓRICO para a conta {account_name}...")
    end_time_ms = int(datetime.now().timestamp() * 1000)
    pnl_history = client.get_pnl_history(start_time_ms, end_time_ms)

    if not pnl_history:
        print(f"Nenhum novo registro de PNL encontrado para a conta {account_name}.")
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    new_records_count = 0
    for record in pnl_history:
        created_at_str = _ms_to_utc_string(record.get('createdAt'))

        # --- INÍCIO DA CORREÇÃO ---
        # Adiciona o campo 'alert_sent' na inserção
        cursor.execute('''
            INSERT OR IGNORE INTO pnl_history_apex 
            (account_name, symbol, size, exitPrice, price, side, totalPnl, createdAt, type, isLiquidate, 
            isDeleverage, fee, closeSharedOpenFee, liquidateFee, exitType, closeSharedFundingFee, closeSharedOpenValue, alert_sent) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            account_name, record.get('symbol'), float(record.get('size') or 0.0),
            float(record.get('exitPrice') or 0.0), float(record.get('price') or 0.0),
            record.get('side'), float(record.get('totalPnl') or 0.0), created_at_str,
            record.get('type'), record.get('isLiquidate'), record.get('isDeleverage'),
            float(record.get('fee') or 0.0), float(record.get('closeSharedOpenFee') or 0.0),
            float(record.get('liquidateFee') or 0.0), record.get('exitType'),
            float(record.get('closeSharedFundingFee') or 0.0),
            float(record.get('closeSharedOpenValue') or 0.0),
            0  # Valor padrão para alert_sent
        ))
        # --- FIM DA CORREÇÃO ---
        if cursor.rowcount > 0:
            new_records_count += 1

    conn.commit()
    conn.close()
    if new_records_count > 0:
        print(f"Sincronização de {new_records_count} novos registros de PNL para a conta {account_name} concluída.")

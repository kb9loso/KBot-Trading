# websocket_manager.py
from datetime import datetime

import ccxt.pro as ccxtpro
import asyncio
import pandas as pd
import time
import logging
import threading
from collections import deque, defaultdict
from data_fetcher import get_historical_klines

log = logging.getLogger("WSManager")


class WebSocketManager:
    SUPPORTED_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    KLINE_CACHE_LIMIT = 500

    def __init__(self):  # Recebe max_min_candles
        self._exchanges = {}
        self._cache = {}
        self._cache_maxlen = defaultdict(lambda: 500)
        self._subscriptions = defaultdict(int)
        self._running = False
        self._loop = None
        self._thread = None
        self._lock = threading.Lock()
        self._initial_subscriptions = set()
        self._active_tasks = {}

        # ADICIONADO: Para callbacks (Meta 4)
        self._event_listeners = defaultdict(list)

        try:
            # Passa apenas 'options' se necessário, ou vazio {}
            self._exchanges['binance'] = ccxtpro.binanceusdm({'options': {'defaultType': 'future'}})
            log.info("Instância ccxt.pro Binance (USD-M Futures) criada.")
        except Exception as e:
            log.error(f"Erro ao inicializar ccxt.pro Binance: {e}")
        try:
            # Passa apenas 'options' se necessário, ou vazio {}
            self._exchanges['bybit'] = ccxtpro.bybit({'options': {'defaultType': 'future'}})
            log.info("Instância ccxt.pro Bybit criada.")
        except Exception as e:
            log.error(f"Erro ao inicializar ccxt.pro Bybit: {e}")

        if not self._exchanges:
            log.critical("Nenhuma instância de exchange ccxt.pro pôde ser criada.")

    def is_running(self) -> bool:
        """Verifica se o loop principal do manager está ativo."""
        return self._running and self._thread is not None and self._thread.is_alive()

    def set_initial_subscriptions(self, initial_pairs: set):
        """Define os pares (symbol, timeframe) que devem ser monitorados desde o início."""
        with self._lock:
            # O formato esperado em initial_pairs é {(symbol_formato_ws, timeframe), ...}
            # Ex: {('BTC-USDT', '15m'), ('ETH-USDT', '1h')}
            self._initial_subscriptions = initial_pairs
            log.info(f"Subscrições iniciais definidas: {len(self._initial_subscriptions)} pares.")
            # Zera o contador _subscriptions, ele será populado pelos bots ao iniciarem
            self._subscriptions.clear()

    # ADICIONADO: (Meta 1 e 3)
    def populate_all_initial_caches(self, pairs_to_populate: dict) -> set:
        """
        Itera sobre o dicionário de pares necessários e chama
        populate_initial_cache para cada um, tratando exceções.

        Retorna:
            set: Um set com as cache_keys (symbol, timeframe)
                 que foram populadas com sucesso (ou já existiam).
        """
        log.info(f"Iniciando população de cache para {len(pairs_to_populate)} pares...")
        successful_keys = set()

        for (symbol, timeframe), min_candles in pairs_to_populate.items():
            cache_key = (symbol, timeframe)
            try:
                # Reusa a lógica de população existente
                self.populate_initial_cache(symbol, timeframe, min_candles)

                # Verifica se a população realmente funcionou
                with self._lock:
                    if cache_key in self._cache and len(self._cache[cache_key]) >= min_candles:
                        successful_keys.add(cache_key)
                        log.info(f"Cache {cache_key} populado com sucesso.")
                    else:
                        log.error(
                            f"Cache {cache_key} não atingiu o mínimo de candles ({len(self._cache.get(cache_key, []))}/{min_candles}) após população.")
            except Exception as e:
                # Meta 3: Loga o erro e continua
                log.error(f"Falha CRÍTICA ao popular cache inicial para {cache_key}: {e}", exc_info=True)

        log.info(
            f"População de cache inicial concluída. {len(successful_keys)}/{len(pairs_to_populate)} bem-sucedidos.")
        return successful_keys

    def populate_initial_cache(self, symbol: str, timeframe: str, min_candles: int, primary_exchange: str = 'binance',
                               fallback_exchange: str = 'bybit'):
        """Busca dados históricos e preenche o cache para um par/timeframe."""
        cache_key = (symbol, timeframe)
        # Define o maxlen para este cache_key (min_candles + buffer)
        # Garante um mínimo (ex: 50)
        current_maxlen = max(50, min_candles + 20)
        with self._lock:
            self._cache_maxlen[cache_key] = current_maxlen

        # Busca um pouco mais que o necessário para o cache
        limit = current_maxlen + 30  # Ajuste o buffer de busca se necessário
        ccxt_market_symbol = symbol.replace('-', '/')

        log.info(
            f"Populando cache inicial {cache_key} (buscando {limit}, min_req: {min_candles}, maxlen: {current_maxlen})...")

        historical_data = get_historical_klines(market=ccxt_market_symbol, timeframe=timeframe, limit=limit,
                                                exchange_name=primary_exchange)
        if historical_data is None or len(historical_data) < min_candles:  # Verifica contra min_candles
            log.warning(
                f"Falha/dados insuficientes ({len(historical_data) if historical_data is not None else 0}/{min_candles}) da {primary_exchange} para {cache_key}. Tentando {fallback_exchange}...")
            if fallback_exchange and fallback_exchange != primary_exchange:
                historical_data = get_historical_klines(market=ccxt_market_symbol, timeframe=timeframe, limit=limit,
                                                        exchange_name=fallback_exchange)
                if historical_data is None or len(historical_data) < min_candles:  # Verifica contra min_candles
                    log.error(
                        f"Falha/dados insuficientes ({len(historical_data) if historical_data is not None else 0}/{min_candles}) também da {fallback_exchange} para {cache_key}. Cache pode estar incompleto.")
                    # Continua mesmo assim, mas pode não ter o mínimo

        if historical_data is not None and not historical_data.empty:
            try:
                df_processed = historical_data.copy()
                required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

                # 1. Handle Index -> 'timestamp' column
                if isinstance(df_processed.index, pd.DatetimeIndex):
                    df_processed.reset_index(inplace=True)
                    # Ensure the new column is named 'timestamp'
                    if 'index' in df_processed.columns and 'timestamp' not in df_processed.columns:
                        df_processed.rename(columns={'index': 'timestamp'}, inplace=True)
                    elif 'date' in df_processed.columns and 'timestamp' not in df_processed.columns:
                        df_processed.rename(columns={'date': 'timestamp'}, inplace=True)
                    # Add a check if 'timestamp' column is still missing after reset
                    elif 'timestamp' not in df_processed.columns:
                        raise ValueError(
                            f"Coluna 'timestamp' não encontrada após reset_index para {cache_key}. Colunas: {df_processed.columns.tolist()}")

                # 2. Verify all required columns are present (NOW including 'timestamp')
                missing_cols = [col for col in required_cols if col not in df_processed.columns]
                if missing_cols:
                    # If only timestamp is missing AFTER reset, something is very wrong
                    if missing_cols == ['timestamp'] and not isinstance(historical_data.index, pd.DatetimeIndex):
                        # Maybe the first column IS the timestamp? Try renaming if numeric.
                        if len(df_processed.columns) > 0 and pd.api.types.is_numeric_dtype(df_processed.iloc[:, 0]):
                            log.warning(f"Tentando usar a primeira coluna como 'timestamp' para {cache_key}")
                            df_processed.rename(columns={df_processed.columns[0]: 'timestamp'}, inplace=True)
                            # Re-check for missing cols
                            missing_cols = [col for col in required_cols if col not in df_processed.columns]
                            if missing_cols:  # If still missing after trying first col
                                raise ValueError(
                                    f"Dados {cache_key} sem colunas: {missing_cols}. Colunas: {df_processed.columns.tolist()}")
                        else:  # First column not numeric, cannot assume it's timestamp
                            raise ValueError(
                                f"Dados {cache_key} sem coluna 'timestamp' e índice não é DatetimeIndex. Colunas: {df_processed.columns.tolist()}")
                    else:  # Other columns missing
                        raise ValueError(
                            f"Dados {cache_key} sem colunas: {missing_cols}. Colunas: {df_processed.columns.tolist()}")

                # 3. Convert 'timestamp' to milliseconds int64
                if pd.api.types.is_datetime64_any_dtype(df_processed['timestamp']):
                    df_processed['timestamp'] = df_processed['timestamp'].astype('int64') // 1_000_000
                elif pd.api.types.is_numeric_dtype(df_processed['timestamp']):
                    df_processed['timestamp'] = df_processed['timestamp'].astype('int64')  # Assume already ms
                else:
                    raise TypeError(f"Coluna 'timestamp' não é datetime nem numérica para {cache_key}")

                # 4. Select and Sort required columns before converting to tuples
                df_final = df_processed[required_cols].sort_values(by='timestamp')
                historical_tuples = [tuple(x) for x in df_final.to_numpy()]

                with self._lock:
                    # Usa current_maxlen definido antes
                    current_maxlen = self._cache_maxlen.get(cache_key, 500)
                    self._cache[cache_key] = deque(historical_tuples, maxlen=current_maxlen)
                    log.info(
                        f"Cache {cache_key} populado com {len(historical_tuples)} candles (maxlen: {current_maxlen}). Último TS: {historical_tuples[-1][0]}")

            except Exception as e:
                log.error(f"Erro ao processar/armazenar dados históricos {cache_key}: {e}", exc_info=True)

    # ADICIONADO: (Meta 4)
    def register_listener(self, cache_key: tuple[str, str], callback):
        """Registra uma função de callback para ser chamada em um novo candle."""
        with self._lock:
            if callback not in self._event_listeners[cache_key]:
                self._event_listeners[cache_key].append(callback)
                log.info(f"Listener {callback.__name__} registrado para {cache_key}")

    # ADICIONADO: (Meta 4)
    def unregister_listener(self, cache_key: tuple[str, str], callback):
        """Remove uma função de callback."""
        with self._lock:
            try:
                self._event_listeners[cache_key].remove(callback)
                log.info(f"Listener {callback.__name__} removido de {cache_key}")
                if not self._event_listeners[cache_key]:
                    del self._event_listeners[cache_key]
            except ValueError:
                log.warning(
                    f"Tentativa de remover listener {callback.__name__} de {cache_key}, mas não foi encontrado.")
            except KeyError:
                pass  # Chave já não existe, tudo bem

    async def _watch_ohlcv(self, exchange_id, symbol, timeframe):
        """Assiste ao stream OHLCV de uma exchange para um par/timeframe."""
        exchange = self._exchanges.get(exchange_id)
        if not exchange:
            log.error(f"[{exchange_id.upper()}] Exchange não inicializada.")
            return

        ccxt_symbol = symbol.replace('-', '/')  # Converte formato interno para ccxt (BTC-USDT -> BTC/USDT)
        cache_key = (symbol, timeframe)  # Usa formato interno (BTC-USDT, 15m)

        log.info(f"[{exchange_id.upper()}] Iniciando watch_ohlcv para {ccxt_symbol} ({timeframe})...")

        # Flag para controlar tentativa de fallback
        fallback_attempted = False
        fallback_exchange_id = 'bybit' if exchange_id == 'binance' else 'binance'

        while self._running:
            # Verifica se ainda precisa desta subscrição antes de tentar conectar/receber
            with self._lock:
                is_needed = cache_key in self._initial_subscriptions or self._subscriptions.get(cache_key, 0) > 0
            if not is_needed:
                log.warning(
                    f"[{exchange_id.upper()}] Subscrição para {cache_key} não é mais necessária. Encerrando watch.")
                break  # Sai do loop while se não for mais necessário

            try:
                ohlcvs = await exchange.watch_ohlcv(ccxt_symbol, timeframe)
                if not self._running: break  # Verifica novamente após await

                with self._lock:
                    # Pega o maxlen específico ou usa default
                    current_maxlen = self._cache_maxlen.get(cache_key, 500)
                    if cache_key not in self._cache:
                        log.info(
                            f"[{exchange_id.upper()}] Cache {cache_key} não existia. Criando deque com maxlen={current_maxlen}.")
                        self._cache[cache_key] = deque(maxlen=current_maxlen)  # Usa maxlen específico

                    cache_deque = self._cache[cache_key]
                    new_candle_added = False  # Flag para saber se um candle NOVO foi adicionado
                    updated_candle_ts = None  # Guarda o TS do candle atualizado (se houver)

                    for candle in ohlcvs:
                        timestamp_ms = candle[0]
                        if cache_deque and timestamp_ms < cache_deque[-1][0]:
                            continue
                        elif cache_deque and timestamp_ms == cache_deque[-1][0]:
                            cache_deque[-1] = tuple(candle)
                            updated_candle_ts = timestamp_ms  # Marca que houve atualização
                        else:
                            cache_deque.append(tuple(candle))
                            new_candle_added = True  # Marca que houve adição
                            updated_candle_ts = timestamp_ms  # Guarda o TS do novo candle

                    # MODIFICAÇÃO (Meta 4): Disparar callbacks
                    if new_candle_added and updated_candle_ts is not None:
                        try:
                            dt_object = datetime.fromtimestamp(updated_candle_ts / 1000)
                            readable_ts = dt_object.strftime('%Y-%m-%d %H:%M:%S')
                            log.info(
                                f"[{exchange_id.upper()}] Novo candle adicionado {cache_key}: {readable_ts} ({updated_candle_ts})")
                        except:
                            log.info(
                                f"[{exchange_id.upper()}] Novo candle adicionado {cache_key}. TS: {updated_candle_ts}")

                        # --- DISPARA EVENTOS ---
                        callbacks = self._event_listeners.get(cache_key, [])
                        if callbacks:
                            log.debug(f"Disparando {len(callbacks)} callbacks para {cache_key} (em threads)")
                            for cb in callbacks:
                                try:
                                    # Executa o callback em uma nova thread para não bloquear o loop asyncio
                                    threading.Thread(
                                        target=cb,
                                        args=(cache_key,),
                                        daemon=True,
                                        name=f"Callback-{cache_key[0]}"
                                    ).start()
                                except Exception as e:
                                    log.error(f"Falha ao iniciar thread de callback para {cache_key}: {e}")

                    # Loga como DEBUG se apenas atualizou um candle existente
                    elif updated_candle_ts is not None:
                        try:
                            dt_object = datetime.fromtimestamp(updated_candle_ts / 1000)
                            readable_ts = dt_object.strftime('%Y-%m-%d %H:%M:%S')
                            log.debug(
                                f"[{exchange_id.upper()}] Candle atualizado {cache_key}: {readable_ts} ({updated_candle_ts})")
                        except:
                            log.debug(f"[{exchange_id.upper()}] Candle atualizado {cache_key}. TS: {updated_candle_ts}")

            except (asyncio.CancelledError):
                log.warning(f"[{exchange_id.upper()}] Task watch_ohlcv para {cache_key} cancelada.")
                break  # Sai do loop se a task foi cancelada externamente
            except ccxtpro.NetworkError as e:
                log.error(f"[{exchange_id.upper()}] Erro de rede {cache_key}: {e}. Tentando reconectar...")
                await asyncio.sleep(5)
            except ccxtpro.ExchangeError as e:
                log.error(f"[{exchange_id.upper()}] Erro da exchange {cache_key}: {e}")
                # Lógica de Fallback
                if not fallback_attempted and fallback_exchange_id in self._exchanges:
                    log.warning(
                        f"[{exchange_id.upper()}] Tentando fallback para {fallback_exchange_id} para {cache_key} devido a erro.")
                    fallback_key = (symbol, timeframe, fallback_exchange_id)
                    with self._lock:
                        # Se já não houver uma task de fallback rodando
                        if fallback_key not in self._active_tasks:
                            # Marca que tentou
                            fallback_attempted = True
                            # Cria a task de fallback (será gerenciada no _run_loop)
                            log.info(f"Agendando task de fallback para {fallback_key}")
                            # Não inicia a task aqui, apenas sinaliza a necessidade dela
                            # O _run_loop vai pegar na próxima iteração
                        else:
                            log.info(f"Task de fallback {fallback_key} já existe ou está sendo iniciada.")
                            fallback_attempted = True  # Marca como tentado mesmo assim
                    # Se tentou fallback, encerra esta task primária
                    if fallback_attempted:
                        log.warning(
                            f"[{exchange_id.upper()}] Encerrando task primária para {cache_key} após tentativa de fallback.")
                        break
                # Se já tentou fallback ou não há fallback, espera e tenta de novo ou desiste
                if 'symbol' in str(e).lower() or 'pair' in str(e).lower() or 'not supported' in str(e).lower():
                    log.error(
                        f"[{exchange_id.upper()}] Símbolo/Timeframe {cache_key} provavelmente inválido/não suportado. Desistindo desta task.")
                    break  # Desiste permanentemente para esta task
                await asyncio.sleep(30)  # Espera para outros erros
            except Exception as e:
                log.error(f"[{exchange_id.upper()}] Erro inesperado {cache_key}: {e}", exc_info=True)
                await asyncio.sleep(10)

        log.warning(f"[{exchange_id.upper()}] Conexão watch_ohlcv para {ccxt_symbol} ({timeframe}) encerrada.")
        # Remove a si mesma do dict de tasks ativas ao sair
        task_key = (symbol, timeframe, exchange_id)
        with self._lock:
            if task_key in self._active_tasks:
                del self._active_tasks[task_key]
                log.info(f"Task {task_key} removida do registro ativo.")

    async def _run_loop(self):
        """O loop principal do asyncio que gerencia as tasks de watch_ohlcv."""
        log.info("Loop asyncio do WebSocketManager iniciado.")
        # self._active_tasks = {} # Movido para __init__

        while self._running:
            needed_pairs = set()
            with self._lock:
                # Combina subscrições iniciais com as ativas (bots rodando)
                needed_pairs = self._initial_subscriptions.union(
                    set(pair for pair, count in self._subscriptions.items() if count > 0)
                )

            # --- Gerencia Tasks ---
            current_task_keys = set(self._active_tasks.keys())  # {(symbol, timeframe, exchange_id), ...}
            required_tasks = set()  # Guarda as tasks que devem estar rodando {(symbol, timeframe, exchange_id), ...}

            # Para cada par necessário, verifica se a task existe ou precisa ser criada
            for symbol, timeframe in needed_pairs:
                cache_key = (symbol, timeframe)
                primary_exchange = 'binance'
                fallback_exchange = 'bybit'

                primary_key = (symbol, timeframe, primary_exchange)
                fallback_key = (symbol, timeframe, fallback_exchange)

                primary_exists = primary_key in current_task_keys
                fallback_exists = fallback_key in current_task_keys

                cache_exists_and_populated = False
                with self._lock:
                    # Verifica se a chave existe e se o deque não está vazio
                    cache_exists_and_populated = cache_key in self._cache and len(self._cache[cache_key]) > 0

                # PULA PARA O PRÓXIMO PAR SE O CACHE NÃO FOI POPULADO INICIALMENTE
                if not cache_exists_and_populated:
                    # Remove warning, pois a população inicial já logou o erro
                    # log.warning(
                    #     f"Cache para {cache_key} não populado (falha na inicialização?). Task WS não será iniciada.")
                    continue

                # Se nenhuma task existe para o par E a exchange primária está disponível
                if not primary_exists and not fallback_exists and primary_exchange in self._exchanges:
                    log.info(f"Iniciando task primária para {primary_key}")
                    task = asyncio.create_task(self._watch_ohlcv(primary_exchange, symbol, timeframe))
                    with self._lock:
                        self._active_tasks[primary_key] = task
                    required_tasks.add(primary_key)
                # Se a primária existe, marca como requerida
                elif primary_exists:
                    required_tasks.add(primary_key)
                # Se a primária não existe (talvez falhou e tentou fallback antes), mas a fallback existe, marca fallback como requerida
                elif fallback_exists:
                    required_tasks.add(fallback_key)
                # Se nenhuma existe E primária falhou E fallback está disponível (lógica de fallback iniciada em _watch_ohlcv)
                elif not primary_exists and not fallback_exists and fallback_exchange in self._exchanges:
                    # Esta condição é coberta pela lógica de fallback dentro de _watch_ohlcv
                    # Se _watch_ohlcv falhar na primária, ela tentará agendar a fallback,
                    # e na próxima iteração deste loop, a task fallback será iniciada aqui se necessário.
                    # Poderíamos iniciar a fallback aqui explicitamente se a primária não estiver disponível?
                    log.info(f"Iniciando task de fallback para {fallback_key} (Primária pode não existir ou falhou)")
                    task = asyncio.create_task(self._watch_ohlcv(fallback_exchange, symbol, timeframe))
                    with self._lock:
                        self._active_tasks[fallback_key] = task
                    required_tasks.add(fallback_key)

            # --- Cancela Tasks Desnecessárias ---
            tasks_to_cancel = []
            keys_to_remove = []
            with self._lock:  # Protege acesso a _active_tasks durante iteração
                for task_key, task_instance in self._active_tasks.items():
                    if task_key not in required_tasks:
                        log.warning(f"Task {task_key} não é mais necessária. Cancelando...")
                        if not task_instance.done():  # Só cancela se não terminou ainda
                            task_instance.cancel()
                            tasks_to_cancel.append(task_instance)
                        keys_to_remove.append(task_key)  # Marca para remover do dict

                # Remove do dict após iterar
                for key in keys_to_remove:
                    if key in self._active_tasks:
                        del self._active_tasks[key]

            # Espera um pouco ou pelas tasks canceladas
            if tasks_to_cancel:
                log.info(f"Aguardando cancelamento de {len(tasks_to_cancel)} tasks...")
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                log.info("Tasks canceladas.")
            else:
                # log.debug("Nenhuma task para cancelar. Verificando novamente em 5s.")
                await asyncio.sleep(5)  # Verifica estado a cada 5 segundos

        # --- Encerramento ---
        log.warning("Encerrando loop asyncio do WebSocketManager...")
        # Cancela todas as tasks restantes
        tasks_to_cancel = []
        with self._lock:
            remaining_keys = list(self._active_tasks.keys())  # Pega chaves antes de modificar
            for key in remaining_keys:
                task = self._active_tasks.pop(key, None)  # Remove do dict
                if task and not task.done():
                    log.info(f"Cancelando task restante: {key}")
                    task.cancel()
                    tasks_to_cancel.append(task)

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            log.info("Tasks restantes canceladas.")

        log.info("Fechando conexões das exchanges ccxt.pro...")
        close_tasks = [exchange.close() for exchange in self._exchanges.values() if hasattr(exchange, 'close')]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        log.info("Conexões ccxt.pro fechadas.")
        log.warning("Loop asyncio do WebSocketManager encerrado.")

    def start(self):
        if self._running:
            log.warning("WebSocketManager já está rodando.")
            return
        log.info("Iniciando WebSocketManager...")
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_asyncio_loop, daemon=True, name="WebSocketThread")
        self._thread.start()
        log.info("Thread do WebSocketManager iniciada.")

    def _run_asyncio_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_loop())
        finally:
            # Limpeza final do loop asyncio
            try:
                # Cancela tarefas que possam ter sido deixadas pendentes
                tasks = asyncio.all_tasks(self._loop)
                for task in tasks:
                    task.cancel()
                if tasks:
                    self._loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                # Roda uma última vez para processar callbacks de cancelamento
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception as e:
                log.error(f"Erro durante shutdown do loop asyncio: {e}")
            finally:
                self._loop.close()
                log.info("Loop asyncio fechado.")

    def stop(self):
        if not self._running:
            log.warning("WebSocketManager não está rodando.")
            return

        log.warning("Parando WebSocketManager...")
        self._running = False  # Sinaliza para o _run_loop parar

        # Adiciona um pequeno delay para o loop perceber a flag
        time.sleep(0.1)

        # O loop asyncio e as tasks internas devem encerrar.
        if self._thread and self._thread.is_alive():
            log.info("Aguardando thread do WebSocketManager encerrar...")
            self._thread.join(timeout=20)  # Aumenta timeout
            if self._thread.is_alive():
                log.error("Timeout esperando thread do WebSocketManager encerrar!")
            else:
                log.info("Thread do WebSocketManager encerrada.")
        self._loop = None  # Limpa referência ao loop
        self._thread = None  # Limpa referência à thread

    def subscribe(self, symbol, timeframe):
        """Registra o interesse de um BOT em um par/timeframe."""
        if timeframe not in self.SUPPORTED_TIMEFRAMES:
            log.error(f"Timeframe '{timeframe}' não suportado.")
            return
        symbol = symbol.replace('/', '-')  # Formato interno
        cache_key = (symbol, timeframe)
        with self._lock:
            self._subscriptions[cache_key] += 1
            count = self._subscriptions[cache_key]
            log.info(f"Bot subscreveu a {cache_key}. Contagem atual: {count}")
            # O _run_loop iniciará a task se necessário

    def unsubscribe(self, symbol, timeframe):
        """Remove o interesse de um BOT em um par/timeframe."""
        symbol = symbol.replace('/', '-')  # Formato interno
        cache_key = (symbol, timeframe)
        with self._lock:
            if cache_key in self._subscriptions:
                self._subscriptions[cache_key] -= 1
                count = self._subscriptions[cache_key]
                log.info(f"Bot cancelou subscrição de {cache_key}. Contagem atual: {count}")
                if self._subscriptions[cache_key] <= 0:
                    # Remove do contador se <= 0
                    del self._subscriptions[cache_key]
                    log.warning(f"Contagem de subscrição zerada para {cache_key}.")
                    # O _run_loop cancelará a task se _initial_subscriptions também não o contiver
            else:
                log.warning(f"Tentativa de cancelar subscrição inexistente para {cache_key}")

    def get_klines(self, symbol, timeframe) -> pd.DataFrame | None:
        """
        Retorna o DataFrame de klines mais recente do cache.
        Retorna None se os dados não estiverem disponíveis.
        A verificação de tamanho mínimo deve ser feita pelo chamador (bot_logic).
        """
        symbol = symbol.replace('/', '-')
        cache_key = (symbol, timeframe)
        with self._lock:
            cache_deque = self._cache.get(cache_key)
            # Verifica apenas se existe e não está vazio
            if not cache_deque:
                return None
            try:
                # ... (código de conversão para DataFrame mantido) ...
                data_copy = list(cache_deque)
                if not data_copy: return None  # Checa se cópia não está vazia
                df = pd.DataFrame(data_copy, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            except Exception as e:
                log.error(f"Erro converter cache p/ DF {cache_key}: {e}", exc_info=True)
                return None
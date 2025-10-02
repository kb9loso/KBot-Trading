# 2. Guia do Painel de Controle

O painel de controle é a interface central para gerenciar e monitorar todas as atividades do KBot.

## Seções Principais

### 1. Contas Gerenciadas

Esta tabela exibe todas as contas de trading que você configurou no bot.

- **Nome da Conta:** O nome que você deu à sua conta (ex: "Conta Principal").
- **Exchange:** A corretora onde a conta está (ex: Pacifica).
- **Chave Pública:** Um trecho da sua chave de API para identificação.
- **Intervalo:** O tempo em segundos que o bot aguarda entre cada verificação de sinais (ex: 180s).
- **Status:** Mostra se o bot está `Executando` ou `Parado` para aquela conta.
- **Ação:** Botões para Iniciar, Parar, Editar ou Remover uma conta.

### 2. Performance (Últimos 30 Dias)

Aqui você encontra um resumo do desempenho das suas operações.

- **PnL Total:** O lucro ou prejuízo (Profit and Loss) total realizado.
- **Trades:** O número total de operações fechadas.
- **Acerto:** A taxa de acerto (percentagem de operações que fecharam com lucro).
- **Ganho Médio:** O valor médio ganho nas operações lucrativas.
- **Perda Média:** O valor médio perdido nas operações com prejuízo.
- **Taxas:** O total de taxas pagas à corretora.

### 3. Setups Ativos

Esta é a seção mais importante. Ela lista todas as estratégias que o bot está a monitorar e para quais ativos.

- **Ativo:** A criptomoeda que está a ser monitorada (ex: BTC, ETH). O ícone verde indica que há uma posição aberta para esse ativo; o amarelo indica que está apenas a ser monitorado.
- **Estratégia:** O nome da estratégia que está a ser aplicada.
- **Alav.:** A alavancagem configurada para as operações.
- **Risco/Trade:** O percentual do seu capital total que você está a arriscar em cada operação.
- **Stop Loss / Take Profit:** Os parâmetros de risco e alvo definidos.
- **Modo Saída / Direção / Trailing Stop:** Configurações avançadas da operação (explicadas no documento `4_Configuracoes_Avancadas.md`).

### 4. Adicionar/Atualizar Setup e Backtest

Este formulário permite-lhe criar novas automações ou testar estratégias.

- **Adicionar/Atualizar:** Cria um novo "Setup Ativo" ou atualiza um existente para o mesmo ativo.
- **Rodar Backtest:** Simula a performance da estratégia e dos parâmetros selecionados com dados dos últimos 30 dias, mostrando o melhor resultado possível.

### 5. Logs

Exibe em tempo real tudo o que o bot está a fazer: verificações, sinais encontrados, ordens enviadas (`EXECUTION`), avisos (`WARNING`) e erros (`ERROR`). É a melhor forma de acompanhar a atividade do bot.
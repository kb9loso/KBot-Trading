# 4. Guia de Configurações Avançadas de Setup

Esta seção detalha cada campo do formulário "Adicionar/Atualizar Setup" para que você possa configurar o bot com precisão.

---

### Campos Principais

- **Conta:** A conta de trading que executará este setup.
- **Ativo:** A criptomoeda a ser negociada (ex: BTC, ETH, SOL).
- **Alavancagem:** O fator de multiplicação da sua posição. Uma alavancagem de 10x significa que uma variação de 1% no preço do ativo resulta numa variação de 10% no seu resultado. **Aumenta tanto os lucros quanto as perdas.**
- **Direção:** Define quais tipos de operações o bot pode abrir:
    - `Long & Short`: Pode operar tanto na alta quanto na baixa.
    - `Long Only`: Apenas operações de compra (aposta na subida do preço).
    - `Short Only`: Apenas operações de venda (aposta na queda do preço).
- **Estratégia:** O conjunto de regras e indicadores que o bot usará (detalhado no arquivo `3_Estrategias.md`).

---

### Gerenciamento de Risco

- **Risco (%):** Este é um dos campos mais importantes. Define qual percentual do **saldo total da sua conta** você está disposto a arriscar em uma única operação. O bot usa este valor, junto com a distância do Stop Loss, para calcular o tamanho da posição.
    - *Exemplo:* Com um saldo de $1000 e Risco de 1%, você arriscará no máximo $10 por operação.

- **Valor SL:** O valor para o seu Stop Loss.
    - Se o tipo for `percentage`, este valor é a **distância percentual** do preço de entrada onde a operação será fechada com perda (ex: 2.5%).
    - Se o tipo for `ATR`, este valor é um **multiplicador** do indicador Average True Range (ATR), que mede a volatilidade. Um SL de "2x ATR" será mais largo em mercados voláteis e mais curto em mercados calmos.

- **TP (RRR):** O alvo de lucro, definido como uma Relação Risco/Retorno (Risk/Reward Ratio).
    - *Exemplo:* Se o seu Stop Loss está a $10 de distância do preço de entrada e o seu TP (RRR) é `2.0`, o seu alvo de lucro será colocado a $20 de distância.

---

### Configurações de Saída

- **Modo Saída:**
    - `Passivo`: A operação só pode ser fechada pelo Stop Loss ou Take Profit.
    - `Ativo`: Além do SL/TP, a própria estratégia pode gerar um sinal para fechar a posição se o mercado reverter (conforme a "Lógica de Saída" da estratégia).

- **Trailing Stop (Stop Móvel):** Um stop loss que se move a favor da sua operação para proteger lucros. Só funciona com `Modo Saída: Ativo`.
    - `Nenhum`: Desativado.
    - `Breakeven`: Move o Stop Loss para o preço de entrada (zero a zero) assim que a operação atinge um determinado alvo de lucro.
        - **Acionar em (RRR):** O alvo (em Risco/Retorno) que ativa o Breakeven. Um valor de `1.0` significa que o SL irá para o ponto de entrada assim que o lucro atingir o mesmo valor do risco inicial.
    - `ATR Trailing`: Um stop loss que "persegue" o preço a uma distância baseada na volatilidade (ATR).
        - **Múltiplo ATR:** Define a distância. Um valor maior deixa mais espaço para o preço variar antes de fechar a operação.
    - **Remover TP:** Se marcado, o alvo de lucro original é removido quando o Trailing Stop é ativado, permitindo que a operação continue a correr enquanto a tendência durar.
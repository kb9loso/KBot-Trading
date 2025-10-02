# 3. Guia Detalhado das Estratégias

Cada estratégia no KBot utiliza uma combinação de indicadores técnicos para tomar decisões. Abaixo, explicamos a lógica de cada uma em linguagem simples.

---

### 1. Tendencia Rapida (15m)

- **Objetivo:** Capturar o início de tendências rápidas.
- **Timeframe:** Gráfico de 15 minutos.
- **Ideal para:** Mercados que estão a começar a mostrar uma direção clara (alta ou baixa).

- **Sinal de Compra (Long) quando:**
    1.  A Média Móvel Exponencial (EMA) de 9 períodos cruza para **cima** da EMA de 21 períodos (sinal de início de tendência de alta).
    2.  O Índice de Força Relativa (RSI) está **acima de 55** (mostrando que há mais força compradora do que vendedora).
    3.  O indicador ADX está **acima de 20** (confirmando que existe uma tendência, e não apenas uma variação lateral).

- **Sinal de Venda (Short) quando:**
    1.  A EMA de 9 cruza para **baixo** da EMA de 21 (sinal de início de tendência de baixa).
    2.  O RSI está **abaixo de 45** (mostrando mais força vendedora).
    3.  O ADX está **acima de 20** (confirmando a tendência).

- **Sinal de Saída da Posição (Modo Ativo):**
    - Se estiver comprado, sai quando a EMA de 9 cruza para baixo da EMA de 21 e o RSI perde força.
    - Se estiver vendido, sai quando a EMA de 9 cruza para cima da EMA de 21 e o RSI ganha força.

---

### 2. Momentum Explosivo (15m)

- **Objetivo:** Entrar em movimentos fortes e repentinos ("explosões").
- **Timeframe:** Gráfico de 15 minutos.
- **Ideal para:** Movimentos de mercado fortes e com aumento de volume.

- **Sinal de Compra (Long) quando:**
    1.  A linha do MACD cruza para **cima** da sua linha de sinal (indicando um ganho de momentum de alta).
    2.  O MACD está **positivo** (acima de zero), confirmando o viés de alta.
    3.  O ADX está **acima de 20** (confirmando que há uma tendência).
    4.  O preço de fecho está **acima da VWAP** (preço médio ponderado pelo volume), mostrando que o preço está forte.

- **Sinal de Venda (Short) quando:**
    1.  A linha do MACD cruza para **baixo** da sua linha de sinal (momentum de baixa).
    2.  O MACD está **negativo** (abaixo de zero).
    3.  O ADX está **acima de 20**.
    4.  O preço de fecho está **abaixo da VWAP**.

- **Sinal de Saída da Posição (Modo Ativo):**
    - Se comprado, sai se o MACD cruzar para baixo de zero (perda de momentum de alta).
    - Se vendido, sai se o MACD cruzar para cima de zero (perda de momentum de baixa).

---

### 3. Scalping Intraday (15m)

- **Objetivo:** Realizar operações curtas para obter ganhos rápidos.
- **Timeframe:** Gráfico de 15 minutos.
- **Ideal para:** Mercados com pequenas tendências definidas, mas com volatilidade.

- **Sinal de Compra (Long) quando:**
    1.  O mercado está em microtendência de alta (EMA 9 > EMA 21).
    2.  O indicador Estocástico (Stoch) estava em **nível de sobrevenda** (abaixo de 20) no candle anterior.
    3.  A linha %K do Estocástico cruza para **cima** da linha %D (sinal de compra do oscilador).
    4.  O preço está **acima da VWAP**.

- **Sinal de Venda (Short) quando:**
    1.  O mercado está em microtendência de baixa (EMA 9 < EMA 21).
    2.  O Estocástico estava em **nível de sobrecompra** (acima de 80) no candle anterior.
    3.  A linha %K cruza para **baixo** da linha %D.
    4.  O preço está **abaixo da VWAP**.

- **Sinal de Saída da Posição (Modo Ativo):**
    - Se comprado, sai se o Estocástico entrar em zona de sobrecompra (>75), indicando possível exaustão do movimento.
    - Se vendido, sai se o Estocástico entrar em zona de sobrevenda (<25).

---

### 4. Breakout Curto (15m)

- **Objetivo:** Lucrar com rompimentos de volatilidade.
- **Timeframe:** Gráfico de 15 minutos.
- **Ideal para:** Momentos em que o mercado "explode" para cima ou para baixo após um período de calmaria.

- **Sinal de Compra (Long) quando:**
    1.  O preço de fecho está **acima da Banda de Bollinger superior** (rompimento de alta).
    2.  O volume da negociação é **1.5x maior que a média** (volume confirma a força do rompimento).
    3.  O MACD está em modo de compra (linha principal acima da linha de sinal).

- **Sinal de Venda (Short) quando:**
    1.  O preço de fecho está **abaixo da Banda de Bollinger inferior** (rompimento de baixa).
    2.  O volume é **1.5x maior que a média**.
    3.  O MACD está em modo de venda.

- **Sinal de Saída da Posição (Modo Ativo):** Sai da operação se o preço retornar para a média central das Bandas de Bollinger.

---

### 5. Swing Curto (1h)

- **Objetivo:** Capturar a maior parte de uma tendência de médio prazo.
- **Timeframe:** Gráfico de 1 hora.
- **Ideal para:** Operações que podem durar várias horas ou dias (Swing Trade).

- **Sinal de Compra (Long) quando:**
    1.  A média móvel de médio prazo (EMA 21) cruza para **cima** da de longo prazo (EMA 50).
    2.  O ADX confirma a força da tendência (>25).

- **Sinal de Venda (Short) quando:**
    1.  A EMA 21 cruza para **baixo** da EMA 50.
    2.  O ADX confirma a força da tendência (>25).

- **Sinal de Saída da Posição (Modo Ativo):** A posição é fechada se as médias móveis mais curtas (EMA 9 e 21) se cruzarem, indicando um possível enfraquecimento da tendência.

---

### 6. Reversão Forte (1h)

- **Objetivo:** Tentar "apanhar" o fundo de uma queda ou o topo de uma subida (reversão).
- **Timeframe:** Gráfico de 1 hora.
- **Ideal para:** Mercados que esticaram muito numa direção e podem estar prestes a corrigir.

- **Sinal de Compra (Long) quando:**
    1.  No candle anterior, o preço fechou **abaixo da Banda de Bollinger inferior** (extremo de venda).
    2.  No candle atual, o preço voltou para **dentro da Banda de Bollinger** (primeiro sinal de reação dos compradores).
    3.  O RSI está **abaixo de 30** (confirmando que o mercado está sobrevendido).

- **Sinal de Venda (Short) quando:**
    1.  No candle anterior, o preço fechou **acima da Banda de Bollinger superior** (extremo de compra).
    2.  No candle atual, o preço voltou para **dentro da Banda**.
    3.  O RSI está **acima de 70** (confirmando sobrecompra).

- **Sinal de Saída da Posição (Modo Ativo):** Sai da operação se o preço atingir a média central das Bandas de Bollinger, que é o alvo natural de uma reversão.

---

### 7. Tendência Confirmada (1h)

- **Objetivo:** Entrar em tendências fortes e já estabelecidas, aproveitando recuos (pullbacks).
- **Timeframe:** Gráfico de 1 hora.
- **Ideal para:** Mercados com tendência clara e saudável.

- **Sinal de Compra (Long) quando:**
    1.  As médias móveis estão alinhadas para alta (EMA 9 > EMA 21 > EMA 50).
    2.  O preço faz um recuo e toca na EMA 21.
    3.  O preço volta a subir, fechando acima da EMA 21.
    4.  O ADX está forte (>25).

- **Sinal de Venda (Short) quando:**
    1.  As médias móveis estão alinhadas para baixa (EMA 9 < EMA 21 < EMA 50).
    2.  O preço faz um recuo e toca na EMA 21.
    3.  O preço volta a cair, fechando abaixo da EMA 21.
    4.  O ADX está forte (>25).

- **Sinal de Saída da Posição (Modo Ativo):** A posição é fechada se a tendência de médio prazo se quebrar (cruzamento da EMA 21 com a EMA 50).

---

### 8. Scalper Volume (1m)

- **Objetivo:** Realizar muitas operações curtas, baseadas em picos de volume.
- **Timeframe:** Gráfico de 1 minuto.
- **Ideal para:** Scalping de altíssima frequência em mercados com liquidez.

- **Sinal de Compra (Long) quando:**
    1.  O preço está acima da média móvel curta (EMA 9).
    2.  O volume é **1.5x maior que a média**.
    3.  O RSI não está sobrecomprado (<60), indicando que ainda há espaço para subir.

- **Sinal de Venda (Short) quando:**
    1.  O preço está abaixo da EMA 9.
    2.  O volume é **1.5x maior que a média**.
    3.  O RSI não está sobrevendido (>40).

- **Sinal de Saída da Posição (Modo Ativo):** A posição é fechada muito rapidamente se o preço cruzar a EMA 9 na direção oposta ou se o Estocástico indicar exaustão do movimento.

---

### 9. MACD/RSI Trend Follower (1h)

- **Objetivo:** Seguir tendências confirmadas por múltiplos indicadores de momentum e força.
- **Timeframe:** Gráfico de 1 hora.
- **Ideal para:** Entrar em tendências mais robustas e confirmadas.

- **Sinal de Compra (Long) quando:**
    1.  O MACD cruza para **cima** da sua linha de sinal e está positivo.
    2.  O RSI de 14 períodos mostra força compradora (>52).
    3.  O ADX confirma que existe uma tendência (>20).

- **Sinal de Venda (Short) quando:**
    1.  O MACD cruza para **baixo** da sua linha de sinal e está negativo.
    2.  O RSI mostra força vendedora (<48).
    3.  O ADX confirma que existe uma tendência (>20).

- **Sinal de Saída da Posição (Modo Ativo):** A posição é fechada se o MACD cruzar de volta a sua linha de sinal, indicando perda de momentum.
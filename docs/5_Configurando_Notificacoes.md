# 5. Guia de Configuração de Notificações (Telegram)

O KBot pode enviar alertas em tempo real para o seu Telegram sempre que uma posição for aberta ou fechada. Para isso, você precisa configurar um bot no Telegram e informar ao KBot as credenciais de acesso.

Siga os passos abaixo com atenção.

---

### Passo 1: Criar o seu Bot no Telegram

Você precisa "pedir" ao Telegram para criar um bot para você. Isso é feito conversando com um bot oficial chamado `BotFather`.

1.  Abra o Telegram (no celular ou no computador).
2.  Na barra de busca, procure por **`BotFather`** (ele terá um selo de verificação azul).
3.  Inicie uma conversa com o BotFather e envie o comando:
    `/newbot`
4.  O BotFather pedirá um nome para o seu bot. Pode ser qualquer nome (ex: "KBot Alertas").
5.  Em seguida, ele pedirá um nome de usuário (username) para o bot. Este nome **precisa ser único** e **terminar com a palavra "bot"** (ex: `meu_kbot_alertas_bot`).
6.  Se o nome de usuário estiver disponível, o BotFather enviará uma mensagem de sucesso contendo o **token de acesso (API Token)**. Este token é uma longa sequência de números e letras.
7.  **Copie este token com cuidado.** Ele é a "senha" do seu bot.

![Exemplo de Token do BotFather](https://i.imgur.com/8x234d3.png)

---

### Passo 2: Obter o seu Chat ID

O Chat ID é o número de identificação único da conversa para onde o bot enviará as mensagens.

1.  No Telegram, procure pelo nome de usuário do bot que você acabou de criar (ex: `@meu_kbot_alertas_bot`).
2.  Inicie uma conversa com ele e envie qualquer mensagem (o comando `/start` é uma boa opção). **Este passo é obrigatório** para que o bot reconheça a conversa.
3.  Agora, você precisa descobrir o ID dessa conversa. A forma mais fácil é usar outro bot para isso. Procure por **`@userinfobot`**.
4.  Inicie uma conversa com o `@userinfobot` e ele imediatamente responderá com as suas informações, incluindo o seu **ID**. Copie esse número.

---

### Passo 3: Configurar no Painel do KBot

Com o Token e o Chat ID em mãos, volte para a interface web do KBot.

1.  Clique no botão **"Notificações"** no canto superior direito.
2.  No formulário que aparecer, cole as informações que você obteve:
    - **Token do Bot:** Cole o token que o `BotFather` lhe deu.
    - **Chat ID:** Cole o ID que o `@userinfobot` lhe deu.
3.  Marque as caixas de seleção para escolher quais tipos de alerta deseja receber (`Notificar Abertura`, `Notificar Fechamento`).
4.  Clique em **"Salvar Configurações"**.

---

### Passo 4: Ativar Notificações por Conta

Como último passo, você precisa dizer ao KBot para quais das suas contas de negociação você deseja receber os alertas.

1.  Vá para a tabela **"Contas Gerenciadas"**.
2.  Na coluna "Ação", ao lado da conta desejada, você verá um ícone de sino.
3.  Se o sino estiver cinza e cortado (`fa-bell-slash`), as notificações para essa conta estão desativadas. Clique nele.
4.  O ícone se tornará um sino verde (`fa-bell`), indicando que as notificações agora estão **ativas** para aquela conta.

Pronto! A partir de agora, você receberá os alertas de trading diretamente no seu Telegram.
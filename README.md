# KBot - Guia de Instalação e Execução Local

Este guia destina-se a ajudar utilizadores a configurar e executar o KBot no seu próprio computador.

---

## 1. Pré-requisitos

Antes de começar, certifique-se de que tem o seguinte software instalado:

* **Python 3.10 ou superior:** O bot foi desenvolvido em Python. Pode descarregar a versão mais recente em [python.org](https://www.python.org/downloads/).
    * **Importante (Windows):** Durante a instalação do Python, **marque a opção "Add Python to PATH"**.

---

## 2. Instalação e Execução

Escolha **um** dos métodos abaixo. O método simplificado é recomendado para utilizadores Windows.

### Método A: Instalação Simplificada (Windows)

A maneira mais fácil de instalar e executar o bot no Windows.

#### Passo 1: Instalação (Execute apenas uma vez)
- Dê um duplo clique no arquivo `setup.bat`.
- Uma janela de terminal aparecerá, criará o ambiente virtual e instalará todas as dependências necessárias.
- Aguarde até que a mensagem "Setup concluido com sucesso!" seja exibida.

#### Passo 2: Execução (Para iniciar o bot)
- Após a instalação, dê um duplo clique no arquivo `start.bat` sempre que quiser iniciar o bot.
- O terminal será aberto e a aplicação começará a rodar.

---

### Método B: Instalação Manual (Todos os Sistemas Operacionais)

#### Passo 1: Criar e Ativar o Ambiente Virtual
1.  Abra um terminal (no Windows, **Prompt de Comando**; no macOS ou Linux, o **Terminal**).
2.  Navegue até à pasta do projeto (`KBot-Trading`).
3.  Crie o ambiente:
```bash
    python -m venv venv
```
4.  Ative o ambiente:
    * **No Windows:** `.\venv\Scripts\activate`
    * **No macOS e Linux:** `source venv/bin/activate`

(Após a ativação, você deverá ver `(venv)` no início da linha do seu terminal).

#### Passo 2: Instalar as Dependências
Com o ambiente ativado, execute o comando abaixo para instalar todas as bibliotecas necessárias:
```bash
  pip install -r requirements.txt
```
### Passo 3: Criar o Arquivo de Configuração

1.  Na pasta do projeto, encontre o ficheiro `config.example.json`.
2.  Faça uma cópia deste ficheiro e renomeie-a para `config.json`. **Não precisa de editar o conteúdo deste arquivo manualmente.**

#### Passo 4: Execução

Ainda no terminal com o ambiente ativado, inicie a aplicação:

```bash
  python app.py
```

-----

## 3\. Configuração Inicial

Com o bot em execução pela primeira vez, siga os passos abaixo.

### Passo 1: Aceder à Interface Web

1.  Abra o seu navegador de internet (Chrome, Firefox, etc.).
2.  Na barra de endereços, digite: `http://127.0.0.1:5000`
3.  Pressione Enter. A interface web do KBot deverá carregar.

### Passo 2: Adicionar a Sua Conta na Interface

1.  Na interface web, clique no botão **"Nova Conta"**.
2.  Preencha o formulário com o nome da conta e as suas chaves de API (`main_public_key` e `agent_private_key`).
3.  Clique em **"Salvar Conta"**.

A partir daqui, todas as configurações de setups de trading e gerenciamento de contas são feitas diretamente pela interface.

Para parar a aplicação, volte ao terminal que está a executar o bot e pressione `Ctrl + C`.

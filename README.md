# KBot - Guia de Instalação e Execução Local

Este guia destina-se a ajudar utilizadores a configurar e executar o KBot no seu próprio computador.

---

## 1. Pré-requisitos

Antes de começar, certifique-se de que tem o seguinte software instalado:

* **Python 3.10 ou superior:** O bot foi desenvolvido em Python. Pode descarregar a versão mais recente em [python.org](https://www.python.org/downloads/).
    * **Importante (Windows):** Durante a instalação do Python, **marque a opção "Add Python to PATH"**.

---

## 2. Configuração da Conta

Esta etapa é necessária tanto para o método simplificado como para o manual.

1.  Na pasta do projeto, encontre o ficheiro `config.example.json`.
2.  Faça uma cópia deste ficheiro e renomeie-a para `config.json`.
3.  Abra o `config.json` com um editor de texto e preencha os campos `main_public_key` e `agent_private_key` com as suas chaves de API.

---

## 3. Instalação e Execução

Escolha **um** dos métodos abaixo. O método simplificado é recomendado para utilizadores Windows.

### Método A: Instalação Simplificada (Windows)

A maneira mais fácil de instalar e executar o bot no Windows é usando os scripts automatizados.

#### Passo 1: Instalação (Execute apenas uma vez)
- Dê um duplo clique no arquivo `setup.bat`.
- Uma janela de terminal aparecerá, criará o ambiente virtual e instalará todas as dependências necessárias automaticamente.
- Aguarde até que o processo termine e a mensagem "Setup concluido com sucesso!" seja exibida.

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
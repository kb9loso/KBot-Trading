# KBot - Guia de Instalação e Execução Local

Este guia destina-se a ajudar utilizadores a configurar e executar o KBot no seu próprio computador.

---

## Método Simplificado (Windows)

A maneira mais fácil de instalar e executar o bot no Windows é usando os scripts automatizados.

### 1. Instalação (Execute apenas uma vez)
- Dê um duplo clique no arquivo `setup.bat`.
- Uma janela de terminal aparecerá, criará o ambiente virtual e instalará todas as dependências necessárias automaticamente.
- Aguarde até que o processo termine e a mensagem "Setup concluido com sucesso!" seja exibida.

### 2. Execução (Para iniciar o bot)
- Após a instalação, dê um duplo clique no arquivo `start.bat` sempre que quiser iniciar o bot.
- O terminal será aberto e a aplicação começará a rodar.
- Abra o seu navegador e acesse o endereço `http://127.0.0.1:5000` para ver o painel de controle.
- Para parar a aplicação, volte ao terminal e pressione `Ctrl + C`.

---

## Método Manual (Para todos os sistemas operacionais)

### Pré-requisitos

Antes de começar, certifique-se de que tem o seguinte software instalado:

1.  **Python 3.10 ou superior:** O bot foi desenvolvido em Python. Pode descarregar a versão mais recente em [python.org](https://www.python.org/downloads/). Durante a instalação, **marque a opção "Add Python to PATH"**.

### Passo 1: Criar um Ambiente Virtual

É uma boa prática criar um "ambiente virtual" para isolar as dependências do projeto.

1.  Abra um terminal (no Windows, **Prompt de Comando** ou **PowerShell**; no macOS ou Linux, o **Terminal**).
2.  Navegue até à pasta do projeto (`KBot-Trading`).
3.  Execute o comando para criar o ambiente:
    ```bash
    python -m venv venv
    ```

### Passo 2: Ativar o Ambiente Virtual
* **No Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
* **No macOS e Linux:**
    ```bash
    source venv/bin/activate
    ```
Depois de ativado, deverá ver `(venv)` no início da linha do seu terminal.

### Passo 3: Instalar as Dependências

Com o ambiente ativado, instale todas as bibliotecas de uma vez usando o arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
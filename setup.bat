@echo off
echo Criando ambiente virtual na pasta 'venv'...
python -m venv venv

echo Ativando ambiente virtual...
call .\venv\Scripts\activate

echo Instalando dependencias do arquivo requirements.txt...
pip install -r requirements.txt

echo.
echo Setup concluido com sucesso!
echo Para iniciar o bot, execute o arquivo 'start.bat'.
pause
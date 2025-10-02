@echo off
echo Ativando ambiente virtual...
call .\venv\Scripts\activate

echo Iniciando o KBot Trading...
echo Acesse http://127.0.0.1:5000 no seu navegador.

python app.py

pause
:: Set a number of environmental variables and locale-related settings needed
:: for this recognizer to run as expected before calling the recognizer itself.
::
:: It seems that recognizer processes invoked by ELAN have their locale set
:: to C.  This implies a default ASCII file encoding, which causes some
:: scripts to refuse to run (since many assume a more Unicode-friendly view
:: of the world somewhere in their code).

SET LC_ALL="en_US.UTF-8"
SET PYTHONIOENCODING="utf-8"

:: Activate the virtual environment, then execute the main recognizer script.
call ".\venv-xls-r-elan\Scripts\activate.bat"
".\venv-xls-r-elan\Scripts\python.exe" "xls-r-elan.py"

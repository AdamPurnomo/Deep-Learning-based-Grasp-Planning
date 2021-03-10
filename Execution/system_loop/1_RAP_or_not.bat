@echo on
cd /d %~dp0

setlocal


(set /p file_input=) < ./loop_num.txt
set /a loop_num=%file_input%

if %loop_num% EQU 6 (
	echo ^0> ./1_RAP_or_not.txt
) else (
	echo ^1> ./1_RAP_or_not.txt
)


endlocal
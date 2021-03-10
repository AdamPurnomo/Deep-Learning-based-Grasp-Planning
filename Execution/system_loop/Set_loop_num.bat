@ECHO off
@rem　引数をtxtファイルに保存します。

SET TEST=%1
ECHO ^%TEST%>loop_num.txt
ECHO 繰り返し回数を%1回にセットしました。
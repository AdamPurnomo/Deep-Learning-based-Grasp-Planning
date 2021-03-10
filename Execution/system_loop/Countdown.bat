@echo off
@rem　txtファイルを-1します。

SET /P COUNT=<loop_num.txt
SET /A COUNT-=1
ECHO ^%COUNT%>loop_num.txt
ECHO 残り%COUNT%回
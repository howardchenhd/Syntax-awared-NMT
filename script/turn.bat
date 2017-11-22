echo turn:复合tag/id 转化，非合法树则会有null输出并且打印输出到对应日志文件中
java -jar turn.jar MT02.ce.tree ../tree-based-out/MT02.ce.tree >> ../tree-based-out/MT02.turn.log 2>&1
java -jar turn.jar MT03.ce.tree ../tree-based-out/MT03.ce.tree >> ../tree-based-out/MT03.turn.log 2>&1
java -jar turn.jar MT04.ce.tree ../tree-based-out/MT04.ce.tree >> ../tree-based-out/MT04.turn.log 2>&1
java -jar turn.jar MT05.ce.tree ../tree-based-out/MT05.ce.tree >> ../tree-based-out/MT05.turn.log 2>&1
java -jar turn.jar MT06.ce.tree ../tree-based-out/MT06.ce.tree >> ../tree-based-out/MT06.turn.log 2>&1
java -jar turn.jar corpus.ch.berkeleytree ../tree-based-out/corpus.ch.berkeleytree >> ../tree-based-out/corpus.ch.berkeleytree.log 2>&1
pause
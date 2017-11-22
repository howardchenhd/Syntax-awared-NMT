### Berkeley parser tree to triple

trun.jar是我新整理出来的结果，包含检查输入是不是合理的功能
可以打开turn.bat文件进行检查输入和句法树转化工作

###命令行说明
java -jar turn.jar input_tree [result_file_name]

若不设置参数2，则默认jar的参数2等于参数1.
转化结果：生成result_file_name.idturn

###日志文件内容说明
如果一个句子不是合法输入，会打印提示信息到控制台，在转化结果文件中对应的是"null"

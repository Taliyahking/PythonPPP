# PythonPPP
-----Data：要使用到的卫星相关的文件、精密星历(前后共三天)、观测值文件等。
-----result：结果输出
        注意：在result中有一个visualization.html文件，是一个可以打开查看可视化的定位结果网页。
如果直接在文件夹中打开可能会导致加载不进去json文件，出现的都是示例结果。
所以如果打开后结果不正确，可以在PyCharm中打开，这样json数据就可以正确加载进去了。


代码部分：
-----datahand.py：数据导入
-----pre_process.py：数据预处理
-----atmo_correct.py：大气改正
-----model_correct.py：模型改正
-----gnss_filter.py：卡尔曼滤波
-----evaluate.py：结果评估
-----helper_function.py：辅助函数
-----main.py：主函数

运行的时候只需要运行main函数即可，这个主函数会组织所有函数进行PPP过程。
其中的参数可以自行修改。

最后的结果请在result文件夹中查看，控制台也有相应的代码运行信息。


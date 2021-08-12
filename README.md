# Generative-Adversarial-Optimizatio-GAO
对论文Generative Adversarial Optimization的复现及源代码改动、可视化  
论文原文: [Generative Adversarial Optimization](https://link.springer.com/chapter/10.1007/978-3-030-26369-0_1)，GAO [code](https://www.cil.pku.edu.cn/test/497838.htm)  

----  
1. 改动如下：  
    - 使用numpy重写了[CEC2013](https://www.al-roomi.org/multimedia/CEC_Database/CEC2013/RealParameterOptimization/CEC2013_RealParameterOptimization_TechnicalReport.pdf)测试集，使之能运行通过  
    - 使用matplotlib可视化测试函数，并展示收敛过程  
    - 使用tkinter制作了GUI界面  
2. 代码使用：  
运行main.py、main_matplotlib.py、main_tkinter.py、main_ui.py中的任意一个  
3. 在以下环境下运行通过(相关依赖)：
tensorflow 1.12.0、keras 2.2.4、numpy 1.16.0、matplotlib 3.0.3、pandas 0.25.3

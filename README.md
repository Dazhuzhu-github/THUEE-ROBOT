# THUEE-ROBOT
清华大学电子系智能机器人设计实验

## pic recognition

图形识别部分
rgb是将图片hsv和rgb打出来，方便houfline调整阈值

因为比赛的场地更换频繁，加上场地是黄色的地板和绿色的线条，调整阈值要多加小心
使用聚类来合并找出来的线，我是取了第一条，平均的效果因为我聚类的阈值大，没很准

## exp

实验分为4个

exp1 测试机器人基本功能（直行）

exp2 辨识红色障碍物避障

exp3 训练三层的神经网络，并使用FPGA硬件加速，达到在PYNQ上面跑的功能

exp4 寻路避障（今年UWB是坏的所以基本同实验2）

## 最终比赛

场地为3*3米的平台，由绿线化为81格正方形

比赛会给

1 出发和终点

2 目标（最终和混肴项）和目标朝向

基本思路是首先写dijkstra找出路径

由于UWB不好使，使用图像识别找出绿线，确保每次都走一格

## 后记

比赛时一直识别不到绿线，不幸的在第二场比赛才发现因为环境从7楼换到了11楼，而且我们出发的区块黄绿不明显，无法区分，加上当天是黄光打下来

第二次最后5min是才发现可以通过手机手电照路来使之辨识成功，可惜要点到屏幕时，center2的定位看来阈值没调太好，鞠躬擦边了屏幕，可惜了

这门课前期划水，后几周很肝，助教人很好，理论课没啥用，要注意时间安排。

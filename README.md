# player  
由Stephen Dranger基于ffplay核心架构编写的播放器，用来作为ffmpeg入门学习的教程，代码阅读需要一定的音视频基础和C语言基础。  
虽然是多年前出的教程，有一些老旧的接口目前已经废弃，但仍然是用来学习ffmpeg的一个非常不错的项目。  

本人目前只学到Tutorial 5，在读完源码后，加上了一些注释，梳理了一张函数调用关系图，替换了一些旧接口，在逻辑思路和架构方面加入了一些自己的想法，代码变动较明显的地方详见更新日志。  
教程：http://www.dranger.com/ffmpeg/ffmpeg.html  
github源码：https://github.com/mpenkov/ffmpeg-tutorial  

运行需以下库：  
SDL2
以及
ffmpeg下的：
libavcodec
libavformat
libswscale
libswresample
libavutil  

编译：gcc player.c -o player `pkg-config --cflags --libs sdl2 libavcodec libavformat libavutil libswresample libswscale` 亦可执行make来编译。  
运行：./player 多媒体路径/XXX.mp4  

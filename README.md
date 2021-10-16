# player
国外友人基于ffplay核心架构，写出的播放器，作为ffmpeg入门学习的教程，需要一定的音视频基础和C语言基础
虽然是多年前出的教程，有一些老旧的接口目前已经废弃，但仍然是用来学习ffmpeg的一个非常不错的项目
我在读完源码后，加上了注释，替换了一些旧接口，由于新接口用法不太一样，因此代码也有较多更改，主要变动详见更新日志
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

# player  
基于Stephen Dranger的ffmpeg教程编写的播放器，需要有音视频相关的知识，对多线程要有一定了解。  
虽然是多年前出的教程，但仍然是学习ffmpeg的一个非常不错途径，很感谢作者写出这么好的教程。  

我在源码基础上做了一些改动，加上了大量注释，替换了一些旧接口，在逻辑思路和架构方面加入了一些自己的想法，梳理了一张函数调用关系图和流程图，代码变动较明显的地方记录在了更新日志中。  
教程：http://www.dranger.com/ffmpeg/ffmpeg.html  
github源码：https://github.com/mpenkov/ffmpeg-tutorial  

**运行需以下库**  
SDL2
以及
ffmpeg下的：
libavcodec
libavformat
libswscale
libswresample
libavutil  

**编译**：gcc player.c -o player `pkg-config --cflags --libs sdl2 libavcodec libavformat libavutil libswresample libswscale` 亦可执行make来编译。  
**运行**：./player 多媒体路径/XXX.mp4  
![Image text](https://raw.githubusercontent.com/ruokaic/player/main/%E7%A8%8B%E5%BA%8F%E8%BF%90%E8%A1%8C.png)

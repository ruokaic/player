# player
a simple player

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

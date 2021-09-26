#include <stdio.h>
#include <assert.h>
#include <math.h>
#include<pthread.h>
#include "SDL2/SDL.h"

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libswresample/swresample.h"
#include "libavutil/avutil.h"
#include"libavutil/avstring.h"      //for av_strlcpy

// compatibility with newer API
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55,28,1)
#define av_frame_alloc avcodec_alloc_frame
#define av_frame_free avcodec_free_frame
#endif

#define SDL_AUDIO_BUFFER_SIZE 1024
#define MAX_AUDIO_FRAME_SIZE 192000 //channels(2) * data_size(2) * sample_rate(48000)

#define MAX_AUDIOQ_SIZE (5 * 16 * 1024)
#define MAX_VIDEOQ_SIZE (5 * 256 * 1024)

#define AV_SYNC_THRESHOLD 0.01
#define AV_NOSYNC_THRESHOLD 10.0

#define SAMPLE_CORRECTION_PERCENT_MAX 10
#define AUDIO_DIFF_AVG_NB 20

#define FF_REFRESH_EVENT (SDL_USEREVENT)
#define FF_QUIT_EVENT (SDL_USEREVENT + 1)

#define VIDEO_PICTURE_QUEUE_SIZE 1

#define DEFAULT_AV_SYNC_TYPE AV_SYNC_AUDIO_MASTER //AV_SYNC_VIDEO_MASTER

typedef struct PacketQueue {
  AVPacketList *first_pkt, *last_pkt;
  int nb_packets;
  int size;
  SDL_mutex *mutex;
  SDL_cond *cond;
} PacketQueue;


typedef struct VideoPicture {
  AVPicture *bmp;
  int width, height; /* source height & width */
  int allocated;
  double pts;
} VideoPicture;

typedef struct VideoState {

  //multi-media file
  char            filename[1024];
  AVFormatContext *pFormatCtx;
  int             videoStream, audioStream;

  //sync
  int             av_sync_type;
  double          external_clock; /* external clock base */
  int64_t         external_clock_time;

  double          audio_diff_cum; /* used for AV difference average computation */
  double          audio_diff_avg_coef;
  double          audio_diff_threshold;
  int             audio_diff_avg_count;

  double          audio_clock;
  double          frame_timer;
  double          frame_last_pts;
  double          frame_last_delay;

  double          video_clock; ///<pts of last decoded frame / predicted pts of next decoded frame
  double          video_current_pts; ///<current displayed pts (different from video_clock if frame fifos are used)
  int64_t         video_current_pts_time;  ///<time (av_gettime) at which we updated video_current_pts - used to have running video pts

  //audio
  AVStream        *audio_st;
  AVCodecContext  *audio_ctx;
  PacketQueue     audioq;
  uint8_t         audio_buf[(MAX_AUDIO_FRAME_SIZE * 3) / 2];
  unsigned int    audio_buf_size;
  unsigned int    audio_buf_index;
  AVFrame         audio_frame;      //为什么音频的packet和frame是成员变量，而视频的却用临时变量(见decode_video_thread)？
  AVPacket        audio_pkt;
  uint8_t         *audio_pkt_data;
  int             audio_pkt_size;


  //video
  AVStream        *video_st;
  AVCodecContext  *video_ctx;
  PacketQueue     videoq;
  struct SwsContext *video_sws_ctx;
  struct SwrContext *audio_swr_ctx;

  VideoPicture    pictq[VIDEO_PICTURE_QUEUE_SIZE];
  int             pictq_size, pictq_rindex, pictq_windex;   //readindex,writeindex？？
  SDL_mutex       *pictq_mutex;
  SDL_cond        *pictq_cond;

  SDL_Thread      *parse_tid;
  SDL_Thread      *video_tid;

  int             quit;
  double duration;          //打印进度用
} VideoState;

//SDL_mutex    *text_mutex;
SDL_Window   *win = NULL;
SDL_Renderer *renderer;
SDL_Texture  *texture;

enum {
  AV_SYNC_AUDIO_MASTER,
  AV_SYNC_VIDEO_MASTER,
  AV_SYNC_EXTERNAL_MASTER,
};

static int screen_left = SDL_WINDOWPOS_CENTERED;
static int screen_top = SDL_WINDOWPOS_CENTERED;
static int screen_width = 0;
static int screen_height = 0;
static int resize = 1;

/* Since we only have one decoding thread, the Big Struct
   can be global in case we need it. */
VideoState *global_video_state;

void packet_queue_init(PacketQueue *q) {
  memset(q, 0, sizeof(PacketQueue));
  q->mutex = SDL_CreateMutex();
  q->cond = SDL_CreateCond();
}
int packet_queue_put(PacketQueue *q, AVPacket *pkt) {

  AVPacketList *pkt1;
  if(av_dup_packet(pkt) < 0) {
    return -1;
  }
  pkt1 = av_malloc(sizeof(AVPacketList));
  if (!pkt1)
    return -1;
  pkt1->pkt = *pkt;     //指向read_frame申请的内存的地址会被拷贝到队列的packet中
  pkt1->next = NULL;

  SDL_LockMutex(q->mutex);

  if (!q->last_pkt)
    q->first_pkt = pkt1;
  else
    q->last_pkt->next = pkt1;
  q->last_pkt = pkt1;
  q->nb_packets++;
  q->size += pkt1->pkt.size;
  SDL_CondSignal(q->cond);  //生产者

  SDL_UnlockMutex(q->mutex);
  return 0;
}

int packet_queue_get(PacketQueue *q, AVPacket *pkt, int block)
{
  AVPacketList *pktList;
  int ret;

  SDL_LockMutex(q->mutex);

  for(;;) {

    if(global_video_state->quit) {
      ret = -1;
      break;
    }

    pktList = q->first_pkt;
    if (pktList) {
      q->first_pkt = pktList->next;
      if (!q->first_pkt)
    q->last_pkt = NULL;
      q->nb_packets--;
      q->size -= pktList->pkt.size;
      *pkt = pktList->pkt;
      av_free(pktList);
      ret = 1;
      break;
    } else if (!block) {
      ret = 0;
      break;
    } else {
      SDL_CondWait(q->cond, q->mutex);      //消费者阻塞等待数据
    }
  }
  SDL_UnlockMutex(q->mutex);
  return ret;
}

double get_audio_clock(VideoState *is) {
  double pts,pts2;
  int hw_buf_size, bytes_per_sec;

  //pts = is->audio_clock; /* maintained in the audio thread */
  pts2 = is->audio_clock;
  //hw_buf_size = is->audio_buf_size - is->audio_buf_index;
  bytes_per_sec = 0;

  if(is->audio_st) {
    bytes_per_sec = is->audio_ctx->sample_rate * is->audio_ctx->channels * 2;   //播放码流（字节每秒） = 采样率*声道数*2byte(16位)
  }
  if(bytes_per_sec) {
    //pts -= (double)hw_buf_size / bytes_per_sec;   //因为audio_clock = 当前帧PTS + 当前音频帧播放时长，实际当前帧未必刚好播放完，应减去缓冲区未播的部分时长；
    pts2 += (double)is->audio_buf_index / bytes_per_sec;    //音频播放时间 = audio_clock + 缓冲区已播放数据时长
  }
  //fprintf(stderr,"pts1 = %f,pts2 = %f\n",pts,pts2);
  //return pts;
  return pts2;
}
double get_video_clock(VideoState *is) {
  double delta;

  delta = (av_gettime() - is->video_current_pts_time) / 1000000.0;
  return is->video_current_pts + delta;
}
double get_external_clock(VideoState *is) {
  return av_q2d(av_get_time_base_q()) / 1000000.0;  //av_gettime()
}

double get_master_clock(VideoState *is) {
  if(is->av_sync_type == AV_SYNC_VIDEO_MASTER) {
    return get_video_clock(is);
  } else if(is->av_sync_type == AV_SYNC_AUDIO_MASTER) {
    return get_audio_clock(is);
  } else {
    return get_external_clock(is);
  }
}


/* Add or subtract samples to get a better sync, return new
   audio buffer size */

//int synchronize_audio(VideoState *is, short *samples,int samples_size, double pts)    //pts根本没用到
int synchronize_audio(VideoState *is, short *samples,int samples_size)  //音频同步到视频才用
{
  int n;
  double ref_clock;

  n = 2 * is->audio_ctx->channels;

  if(is->av_sync_type != AV_SYNC_AUDIO_MASTER) {
    double diff, avg_diff;
    int wanted_size, min_size, max_size /*, nb_samples */;

    ref_clock = get_master_clock(is);
    diff = get_audio_clock(is) - ref_clock;

    if(diff < AV_NOSYNC_THRESHOLD) {
      // accumulate the diffs
      is->audio_diff_cum = diff + is->audio_diff_avg_coef
    * is->audio_diff_cum;
      if(is->audio_diff_avg_count < AUDIO_DIFF_AVG_NB) {
    is->audio_diff_avg_count++;
      } else {
    avg_diff = is->audio_diff_cum * (1.0 - is->audio_diff_avg_coef);
    if(fabs(avg_diff) >= is->audio_diff_threshold) {
      wanted_size = samples_size + ((int)(diff * is->audio_ctx->sample_rate) * n);
      min_size = samples_size * ((100 - SAMPLE_CORRECTION_PERCENT_MAX) / 100);
      max_size = samples_size * ((100 + SAMPLE_CORRECTION_PERCENT_MAX) / 100);
      if(wanted_size < min_size) {
        wanted_size = min_size;
      } else if (wanted_size > max_size) {
        wanted_size = max_size;
      }
      if(wanted_size < samples_size) {
        /* remove samples */
        samples_size = wanted_size;
      } else if(wanted_size > samples_size) {
        uint8_t *samples_end, *q;
        int nb;

        /* add samples by copying final sample*/
        nb = (samples_size - wanted_size);
        samples_end = (uint8_t *)samples + samples_size - n;
        q = samples_end + n;
        while(nb > 0) {
          memcpy(q, samples_end, n);
          q += n;
          nb -= n;
        }
        samples_size = wanted_size;
      }
    }
      }
    } else {
      /* difference is TOO big; reset diff stuff */
      is->audio_diff_avg_count = 0;
      is->audio_diff_cum = 0;
    }
  }
  return samples_size;
}

//音频解码、重采样、获取audio_clock
//int audio_decode_frame(VideoState *is, uint8_t *audio_buf, int buf_size, double *pts_ptr)
int audio_decode_frame(VideoState *is, uint8_t *audio_buf, int buf_size)
{
  int len1, data_size = 0;
  AVPacket *pkt = &is->audio_pkt;
  double pts;


  for(;;) {
    //while(is->audio_pkt_size > 0) {
      if(is->audio_pkt_size > 0) {  //先用，用完，再从队列拿包，第一次进入函数audio_pkt_size大小为0，从packet_queue_get开始执行
      //is->audio_pkt_size 是packet的大小，len1是packet输入消耗的字节数，is->audio_pkt_size -= len1，只有当前packet全部解码完毕才会读取下一个packet
      int got_frame = 0;
      len1 = avcodec_decode_audio4(is->audio_ctx, &is->audio_frame, &got_frame, pkt);
      if(len1 < 0) {
    /* if error, skip frame */
    is->audio_pkt_size = 0;
    break;
      }
      data_size = 0;
      if(got_frame) {
        /*
    data_size = av_samples_get_buffer_size(NULL,
                           is->audio_ctx->channels,
                           is->audio_frame.nb_samples,
                           is->audio_ctx->sample_fmt,
                           1);
        */
         //确定输入缓冲区的数据大小，nb_samples值指单个通道中的某个时间范围内的采样个数，nb_sample *2（双声道）*2（16位采样大小）= 总数据量data_size;
        data_size = 2 * is->audio_frame.nb_samples * 2;
    assert(data_size <= buf_size);
        swr_convert(is->audio_swr_ctx,
                        &audio_buf,     //重采样到audio_buf
                        MAX_AUDIO_FRAME_SIZE*3/2,
                        (const uint8_t **)is->audio_frame.data,
                        is->audio_frame.nb_samples);

    //memcpy(audio_buf, is->audio_frame.data[0], data_size);    //不进行重采样直接拷贝
      }
      is->audio_pkt_data += len1;
      is->audio_pkt_size -= len1;
      if(data_size <= 0) {
    /* No data yet, get more frames */
    continue;
      }
      pts = is->audio_clock;    //音频帧的起始时间
      //*pts_ptr = pts;

      /* 总数据量/时间 = 码率，码率（字节）= 采样频率samplerate *2（双声道）*2（16位采样大小）;
      * nb_sample/sample_rate = 时间(采样时间或播放时间)
      *
      * 音频包packet可能解码后含有多个帧，取pakcet的pts，只能代表第一个音频帧开始播放的时间
      * 音频帧的播放时长 = 一帧的采样个数 / 采样频率
      * 此处可理解为：当前音频播放时长 = pts + 当前音频包解码出的所有音频帧frame的播放时长（一个解码帧数据大小/码率（字节）也可化简为nb_samples/sample_rate）；
      * 当前音频播放时长并不精确，取的是frame播放完毕的时间点，实际应该取pts + frame播放时长中的一个点，这个精确时间 = 在音视频同步时取当前audio_clock减去未读入buffer的数据播放时长*/
      //is->audio_clock += (double)data_size /(double)(2 * is->audio_ctx->channels * is->audio_ctx->sample_rate);   //上述注释对应此行代码，目前已更换策略，详见更新日志


      /* We have data, return it and come back for more later */
      return data_size;
    }
    if(pkt->data)
      av_free_packet(pkt);  //记得释放在av_read_frame申请的内存

    if(is->quit) {
      return -1;
    }
    /* next packet */
    if(packet_queue_get(&is->audioq, pkt, 1) < 0) {
      return -1;
    }
    is->audio_pkt_data = pkt->data;
    is->audio_pkt_size = pkt->size;
    /* if update, update the audio clock w/pts */
    if(pkt->pts != AV_NOPTS_VALUE) {
      is->audio_clock = av_q2d(is->audio_st->time_base)*pkt->pts;   //获取audio_clock，解码后取frame的PTS一样可行
    }
  }
}

//SDL回调函数，调用audio_decode_frame解码音频帧，并通过SDL_MixAudio*****播放音频
void audio_callback(void *userdata, Uint8 *stream, int len)
{
    //userdata为用户传入参数，stream指向声卡缓冲区，len为声卡索求的数据大小（声卡缓冲区空闲大小）

    VideoState *is = (VideoState *)userdata;
    int len1, audio_size;
    //double pts;
    SDL_memset(stream, 0, len); //固定操作，先清空声卡的缓冲区

    while(len > 0)  //大于0表示声卡缓冲区有空位
    {
        if(is->audio_buf_index >= is->audio_buf_size) //若外部的缓冲区数据已经发完，再从队列读包解码填充缓冲区，audio_buf_index和audio_buf_size初始值0
        {
            /* We have already sent all our data; get more */
            //audio_size = audio_decode_frame(is, is->audio_buf, sizeof(is->audio_buf), &pts);
            audio_size = audio_decode_frame(is, is->audio_buf, sizeof(is->audio_buf));      //解码数据存在audio_buf中
            if(audio_size < 0) {
                /* If error, output silence */
                is->audio_buf_size = 1024 * 2 * 2;
                memset(is->audio_buf, 0, is->audio_buf_size);   //解码失败。置0来播放静默音
            } else {
                //audio_size = synchronize_audio(is, (int16_t *)is->audio_buf,audio_size, pts);
                audio_size = synchronize_audio(is, (int16_t *)is->audio_buf,audio_size);    //视频同步到音频时，函数不起作用，直接返回audio_size
                is->audio_buf_size = audio_size;
            }
            is->audio_buf_index = 0;
        }
        len1 = is->audio_buf_size - is->audio_buf_index;    //发给声卡缓冲区的数据大小
        if(len1 > len)  //最多只能发送声卡缓冲区大小的数据
            len1 = len;
        SDL_MixAudio(stream,(uint8_t *)is->audio_buf + is->audio_buf_index, len1, SDL_MIX_MAXVOLUME);   //从audio_buf当前指针位置读取已解码音频帧到声卡缓冲区，使用memcpy也行
        //memcpy(stream, (uint8_t *)is->audio_buf + is->audio_buf_index, len1);
        len -= len1;
        stream += len1;
        is->audio_buf_index += len1;
    }
}

// 2、发出FF_REFRESH_EVENT来调用video_refresh_timer
static Uint32 sdl_refresh_timer_cb(Uint32 interval, void *opaque)
{
  //fprintf(stderr,"timer pid:%ld\n",pthread_self());
  SDL_Event event;
  event.type = FF_REFRESH_EVENT;
  event.user.data1 = opaque;
  SDL_PushEvent(&event);
  return 0; /* 0 means stop timer */
}

/* schedule a video refresh in 'delay' ms */
//1、XXXms后调用一次（仅一次）回调函数sdl_refresh_timer_cb，调了帧数会导致音视频不同步？
static void schedule_refresh(VideoState *is, int delay)
{
  SDL_AddTimer(delay, sdl_refresh_timer_cb, is);
}

//4、把picture传给SDL进行渲染，实现*******播放视频
void video_display(VideoState *is)
{
  fprintf(stderr,"%5.2f%%\n",100*(is->video_clock/(global_video_state->duration/1000000)));
  SDL_Rect rect;
  VideoPicture *vp;
//  float aspect_ratio;
  int w, h;
//  int i;

  if(screen_width && resize){   //resize令其只执行一次
//      SDL_SetWindowSize(win, screen_width, screen_height);
//      SDL_SetWindowPosition(win, screen_left, screen_top);
//      SDL_ShowWindow(win);
      //IYUV: Y + U + V  (3 planes)
      //YV12: Y + V + U  (3 planes)
      Uint32 pixformat= SDL_PIXELFORMAT_IYUV;

      //create texture for render
      texture = SDL_CreateTexture(renderer,
              pixformat,
              SDL_TEXTUREACCESS_STREAMING,
              screen_width,
              screen_height);
      resize = 0;
  }

  vp = &is->pictq[is->pictq_rindex];
  if(vp->bmp) {

    SDL_UpdateYUVTexture( texture, NULL,
                          vp->bmp->data[0], vp->bmp->linesize[0],
                          vp->bmp->data[1], vp->bmp->linesize[1],
                          vp->bmp->data[2], vp->bmp->linesize[2]);

    SDL_GetWindowSize(win, &w, &h);
    rect.x = 0;
    rect.y = 0;
    rect.w = w;
    rect.h = h;
    //SDL_LockMutex(text_mutex);
    SDL_RenderClear( renderer );
    SDL_RenderCopy( renderer, texture, NULL, &rect);
    SDL_RenderPresent( renderer );
    //SDL_UnlockMutex(text_mutex);
  }
}

//3、必定反复调用schedule_refresh来实现视频帧刷新，同时通过delay来实现音视频同步；若picture队列不为空，则调用video_display
void video_refresh_timer(void *userdata)
{

    VideoState *is = (VideoState *)userdata;
    VideoPicture *vp;
    double actual_delay, delay, sync_threshold, ref_clock, diff;

    if(is->video_st)    //判断视频流有无错误，如源视频流缺漏（只有音频）或格式错误等情况
    {
        if(is->pictq_size == 0) {
            schedule_refresh(is, 1);    //1ms间隔不断刷新等待视频解码线程生产数据
            //fprintf(stderr, "no picture in the queue!!!\n");
        }
        else {
            //fprintf(stderr, "get picture from queue!!!\n");
            /*如何进行同步：
             * 采用视频同步到音频的方式，音频不动，调控的对象是视频帧的刷新间隔；
             *     先取得当前视频帧与上一帧的播放间隔delay作为下一帧间隔的参考，再用
             * 当前视频播放时间和当前音频播放时间作差得到diff差值，通过diff与delay
             * 对比，要求 差值大于-delay，小于+delay，也就是说，音频和视频的播放进
             * 度差值，不能超过视频帧间隔，如果超了，比如音频播放比视频快，那我就要加
             * 快视频帧的刷新，将下一帧间隔设为0，反之放慢下一帧视频帧刷新间隔，设为
             * 2*delay；
             *     到这里已经实现基本的同步了，但是，虽然我们参考的是音频帧，正常情
             * 况下音频时间间隔是固定的，但也有可能出现偏差导致音频播放时长与实际播放
             * 时长不同的情况，为了视频播放时长与实际时长吻合，还要进一步参考系统时间：
             *     我们在视频开始播放时，获取系统时间，把每次得出的delay值在此时间上
             * 进行不断累加，得出一个理论上下一帧的播放时间，再减去当前系统时间，得出
             * 下一帧刷新间隔（包含了与实际播放时长的误差修正值）*/
            vp = &is->pictq[is->pictq_rindex];

            is->video_current_pts = vp->pts;    //解码帧对应的时间戳，对于没有时间戳的解码帧，pts = video_clock
            is->video_current_pts_time = av_gettime();  //不关心，音频同步到视频用
            delay = vp->pts - is->frame_last_pts;   //当前pts减上次pts得出上一帧播放间隔，作为下一帧间隔的参考，delay单位秒
            if(delay <= 0 || delay >= 1.0) {
                /* if incorrect delay, use previous one */
                delay = is->frame_last_delay;   //初值为0.04s
            }
            /* save for next time */
            is->frame_last_delay = delay;
            is->frame_last_pts = vp->pts;

            /* update delay to sync to audio if not master source */
            if(is->av_sync_type != AV_SYNC_VIDEO_MASTER) {
                ref_clock = get_master_clock(is);   //调用get_audio_clock获取音频时间，而音频时间在音频解码时获得
                diff = vp->pts - ref_clock;     //得出视频与音频播放的时间差

                /* Skip or repeat the frame. Take delay into account
       FFPlay still doesn't "know if this is the best guess." */
                sync_threshold = (delay > AV_SYNC_THRESHOLD) ? delay : AV_SYNC_THRESHOLD;       //确定最低视频帧刷新间隔，不能低于AV_SYNC_THRESHOLD = 10毫秒
                if(fabs(diff) < AV_NOSYNC_THRESHOLD) {      //如果diff绝对值小于非同步阀值（10s）,说明还有救，可以进行同步来调整
                    if(diff <= -sync_threshold) {     //如果画面比声音慢，超过两帧播放间隔，加快视频刷新
                        delay = 0;
                    } else if(diff >= sync_threshold) {       //如果画面比声音快，超过两帧播放间隔
                        delay = 2 * delay;
                    }
                }
            }
            is->frame_timer += delay;         //frame_timer在stream_component_open初始化为系统时间
            /* computer the REAL delay */
            actual_delay = is->frame_timer - (av_gettime() / 1000000.0);
            if(actual_delay < 0.010) {
                /* Really it should skip the picture instead */
                actual_delay = 0.010;   //最小间隔10ms
            }
            fprintf(stderr,"actual_delay:%f    delay:%f\n",actual_delay,delay);
            schedule_refresh(is, (int)(actual_delay * 1000 + 0.5));   //乘1000换算为毫秒，加0.5四舍五入，refresh启用了其他线程，不会影响接下来调用的videoplay

            //fprintf(stderr,"refresh pid:%ld\n",pthread_self());   //测试得出schedule_refresh启用其他线程，不会影响video_display
            /* show the picture! */
            video_display(is);                                  ////////////////video_display//////////////////////

            /* update queue for next picture! */
            if(++is->pictq_rindex == VIDEO_PICTURE_QUEUE_SIZE) {
                is->pictq_rindex = 0;
            }
            SDL_LockMutex(is->pictq_mutex);
            is->pictq_size--;                   //消费者
            SDL_CondSignal(is->pictq_cond);
            SDL_UnlockMutex(is->pictq_mutex);
        }
    }
    else {
        schedule_refresh(is, 100);  //类似demux_thread结尾
    }
}

void alloc_picture(void *userdata)
{
  int ret;

  VideoState *is = (VideoState *)userdata;
  VideoPicture *vp;

  vp = &is->pictq[is->pictq_windex];
  if(vp->bmp) {

    // we already have one make another, bigger/smaller
    avpicture_free(vp->bmp);
    free(vp->bmp);

    vp->bmp = NULL;
  }

  // Allocate a place to put our YUV image on that screen
  //SDL_LockMutex(text_mutex);

  vp->bmp = (AVPicture*)malloc(sizeof(AVPicture));
  ret = avpicture_alloc(vp->bmp, AV_PIX_FMT_YUV420P, is->video_ctx->width, is->video_ctx->height);
  if (ret < 0) {
      fprintf(stderr, "Could not allocate temporary picture: %s\n", av_err2str(ret));
  }

  //SDL_UnlockMutex(text_mutex);

  vp->width = is->video_ctx->width;
  vp->height = is->video_ctx->height;
  vp->allocated = 1;

}

//*4、将解码视频帧AVFrame格式转换后存入picture数组
int queue_picture(VideoState *is, AVFrame *pFrame, double pts)
{

  VideoPicture *vp;

  /* wait until we have space for a new pic */
  SDL_LockMutex(is->pictq_mutex);
  while(is->pictq_size >= VIDEO_PICTURE_QUEUE_SIZE && !is->quit)    //生产者
  {
    SDL_CondWait(is->pictq_cond, is->pictq_mutex);  //数据已满，等待消费者取数据
  }
  SDL_UnlockMutex(is->pictq_mutex);

  if(is->quit)
    return -1;

  // windex is set to 0 initially
  vp = &is->pictq[is->pictq_windex];

  /* allocate or resize the buffer! */
  //一开始picture队列中的AVPicture没有初始化才会触发（分辨率大小发生变化也需要重新初始化，这个条件没有触发过
  if(!vp->bmp ||vp->width != is->video_ctx->width ||vp->height != is->video_ctx->height)
  {
    vp->allocated = 0;  //这个参数不起作用
    alloc_picture(is);  //只在程序开始运行时调用一次，为AVPicture*bmp分配空间，初始化
    if(is->quit) {
      return -1;
    }
  }

  /* We have a place to put our picture on the queue */
  if(vp->bmp) {     //将时间戳和picture存入pictured队列

    vp->pts = pts;

    // Convert the image into YUV format that SDL uses
    sws_scale(is->video_sws_ctx, (uint8_t const * const *)pFrame->data,        //pFrame->data到vp->bmp->data
          pFrame->linesize, 0, is->video_ctx->height,
          vp->bmp->data, vp->bmp->linesize);

    /* now we inform our display thread that we have a pic ready */
    if(++is->pictq_windex == VIDEO_PICTURE_QUEUE_SIZE) {
      is->pictq_windex = 0;
    }

    SDL_LockMutex(is->pictq_mutex);
    is->pictq_size++;
    SDL_UnlockMutex(is->pictq_mutex);
  }
  return 0;
}

double synchronize_video(VideoState *is, AVFrame *src_frame, double pts)
{
    /* video_clock变量是为没有时间戳的frame服务的
     * synchronize_video不断更新推算出来的下一帧播放时间video_clock，遇到没有时间戳的frame，
     * 就用上次推算的video_clock（对应当前frame的播放时间）作为他对应的pts*/
  double frame_delay;

  if(pts != 0) {
    /* if we have pts, set video clock to it */
    is->video_clock = pts;
  } else {
    /* if we aren't given a pts, set it to the clock */
    pts = is->video_clock;      //将上次的pts + frame_delay赋值给它
    fprintf(stderr,"pts = 0\n");
  }
  /* update the video clock */
  frame_delay = av_q2d(is->video_ctx->time_base);       //按照time_base计算出相应帧率下帧之间的间隔。此为一般情况下的帧延迟
  /* if we are repeating a frame, adjust clock accordingly */
  frame_delay += src_frame->repeat_pict * (frame_delay * 0.5);      //加上了它的附加延迟。FFmpeg给出了公式：extra_delay = repeat_pict / (2*fps)。相加即为其的总延迟。
  is->video_clock += frame_delay;
  return pts;
}

//*3、队列取出视频包调用avcodec_decode_video2解码
int decode_video_thread(void *arg)
{
  VideoState *is = (VideoState *)arg;
  AVPacket pkt1, *packet = &pkt1;
  int frameFinished;
  AVFrame *pFrame;
  double pts;

  pFrame = av_frame_alloc();

  for(;;) {
      if(packet_queue_get(&is->videoq, packet, 1) < 0) {
          // means we quit getting packets
          break;
      }
      pts = 0;

      // Decode video frame
      avcodec_decode_video2(is->video_ctx, pFrame, &frameFinished, packet);

      /*从解码帧获取时间戳，会有极个别解码出的frame时间戳不是有效值，则将其时间戳设为0（为什么会有frame没有时间戳？
       * 为什么不用frame->pts？（不能用packet的pts，那是包不是图像帧）,av_frame_get_best_effort_timestamp会做逻辑处理，比frame->pts更精准*/
      if((pts = av_frame_get_best_effort_timestamp(pFrame)) == AV_NOPTS_VALUE)
          pts = 0;
      pts *= av_q2d(is->video_st->time_base);   //时间戳乘上时间基，得出时间戳（秒），即该帧的显示时间

      // 只有是完整的解码帧，时间戳和frame才会入队
      if(frameFinished) {
          pts = synchronize_video(is, pFrame, pts); //处理没有时间戳的frame
          if(queue_picture(is, pFrame, pts) < 0) {  //将解码帧和最终得出的时间戳存入picture队列
              break;
          }
      }
      av_free_packet(packet);   //释放read_frame申请的内存
  }
  av_frame_free(&pFrame);
  return 0;
}

//*2、根据编码器上下文判断是音频还是视频，初始化音频包或视频包的队列，打开编码器、声卡等并初始化参数，重采样初始化，创建解码线程
int stream_component_open(VideoState *is, int stream_index)
{
  AVFormatContext *pFormatCtx = is->pFormatCtx;
  AVCodecContext *codecCtx = NULL;
  AVCodec *codec = NULL;
  SDL_AudioSpec spec;

  if(stream_index < 0 || stream_index >= pFormatCtx->nb_streams) {
    return -1;
  }

  codecCtx = avcodec_alloc_context3(NULL);

//  旧的版本直接通过如下的代码获取到AVCodecContext结构体参数：
//  codecCtx = pFormatCtx->streams[stream_index]->codec;
  int ret = avcodec_parameters_to_context(codecCtx, pFormatCtx->streams[stream_index]->codecpar);
  if (ret < 0)
    return -1;

  codec = avcodec_find_decoder(codecCtx->codec_id);
  if(!codec) {
    fprintf(stderr, "Unsupported codec!\n");
    return -1;
  }

  if(avcodec_open2(codecCtx, codec, NULL) < 0) {
    fprintf(stderr, "Unsupported codec!\n");
    return -1;
  }

  if(codecCtx->codec_type == AVMEDIA_TYPE_AUDIO) {  //仅音频流，设置SDL相关音频输出格式和回调函数

      // Set audio settings from codec info
      spec.freq = codecCtx->sample_rate;
      spec.format = AUDIO_S16SYS;
      spec.channels = 2;//codecCtx->channels;
      spec.silence = 0;
      spec.samples = SDL_AUDIO_BUFFER_SIZE;
      spec.callback = audio_callback;
      spec.userdata = is;

      fprintf(stderr, "wanted spec: channels:%d, sample_fmt:%d, sample_rate:%d \n",
            2, AUDIO_S16SYS, codecCtx->sample_rate);

      if(SDL_OpenAudio(&spec, NULL) < 0) {
          fprintf(stderr, "SDL_OpenAudio: %s\n", SDL_GetError());
          return -1;
      }
  }

  switch(codecCtx->codec_type) {
  case AVMEDIA_TYPE_AUDIO:
    //is->audioStream = stream_index;
    is->audio_st = pFormatCtx->streams[stream_index];
    is->audio_ctx = codecCtx;
    is->audio_buf_size = 0;
    is->audio_buf_index = 0;
    memset(&is->audio_pkt, 0, sizeof(is->audio_pkt));
    packet_queue_init(&is->audioq);

    //Out Audio Param
    uint64_t out_channel_layout=AV_CH_LAYOUT_STEREO;

    //AAC:1024  MP3:1152
//    int out_nb_samples= is->audio_ctx->frame_size;
    //AVSampleFormat out_sample_fmt = AV_SAMPLE_FMT_S16;

    int out_sample_rate=is->audio_ctx->sample_rate;
    int out_channels=av_get_channel_layout_nb_channels(out_channel_layout);
    //Out Buffer Size
    /*
    int out_buffer_size=av_samples_get_buffer_size(NULL,
                                                   out_channels,
                                                   out_nb_samples,
                                                   AV_SAMPLE_FMT_S16,
                                                   1);
                                                   */

    //uint8_t *out_buffer=(uint8_t *)av_malloc(MAX_AUDIO_FRAME_SIZE*2);
    int64_t in_channel_layout=av_get_default_channel_layout(is->audio_ctx->channels);

    struct SwrContext *audio_convert_ctx;
    audio_convert_ctx = swr_alloc();
    swr_alloc_set_opts(audio_convert_ctx,
                       out_channel_layout,
                       AV_SAMPLE_FMT_S16,
                       out_sample_rate,
                       in_channel_layout,
                       is->audio_ctx->sample_fmt,
                       is->audio_ctx->sample_rate,
                       0,
                       NULL);
    fprintf(stderr, "swr opts: out_channel_layout:%lld, out_sample_fmt:%d, out_sample_rate:%d, in_channel_layout:%lld, in_sample_fmt:%d, in_sample_rate:%d",
            out_channel_layout, AV_SAMPLE_FMT_S16, out_sample_rate, in_channel_layout, is->audio_ctx->sample_fmt, is->audio_ctx->sample_rate);
    swr_init(audio_convert_ctx);

    is->audio_swr_ctx = audio_convert_ctx;

    SDL_PauseAudio(0);  //开关，开启音频播放
    break;
  case AVMEDIA_TYPE_VIDEO:
    //is->videoStream = stream_index;
    is->video_st = pFormatCtx->streams[stream_index];
    is->video_ctx = codecCtx;

    is->frame_timer = (double)av_gettime() / 1000000.0;
    is->frame_last_delay = 40e-3;
    is->video_current_pts_time = av_gettime();

    packet_queue_init(&is->videoq);
    is->video_sws_ctx = sws_getContext(is->video_ctx->width, is->video_ctx->height,
                 is->video_ctx->pix_fmt, is->video_ctx->width,
                 is->video_ctx->height, AV_PIX_FMT_YUV420P,
                 SWS_BILINEAR, NULL, NULL, NULL
                 );
    is->video_tid = SDL_CreateThread(decode_video_thread, "decode_video_thread", is);
    break;
  default:
    break;
  }
}

//*1、解复用线程，获取音视频信息、编解码器上下文，调用av_read_frame读取文件内容到packet，经过*2之后，通过packet->stream_index将音频和视频分别入队不同的音频包队列和视频包队列
int demux_thread(void *arg)
{

    VideoState *is = (VideoState *)arg;
    AVPacket pkt1, *packet = &pkt1;

  // main decode loop

  for(;;) {
    if(is->quit) {
      break;
    }
    // seek stuff goes here
    if(is->audioq.size > MAX_AUDIOQ_SIZE ||     //队列满了循环等待
       is->videoq.size > MAX_VIDEOQ_SIZE) {
      SDL_Delay(10);
      continue;
    }
    if(av_read_frame(is->pFormatCtx, packet) < 0) {     //申请的内存需要在解码后才能释放
      if(is->pFormatCtx->pb->error == 0) {
    SDL_Delay(100); /* no error; wait for user input */
    continue;
      } else {
    break;
      }
    }
    // Is this a packet from the video stream?
    if(packet->stream_index == is->videoStream) {    //stream_index从哪来的，为什么可以和videosteam对应？videoStream本来就是从streams[i]获得的，ffmpeg内部stream_index和i的值一致
      packet_queue_put(&is->videoq, packet);
    } else if(packet->stream_index == is->audioStream) {
      packet_queue_put(&is->audioq, packet);
    } else {
      av_free_packet(packet);
    }
  }
  /* all done - wait for it */
  while(!is->quit) {
    SDL_Delay(100);
  }
  if(1){
    SDL_Event event;
    event.type = FF_QUIT_EVENT;
    event.user.data1 = is;
    SDL_PushEvent(&event);
  }
  return 0;
}

int open_input(VideoState *is)
{
    int err_code;
    char errors[1024] = {0,};

    //int w, h;

    AVFormatContext *pFormatCtx = NULL;

    int i;

    is->videoStream=-1;
    is->audioStream=-1;

    global_video_state = is;

    /* open input file, and allocate format context */
    if ((err_code=avformat_open_input(&pFormatCtx, is->filename, NULL, NULL)) < 0) {
        av_strerror(err_code, errors, 1024);
        fprintf(stderr, "Could not open source file %s, %d(%s)\n", is->filename, err_code, errors);
        return -1;
    }

    is->pFormatCtx = pFormatCtx;

    // Retrieve stream information
    if(avformat_find_stream_info(pFormatCtx, NULL)<0)
      return -1; // Couldn't find stream information

    // Dump information about file onto standard error
    av_dump_format(pFormatCtx, 0, is->filename, 0);

    // Find the first video stream
    //新版用av_find_best_stream返回索引
    for(i=0; i<pFormatCtx->nb_streams; i++)
    {
      if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO &&
         is->videoStream < 0) {
        is->videoStream=i;
      }
      if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_AUDIO &&
         is->audioStream < 0) {
        is->audioStream=i;
      }
    }
    if(is->videoStream < 0 || is->audioStream < 0) {
      fprintf(stderr, "%s: could not open codecs\n", is->filename);
      return -1;
    }

    if(is->audioStream >= 0) {
      stream_component_open(is, is->audioStream);
    }
    if(is->videoStream >= 0) {
      stream_component_open(is, is->videoStream);
    }

    screen_width = is->video_ctx->width;
    screen_height = is->video_ctx->height;
    is->duration = pFormatCtx->duration;
    return 0;
}

int main(void)
{
  SDL_Event       event;
  VideoState      *is;

  is = av_mallocz(sizeof(VideoState));  //分配内存后置0

  // Register all formats and codecs
  //av_register_all();

  if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER)) {
    fprintf(stderr, "Could not initialize SDL - %s\n", SDL_GetError());
    exit(1);
  }

  char path[] = "test1.mp4";
  //char path[] = "test2.flv";
  //char path[] = "test3.mp3";
  //char path[] = "test4.ts";

  av_strlcpy(is->filename, path, sizeof (is->filename));    //ffmpeg自带strcpy，#include"libavutil/avstring.h"

  //open file
  int ret = open_input(is);
  if(ret < 0)
  {
      fprintf(stderr, "Loading file failed!\n");
      exit(1);
  }

  //creat window from SDL
  win = SDL_CreateWindow("Media Player",
                         100,
                         100,
                         is->video_ctx->width, is->video_ctx->height,
                         SDL_WINDOW_RESIZABLE);
  if(!win) {
      fprintf(stderr, "\nSDL: could not set video mode:%s - exiting\n", SDL_GetError());
      exit(1);
  }

  renderer = SDL_CreateRenderer(win, -1, 0);

  //text_mutex = SDL_CreateMutex();

  is->pictq_mutex = SDL_CreateMutex();
  is->pictq_cond = SDL_CreateCond();

 //应注意与视频实际帧率匹配
  schedule_refresh(is, 40);     //开关，开启视频刷新
  is->av_sync_type = DEFAULT_AV_SYNC_TYPE;  //使用默认同步策略：视频同步到音频
  is->parse_tid = SDL_CreateThread(demux_thread,"demux_thread", is);
  if(!is->parse_tid) {
    av_free(is);
    return -1;
  }
  for(;;) {

    SDL_WaitEvent(&event);
    switch(event.type) {
    case FF_QUIT_EVENT:
    case SDL_QUIT:
      is->quit = 1;     //其他线程每次循环优先检测quit的值，为1则退出线程
      SDL_CondSignal(is->audioq.cond);      //如果退出时其他线程刚好在阻塞，则这些线程将无法结束
      SDL_CondSignal(is->pictq_cond);
      SDL_Quit();
      return 0;
      break;
    case FF_REFRESH_EVENT:
      video_refresh_timer(event.user.data1);
      break;
    default:
      break;
    }
  }
  return 0;
}

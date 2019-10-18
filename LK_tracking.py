"""
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack

for track initialization and back-tracking for match verification

between frames.


Usage

-----

lk_track.py [<video_source>]


Keys

----

ESC - exit

"""

import numpy as np
import cv2
import video
import torch

from common import anorm2, draw_str
from time import clock

#  如下是lucas kanade 参数
#  winSize      搜索方框的大小（邻域大小），即对15*15个点进行求解，maxLevel 最大的金字塔层数,
#  maxCorners   最大的角点数，如果检测出的角点多余最大角点数，将取出最强最大角点数个角点
#  qualityLevel 反应一个像素点强度有多强才能成为关键点 minDistance 关键点之间的最少像素点。
#  blocksize    计算一个像素点是否为关键点时所取的区域大小。
#  TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT ：要么达到了最大迭代次数10，要么按达到阈值0.03作为收敛结束条件。
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# 构建角点检测所需参数，在minDistance = 7 这个范围内只存在一个品质最好的角点
feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)


class App:

    def __init__(self, video_src):   # 构造方法，初始化一些参数和视频路径
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)                                  # 打开摄像头
        self.frame_idx = 0

    def run(self):  # 光流运行方法

        while True:
            # self.cam.read()按帧读取视频，ret,frame是获self.cam.read()方法的两个返回值。其中ret是布尔值，如果读取帧是正确的则
            # 返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                     # 转化为灰度虚图，作为前一帧图片
            vis = frame.copy()                                                       # 赋值frame的值，不覆盖frame本身
            print('len_self.tracks:', len(self.tracks))
            print('self.tracks：', self.tracks)                                      # self.tracks是列表类型，（x，y）
            if len(self.tracks) > 0:                                                 # 检测到角点后进行光流跟踪
                img0, img1 = self.prev_gray, frame_gray                              # 上一帧和这一帧
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)    # p0_size (10, 1, 2) 10
                print('p0_size', p0.shape, len(p0), p0)

                # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置，用于获得光流检测后的角点位置
                # p1 表示角点的位置，st表示是否是运动的角点，err表示是否出错
                # old_gray表示输入前一帧图片，frame_gray表示后一帧图片，p0表示前一帧检测到的角点，lk_params：winSize表示选择多少个
                # 点进行u和v的求解，maxLevel表示空间金字塔的层数. 作用是：前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

                # 当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                # 得到角点回溯与前一帧实际角点的位置变化关系
                a = abs(p0 - p0r)                             # a_dim: 3
                print('a_dim:', torch.from_numpy(a).dim())
                c = abs(p0 - p0r).reshape(-1, 2)              # 将3维转化为2维  即(10, 1, 2)  ——————> (10, 2)
                print('c_dim:', torch.from_numpy(c).dim())
                print('c', c)
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)        # 在列的维度上 取最大值
                print('d:', d.shape, len(d), d)                 # 此时d是10个最大的数

                # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
                good = d < 1
                print('good:', good.shape, good)
                new_tracks = []

                # 将跟踪正确的点列入成功跟踪点。zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，
                # 然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    print('tr:', tr, '(x, y):', (x, y), 'good_flag:', good_flag)
                    # 筛选出 正确的角点
                    if not good_flag:
                        continue

                    # tr是前一帧的角点，与当前帧正确的角点(x,y)合并。标志为good_flag
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)    # 当前帧角点画圆，原图像， 圆心， 半径， 颜色， 线条类型

                self.tracks = new_tracks                           # self.tracks中的值的格式是：(前一帧角点)(当前帧角点)

                # 以上一振角点为初始点，当前帧跟踪到的点为终点划线
                # cv2.polylines()可以被用来同时画多条线，只需要同时创建多个线段，传入函数中， False表示首尾不封闭
                # 以上一帧角点为初始点，当前帧跟踪到的点为终点划线
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            print('self.frame_idx:', self.frame_idx)
            if self.frame_idx % self.detect_interval == 0:  # 每5帧检测一次特征点
                mask = np.zeros_like(frame_gray)            # 初始化和视频大小相同的图像吗，创建一个mask, 用于进行横线的绘制
                mask[:] = 255                               # 将mask赋值255也就是检测整个图像区域的角点
                print('len_track:', len(self.tracks))
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:  # 跟踪的角点画圆。 刚开始，里面没角点，跳过该循环
                    print('(x,y):', (x, y))
                    cv2.circle(mask, (x, y), 5, 0, -1)
                # 输入第一帧灰度化后的图片，mask：指定检测角的区域。 返回值：角点
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)  # 像素级别角点检测

                print('p:', type(p), p)                            # 10
                if p is not None:

                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])            # 将检测到的角点放在待跟踪序列中  这里将10个角点加入追踪里面

            self.frame_idx += 1
            print('print(self.frame_idx):', self.frame_idx)
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)                         # 第一个参数是窗口的名字，第二个是我们的图像
            ch = 0xFF & cv2.waitKey(1)                          # 按esc退出

            if ch == 27:
                break


def main():
    import sys

    try:
        video_src = sys.argv[1]       # 外部给予参数。此处是摄像头。sys.argv[0]表示代码本身文件路径，返回的是该文件的完整路径
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

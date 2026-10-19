# 2026-10-19 修复与硬件适配

项目不限定 Intel/AMD 品牌或某款 CPU。


| 无法分辨“请求 GPU”和“实际使用 GPU” | `auto` 按平台尝试，失败回退并记录原因；强制 GPU 失败报错；摘要和界面显示实际设备 | GPU 分支用替身测试，本机实测实际为 CPU |
| 已有损坏模型只凭文件存在就使用；中断下载留不完整文件 | 验证 task bundle/CRC；唯一临时文件下载、验证后替换；失败清理本次临时文件，已有坏文件不擅自删除 | 损坏缓存/离线下载回归；Full 官方模型真实下载 |
| 规则能返回无法判断，但畸形关键点仍进入绘图，可能崩溃 | 不可靠姿态不绘骨架；检查结构/有限值/置信度；限制非关键点极端绘图坐标 | 缺失/错误形状/NaN/越界置信度/极大坐标测试 |
| 快速分析时预览帧会丢弃，提醒也可能跟着丢弃 | 事件通知使用独立信号，不依赖有界预览队列；预览依旧只保留最新两帧 | 队列溢出后仍收到事件信号 |
| 运行设置不可复现，依赖启动脚本对 Python 版本不匹配处理不明确 | 运行设置和主机信息进入摘要，GUI 参数可导出/载入；CLI 覆盖；启动时单独校验 Python | 文件配置覆盖、线程恢复、环境启动检查 |

依据：[MediaPipe BaseOptions](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/BaseOptions) 说明 CPU/GPU delegate 和平台限制；[官方模型说明](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) 提供 Lite/Full/Heavy；[视频接口说明](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) 要求逐帧传递时间戳，本项目不按处理快慢改变文件源时间。

[OpenCV 官方线程接口](https://docs.opencv.org/4.x/db/de0/group__core__utils.html) 设置的是 OpenCV 并行区域，不是 MediaPipe 模型线程，且不能并发修改。因此一个进程只处理一个分析会话；GUI 使用 QImage 直接显示 BGR，不再与后台线程并发执行 OpenCV 色彩转换。多路分析需分进程，这次没有扩展成多摄像头系统。

## 回归与实际模型验证

`python -m unittest -v test_system test_integration test_runtime`：**17 项通过**，原有状态机与事件回放流程保持通过。含区域框选、确认/误报、停止收尾、媒体截断和后台线程恢复。当前界面截图见 [interface-preview-revision.png](evidence/interface-preview-revision.png)。

真实模型检查均使用同一静态全身图片重复 90 帧、640×426 输入；不是动作视频、不是跌倒数据集。完整数据见：

| 设置 | 实际设备 | 处理帧 | 无法判断 | 本次处理吞吐 | 报告 |
| --- | --- | ---: | ---: | ---: | --- |
| auto / Lite / OpenCV 2 线程 / 最长边 640 | CPU，记录平台回退原因 | 90 | 0 | 31.05 FPS | [Lite](evidence/model-smoke-revision-lite.json) |
| cpu / Full / OpenCV 1 线程 / 最长边 640 | CPU | 90 | 0 | 26.09 FPS | [Full](evidence/model-smoke-revision-full.json) |

两次设置不同，且测试时存在其它任务负载，**不能拿这两个数作严格模型速度对比，也不能和历史 16.58 FPS 计算“优化百分比”**。不含初始化的逐帧吞吐不等于端到端实时性能；Full 首次还包含下载。数据只证明可在不同配置下实际运行。模型内部有固定输入尺寸，调小外部尺寸并不保证推理按比例加速。

没有测试：Heavy 的真实推理、GPU 硬件、其它 CPU/操作系统、摄像头长时运行、真实跌倒事件召回与误报率。GPU 回退的自动化测试是控制流验证，不冒充硬件实测。

## 新机器怎么用

界面点“运行设备与性能”，或执行：

```powershell
python app.py --device auto --model-variant lite --max-side 640 --opencv-threads 2
python verify_model.py your_full_body_photo.jpg --device cpu --model-variant full --report runs/local-smoke.json
```

上述只是可调整的起点，不是固定硬件推荐。默认仍为 auto/Lite/最长边 960/OpenCV 库默认线程。OpenCV 线程参数不控制 MediaPipe 推理线程。当前官方 Python GPU 支持限 Ubuntu；本项目在 Windows/macOS 的 auto 路径用 CPU，显式 GPU 则报错，不承诺核显可加速。依赖能否安装取决于操作系统与处理器架构。



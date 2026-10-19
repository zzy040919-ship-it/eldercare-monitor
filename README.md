# 基于人体姿态与时序分析的居家养老异常行为监测系统


**这是辅助提醒原型，不是医疗器械，不可用作唯一的看护手段。未触发事件不等于人员安全。**

## 已实现

| 模块 | 实现范围 |
| --- | --- |
| 疑似跌倒 | MediaPipe 关键点、平滑、躯干角度、髋部下降量/速度、持续时间、事件锁与恢复条件 |
| 视频分析 | 本地视频 / 用户主动开启的摄像头；自动/指定运行设备；后台分析线程与有界预览队列 |
| 证据留存 | 事件截图、提醒前后目标各 3 秒的片段、逐帧状态、会话参数和运行摘要 |
| 人工复核 | 待确认 / 已确认 / 误报、备注、SQLite 审计记录、状态筛选、数量统计、JSON 导出 |
| 区域进入提醒 | 鼠标框选矩形区域；髋部中心由外进入且停留超过防抖时间；开始就在区域内不提醒 |
| 持续低活动 | 仅在持续可靠检测到单人时计时；默认 120 秒；姿态缺失立即清空连续计时 |
| 参数校准 | 调整低位髋部阈值、姿态持续时间、低活动时长；场景参数可保存/载入 |

没有实现：身份识别、多人跟踪、离床/返回、自动拨打电话、短信/微信、云端看护服务、自动学习人工反馈。标记误报用于复核与后续分析，不会自动训练模型。

## 快速启动（Windows）

建议 Python **3.11，64 位**。在本目录打开 PowerShell：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

脚本只对这个进程使用执行策略参数，不修改系统策略；首次会创建 `.venv` 并安装依赖。当前交付工作区会复用已验证的 `work/.venv-care`，解压到其他位置则独立安装。也可以手动操作：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

首次真实识别会从 Google 官方地址下载模型（需要联网）；合成演示不需要模型。依赖安装也需要联网。下载失败时可重试，或提前执行：

```powershell
.\.venv\Scripts\python.exe monitor.py --download-model
```

压缩包不含 Python 环境和模型权重。当前工作区已经完成依赖安装和模型下载。程序不会自动开启摄像头，不会自动上传视频或事件；首次模型下载会连接官方服务器。

## 先看演示，再测真实视频

1. 点击“合成演示（非视频识别）”，再点击“开始分析”。
2. 约 20 秒的关键点序列依次产生疑似跌倒、区域进入、持续低活动事件；演示低活动阈值为 4 秒，真实输入恢复默认 120 秒。
3. 选中事件，查看截图或回放片段；填写备注，操作“确认事件”或“标记误报”。
4. 点击“选择视频”，加载经授权的单人全身视频。开始前校准场景，必要时框选区域，再分析。
5. 取消“按源时间预览”可尽快分析文件；只减少界面等待，不跳过分析帧，也不改变源时间计时。

附带的演示位于 `evidence/demo`。查看预置记录：

```powershell
.\.venv\Scripts\python.exe app.py --data evidence/demo
```

默认新分析存入 `data`，不会混入附带演示记录。事件列表标注“输入类型”，合成演示的统计不可作为真实检测效果。回放调用系统默认播放器；JSON 导出不包含媒体二进制文件，迁移时应复制整个数据目录。媒体路径相对数据库目录保存。

## 场景与参数

- 固定相机、单人、尽可能完整显示肩/髋/膝/踝。先用侧向或斜侧向画面做实验。
- `floor_y` 实际是**低位髋部的画面 Y 阈值**，不是地面标定或真实高度。Y 从画面顶部 0 到底部 1；默认 0.65 必须按拍摄位置调整。
- `hip_drop=0.12` 是画面高度比例；`down_speed=0.22` 是画面高度比例/秒，并非米/秒。`horizontal_angle=60` 是躯干偏离垂直的角度阈值。
- 检测流程：近期直立 → 快速下降且近水平低位 → 连续保持 1.2 秒 → 疑似事件；之后需要持续直立恢复和冷却期才能重新提醒。
- `NORMAL` 只表示未触发疑似跌倒，**不是“健康/安全”或精确动作分类**。不输出医学上的跌倒诊断，也不输出可靠的“正在坐下/弯腰”动作类别。
- `UNKNOWN` 表示姿态不可用（未检出、检测到多人、关键点低置信度或出画），立即重置连续计时。帧间隔大于 `max_gap_seconds=0.5` 也重置；摄像头处理过慢时可能难以形成连续事件。
- 摄像头源默认编号 0，仅在用户选择并开始后启用；首次可观察画面后停止，再设置区域和重新开始。
- 片段以提醒时间为中心，目标前后各 3 秒，10 FPS 按源时间重采样。接近视频末尾或提前停止时，后段不足会标记 `post_truncated`，不是伪造完整 6 秒。
- 所有高级参数见 `config.json`，界面可以载入完整配置。参数不应在测试集上反复调到“全对”。

## 运行设备与不同电脑

界面点击“运行设备与性能”。运行配置和场景参数分开，保存参数 JSON 时一并导出 `runtime`；载入时恢复。命令行参数优先于配置文件，未指定时默认 `auto / lite / 960 / OpenCV 库默认线程`。

```powershell
# 保守的 CPU 设置示例；不要求某款 CPU，不保证所有电脑都实时
python app.py --device cpu --model-variant lite --max-side 640 --opencv-threads 2
# 更大的模型用于对比，不能预先保证效果更好；首次下载对应模型
python monitor.py your_video.mp4 --device auto --model-variant full --max-side 1280
# 仅在支持 GPU delegate 的平台与环境下使用；否则明确报错
python monitor.py your_video.mp4 --device gpu
```

- `auto`：当前 Windows/macOS 路径使用 CPU；在 Linux 上尝试 GPU delegate，初始化失败则回退 CPU并记录原因。官方当前仅承诺 Ubuntu 的 Python GPU 支持，不宣称其它 Linux 环境已经验证。
- `cpu`：明确使用 CPU，没有品牌/型号判断。底层包仍受操作系统和处理器架构限制；没有覆盖所有机器测试。
- `gpu`：强制 GPU；不支持的平台或初始化失败明确报错，不偷偷当作 GPU 成功。
- `lite/full/heavy`：使用官方模型包；模型越大不一定能改善本任务，需要分别实测。
- `--max-side 256～3840`：降低尺寸可减少部分解码后处理、绘图和缓存开销，但模型内部输入尺寸固定，不能承诺推理按比例变快，也可能损失关键点质量。不会为了提速跳过分析帧。
- `--opencv-threads 0～64`：仅控制 OpenCV；0 保留库默认。**MediaPipe Tasks 的此处 Python API 没有提供模型推理线程参数，不能把这个选项称为 CPU 推理线程。**

摘要和界面显示 `effective_device`，报告记录请求设置、实际设备、回退原因和主机信息。合成关键点标记 `synthetic_no_model`，不冒充 CPU/GPU 推理。新机器应自行运行测试，不能套用另一台电脑的 FPS。

依据：[MediaPipe BaseOptions 的 delegate 平台限制](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/BaseOptions)、[官方模型与输入尺寸说明](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)。本次没有真实 GPU 硬件验证，也不保证 Windows 核显可以直接加速。

## 命令行与验证

以下命令中的 Python 应替换为实际虚拟环境路径；当前工作区也可用 `..\..\work\.venv-care\Scripts\python.exe`。

```powershell
# 合成演示：必须带演示参数，低活动才会在 20 秒内出现
python monitor.py examples/showcase.json --config examples/showcase_config.json --output runs/demo --annotated
# 真实视频：使用默认或自己导出的场景参数
python monitor.py "D:/videos/authorized-test.mp4" --config config.json --output runs/video --annotated
# 离线测试：不会访问摄像头或下载模型
python -m unittest -v test_system test_integration test_runtime
# 已有模型下的真实推理冒烟：用全身静态图片重复 90 帧
python verify_model.py "D:/images/authorized-full-body.jpg" --report runs/model-smoke.json
```

需要完整依赖快照时，可用 `requirements-lock.txt` 替换 `requirements.txt`。`--annotated` 生成完整标注视频，适用于文件演示；摄像头全程录制采用近似固定帧率，不作为精确时间证据，事件片段按实际时间重采样。

## 已验证与待验证

完整证据见 [验证报告](VALIDATION.md)。

- 12 类人工构造关键点序列验证状态机，不验证姿态模型准确率。
- 端到端覆盖事件片段、截断标记、停止收尾、错误会话记录、区域框选、复核与筛选。
- 已实际调用 CPU 姿态模型处理 90 帧静态图片重复视频；这只能证明模型管线可运行。
- **还没有独立真实跌倒视频测试集，所以不报告真实事件召回率、每小时误报或真实提醒延迟，也不承诺实时性能。**

## 明确局限

遮挡时发生的跌倒、视频开始时已经倒地、缓慢滑落、朝镜头方向跌倒、床上快速躺下，均可能漏检或误报。相机移动、透视变化、光照差、宽松衣物会影响关键点。虽然请求最多检测两个人以识别部分多人情况，但模型可能漏检第二个人；也没有跨帧身份跟踪，不能用于人来人往的场景。长时间低活动不等于失去意识，区域进入不等于危险。

只拍摄已知情同意的成年人；不要让老人或任何人真的摔倒来收集数据。优先使用许可明确的公开视频和安全动作对照。图片、片段、日志默认保留在本机，没有登录鉴权、加密、自动过期清理；不要公开包含个人影像的 `data`。截图和片段包含可识别影像。

## 代码导航

- `logic.py`：不依赖界面的姿态特征、时序规则、状态机。
- `monitor.py`：MediaPipe 接入、视频时间、可视化、截图与环形片段缓存。
- `storage.py`：SQLite 会话、事件、复核审计与导出。
- `app.py`：PySide6 界面、后台线程、框选区域与人工确认。
- `examples.py` / `examples/`：可复现的合成关键点；生成器采用独占创建，已有样例不会被覆盖。
- `test_system.py` / `test_integration.py`：规则和系统检查。
- `test_runtime.py`：设备选择/回退、运行配置、损坏缓存与异常姿态检查；GPU 分支用替身测试。
- `verify_model.py`：独立的真实姿态模型冒烟脚本。

数据流：视频解码 → 姿态模型 → 关键点质量检查 → 平滑与时序规则 → 事件锁/区域计时 → SQLite 与证据片段 → 人工复核。

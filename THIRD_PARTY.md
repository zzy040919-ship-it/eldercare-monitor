# 第三方模型与资料

本项目自己实现的是规则、状态机、视频处理、界面和记录流程。人体关键点由已有 MediaPipe 模型提供。

- [MediaPipe Pose Landmarker Python 官方说明](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)：使用 VIDEO 模式与递增毫秒时间戳。API 支持返回人体关键点；本项目并不训练该模型。
- [Pose Landmarker 模型说明](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)：默认 Lite，可选 Full/Heavy，设备运行参数可配置。
- [BaseOptions 官方 API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/BaseOptions)：GPU/CPU delegate 与 GPU 平台限制。本项目不将可选 GPU 等同于所有平台均支持。
- [模型下载源](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task)：首次真实识别自动下载到 `models/`。已测试 Lite 文件 SHA-256：`59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`。下载先写唯一临时文件，检查 bundle 内容与 CRC 后再替换目标；已有损坏文件明确报错，不擅自删除。每个模型的 `*.source.json` 记录实际文件摘要，该摘要不是签名验证。初版 Lite 下载记录仍保留为 `source.json`。
- [官方 Python 示例 notebook](https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb) 引用的 [Pixabay 原图页面](https://pixabay.com/photos/girl-woman-fitness-beautiful-smile-4051811/) / [图片地址](https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg)：开发时仅用于模型冒烟，图片与其重复视频不随项目包分发，未作为跌倒样本。图片权利与 notebook 代码许可是两回事，使用原图应核对其自身条款。

项目包不重新分发模型权重和第三方字体。Windows 预览使用本机安装的微软雅黑，字体文件没有复制到项目。软件依赖遵守各自许可；若以后分发可执行安装包，需另外核对 Qt/PySide、OpenCV、MediaPipe 及所有依赖和模型的分发条款。当前交付为源码和合成数据。

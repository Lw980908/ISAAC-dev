# isaac_ros_rtsp_server

基于 live555 的 ROS 2 RTSP 推流节点，订阅 H264/H265 压缩码流话题，
不做二次编码，尽可能降低 CPU 占用。

## 依赖

需要 live555 已编译（动态库或静态库均可）：

```bash
cd /home/alienware/Sensor/Cuda_Project/ISAAC/src/live555
./genMakefiles linux
make -j
```

若你已安装到 `/usr/local`，也可直接使用系统库：

```bash
sudo make install
```

## 编译

```bash
cd /home/alienware/Sensor/Cuda_Project/ISAAC
colcon build --packages-up-to isaac_ros_rtsp_server
```

## 运行（多路）

默认读取 `config/rtsp_server_multi.yaml`：

```bash
ros2 launch isaac_ros_rtsp_server isaac_ros_rtsp_server_multi.launch.py
```

启动后控制台会输出 RTSP URL，例如：

```
rtsp://127.0.0.1:8554/left
```

## 与 ISAAC H264 编码对接

`isaac_ros_h264_encoder` 的 `image_compressed` 会发布 `sensor_msgs/CompressedImage`
（格式为 `h264`），本包直接订阅此话题并推流。

建议在同一 `ComposableNodeContainer` 内运行编码与 RTSP 节点，并开启
`use_intra_process_comms` 降低拷贝开销。

## 协商话题（NITROS）

如需订阅 NITROS 协商话题，可在 YAML 中设置：

- `use_nitros: true`
- `nitros_format: "nitros_compressed_image"`

RTSP 节点会通过 NITROS 协商订阅 `nitros_compressed_image`，并在节点内
转换为 `sensor_msgs/CompressedImage` 后推流。

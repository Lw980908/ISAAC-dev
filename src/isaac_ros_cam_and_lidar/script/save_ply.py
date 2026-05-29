#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import argparse
import math
import numbers
import struct
from rclpy.qos import QoSProfile, ReliabilityPolicy


def unpack_rgb(value):
    """Decode PointCloud2 PCL-style packed RGB into red/green/blue bytes."""
    if isinstance(value, numbers.Integral):
        packed = int(value) & 0xFFFFFFFF
    else:
        packed = struct.unpack('<I', struct.pack('<f', float(value)))[0]
    return (packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF

class PLYSaver(Node):
    def __init__(self, topic, output, max_pts=500000):
        super().__init__('ply_saver')
        self.output = output
        self.max_pts = max_pts
        self.pts = []
        self.done = False
        
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.sub = self.create_subscription(PointCloud2, topic, self.cb, qos)
        self.get_logger().info(f" 监听: {topic} |  输出: {output}")

    def cb(self, msg):
        if self.done: return
        field_names = [field.name for field in msg.fields]
        read_fields = ("x", "y", "z", "rgb") if "rgb" in field_names else ("x", "y", "z")
        try:
            for p in pc2.read_points(msg, field_names=read_fields, skip_nans=True):
                if len(self.pts) >= self.max_pts: break
                x,y,z = p[0],p[1],p[2]
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)): continue
                
                if len(p) > 3:
                    r, g, b = unpack_rgb(p[3])
                    self.pts.append((x, y, z, r, g, b))
                else:
                    self.pts.append((x,y,z, 200,200,200)) # 默认灰色
        except Exception as e:
            self.get_logger().warn(f"解析异常: {e}")
            return

        if len(self.pts) > 0:
            self._write_ply()
            self.done = True
            self.get_logger().info("✅ 保存完成，正在退出...")
            rclpy.shutdown()

    def _write_ply(self):
        has_rgb = len(self.pts[0]) == 6
        with open(self.output, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(self.pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            if has_rgb:
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p in self.pts:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
                if has_rgb: f.write(f" {p[3]} {p[4]} {p[5]}")
                f.write("\n")
        self.get_logger().info(f" 已写入 {len(self.pts)} 点到 {self.output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='/front/projection_colored_cloud')
    parser.add_argument('--output', default='cloud.ply')
    parser.add_argument('--max', type=int, default=500000)
    args = parser.parse_args()
    
    rclpy.init()
    node = PLYSaver(args.topic, args.output, args.max)
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__': main()

# # 运行（保存第一帧后自动退出）
# python3 save_ply.py --topic /front/projection_colored_cloud --output scan.ply

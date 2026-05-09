/*
 * TensorRT 版本兼容层
 *
 * TensorRT 10.x 移除了 samples/common/utils/timingCache.h
 * 此文件提供空实现以保持向后兼容
 */

#ifndef TENSORRT_COMPAT_TIMING_CACHE_H
#define TENSORRT_COMPAT_TIMING_CACHE_H

#include <NvInfer.h>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace sample {
namespace utils {

// TensorRT 10.x 兼容：提供空实现
inline std::vector<char> loadTimingCacheFile(const std::string & /*path*/) {
  return {};
}

inline void saveTimingCacheFile(const std::string & /*path*/,
                                const std::vector<char> & /*cache*/) {
  // 空实现
}

// 用于 TensorRT 8.x 兼容
inline std::vector<char> loadTimingCacheFile(std::string const & /*inFileName*/,
                                             bool /*verbose*/) {
  return {};
}

inline void saveTimingCacheFile(std::string const & /*outFileName*/,
                                void const * /*cacheData*/,
                                size_t /*cacheSize*/) {
  // 空实现
}

} // namespace utils
} // namespace sample

// TensorRT 10.x 使用 nvinfer1::utils 命名空间
namespace nvinfer1 {
namespace utils {

// 从文件加载 timing cache
inline std::vector<char>
loadTimingCacheFile(std::string const & /*inFileName*/) {
  return {};
}

// 从文件构建 timing cache
inline nvinfer1::ITimingCache *
buildTimingCacheFromFile(nvinfer1::IBuilder &builder,
                         std::vector<char> const & /*timingCacheFile*/,
                         std::ostream & /*err*/) {
  // 返回空的 timing cache
  auto config = builder.createBuilderConfig();
  if (config) {
    return config->createTimingCache(nullptr, 0);
  }
  return nullptr;
}

// 保存 timing cache 到文件
inline void
saveTimingCacheFile(std::string const & /*outFileName*/,
                    nvinfer1::ITimingCache const * /*timingCache*/) {
  // 空实现
}

// 更新 timing cache 文件
inline void
updateTimingCacheFile(std::string const & /*fileName*/,
                      nvinfer1::ITimingCache const * /*timingCache*/,
                      std::ostream & /*err*/) {
  // 空实现
}

} // namespace utils
} // namespace nvinfer1

#endif // TENSORRT_COMPAT_TIMING_CACHE_H

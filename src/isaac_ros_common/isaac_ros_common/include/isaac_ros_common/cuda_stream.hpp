#ifndef ISAAC_ROS_COMMON__CUDA_STREAM_HPP_
#define ISAAC_ROS_COMMON__CUDA_STREAM_HPP_

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define CHECK_CUDA_ERROR(cuda_call, error_message) \
  do { \
    const cudaError_t _err = (cuda_call); \
    if (_err != cudaSuccess) { \
      throw std::runtime_error(std::string(error_message) + ": " + cudaGetErrorString(_err)); \
    } \
  } while (0)

namespace nvidia
{
namespace isaac_ros
{
namespace common
{

inline cudaError_t initNamedCudaStream(cudaStream_t & stream, const char * name)
{
  (void)name;
  return cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
}

inline cudaError_t initNamedCudaStream(cudaStream_t & stream, const std::string & name)
{
  return initNamedCudaStream(stream, name.c_str());
}

}  // namespace common
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_COMMON__CUDA_STREAM_HPP_

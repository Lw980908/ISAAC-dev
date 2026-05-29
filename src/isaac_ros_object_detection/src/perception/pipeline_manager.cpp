#include "pipeline_manager.h"
#include "multi_threads_multi_sources.h"
#include "shared_model_thread_multi_sources.h"
#include "single_thread_single_source.h"
#include <logger.h>

std::unique_ptr<PipelineInterface>
createPipelineManager(InputConfig input_config) {
  if (input_config.process_method ==
      ProcessMethod::MULTI_THREADS_MULTI_SOURCES) // 对多源进行多线程处理
  {
    sample::gLogInfo << "Creating MultiThreadsMultiSourcesPipelineManager..."
                     << std::endl;
    return std::make_unique<MultiThreadsMultiSourcesPipelineManager>(
        input_config);
  } else if (input_config.process_method ==
             ProcessMethod::SINGLE_THREAD_SINGLE_SOURCE) // 对单源进行单线程处理
  {
    sample::gLogInfo << "Creating SingleThreadPipelineManager..." << std::endl;
    return std::make_unique<SingleThreadSingleSourcePipelineManager>(
        input_config);
  } else if (input_config.process_method ==
             ProcessMethod::
                 SHARED_MODEL_THREAD_MULTI_SOURCES) // 对多源进行共享模型处理
  {
    sample::gLogInfo << "Creating SharedModelMultiSourcesPipelineManager..."
                     << std::endl;
    return std::make_unique<SharedModelMultiSourcesPipelineManager>(
        input_config);
  }

  return nullptr;
}

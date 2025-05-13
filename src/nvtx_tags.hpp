#pragma once
#ifdef USE_NVTX
  /*  header con le API C di NVTX  */
  #ifdef __cplusplus
  extern "C" {
  #endif
  #include <nvToolsExt.h>
  #ifdef __cplusplus
  }
  #endif

  #define NVTX_PUSH(msg) nvtxRangePushA(msg)
  #define NVTX_POP()     nvtxRangePop()
#else
  #define NVTX_PUSH(msg)
  #define NVTX_POP()
#endif
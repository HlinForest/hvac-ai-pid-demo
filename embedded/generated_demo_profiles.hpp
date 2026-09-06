#pragma once

#include <cstdint>

namespace hvac_mcu { namespace demo {

enum class AlgorithmId : uint8_t {
  ZN,
  IMC,
  BO,
  SAFE_BO,
  FNN,
  RL,
  LLM
};

struct ControllerProfile {
  const char *name;
  float kp;
  float ki;
  bool accepted;
  bool fallback_required;
};

static constexpr ControllerProfile kProfiles[7] = {
  {"zn", 0.517486269f, 0.00405362595f, false, true},
  {"imc", 0.517486269f, 0.00405362595f, true, false},
  {"bo", 0.517486269f, 0.00405362595f, false, true},
  {"safe-bo", 0.383783106f, 0.00444109808f, true, false},
  {"fnn", 0.505799113f, 0.00486435113f, true, false},
  {"rl", 0.517486269f, 0.00405362595f, true, false},
  {"llm", 0.476708351f, 0.00342936755f, true, false}
};

static constexpr uint32_t kProfileVersion = 1u;
static constexpr const char kProfileManifest[] = "zn:0.517486269:0.00405362595:0:1;imc:0.517486269:0.00405362595:1:0;bo:0.517486269:0.00405362595:0:1;safe-bo:0.383783106:0.00444109808:1:0;fnn:0.505799113:0.00486435113:1:0;rl:0.517486269:0.00405362595:1:0;llm:0.476708351:0.00342936755:1:0";
static constexpr uint32_t kProfileCrc32 = 0x557B3206u;
static constexpr uint32_t kPidPeriodMs = 100u;
static constexpr uint32_t kAdaptivePeriodMs = 2000u;
static constexpr uint32_t kTelemetryPeriodMs = 500u;

} }  // namespace hvac_mcu::demo

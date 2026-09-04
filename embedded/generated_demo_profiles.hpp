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
  {"zn", 0.450823722f, 0.00307651849f, false, true},
  {"imc", 0.450823722f, 0.00307651849f, true, false},
  {"bo", 0.588236022f, 0.00527096547f, true, false},
  {"safe-bo", 0.383783106f, 0.00444109808f, true, false},
  {"fnn", 0.450823722f, 0.00307651849f, false, true},
  {"rl", 0.586070838f, 0.00230738886f, true, false},
  {"llm", 0.415298813f, 0.00260273464f, true, false}
};

static constexpr uint32_t kProfileVersion = 1u;
static constexpr const char kProfileManifest[] = "zn:0.450823722:0.00307651849:0:1;imc:0.450823722:0.00307651849:1:0;bo:0.588236022:0.00527096547:1:0;safe-bo:0.383783106:0.00444109808:1:0;fnn:0.450823722:0.00307651849:0:1;rl:0.586070838:0.00230738886:1:0;llm:0.415298813:0.00260273464:1:0";
static constexpr uint32_t kProfileCrc32 = 0x52C8D4E8u;
static constexpr uint32_t kPidPeriodMs = 100u;
static constexpr uint32_t kAdaptivePeriodMs = 2000u;
static constexpr uint32_t kTelemetryPeriodMs = 500u;

} }  // namespace hvac_mcu::demo

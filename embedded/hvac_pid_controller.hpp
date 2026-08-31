#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "generated_policy.hpp"

namespace hvac_mcu {

template <typename T>
inline T clamp_value(T value, T lower, T upper) {
  return value < lower ? lower : (value > upper ? upper : value);
}

struct Gains { float kp; float ki; };
struct Diagnostics { Gains gains; float output; bool fallback_active; bool output_limited; };
static constexpr Gains kFactoryFallbackGains{0.45f, 0.003f};

inline uint32_t manifest_crc_byte(uint32_t crc, uint8_t value) {
  crc ^= value;
  for (uint8_t bit=0; bit<8; ++bit)
    crc=(crc&1u)?(crc>>1u)^0xEDB88320u:crc>>1u;
  return crc;
}

inline uint32_t manifest_crc_u32(uint32_t crc, uint32_t value) {
  for (int shift=0; shift<32; shift+=8) crc=manifest_crc_byte(crc,static_cast<uint8_t>(value>>shift));
  return crc;
}

inline uint32_t manifest_crc_float(uint32_t crc, float value) {
  uint32_t bits=0;
  std::memcpy(&bits,&value,sizeof(bits));
  return manifest_crc_u32(crc,bits);
}

inline uint32_t policy_manifest_crc32() {
  uint32_t crc=0xFFFFFFFFu;
  const char magic[]="HVACPID3";
  for (int i=0;i<8;++i) crc=manifest_crc_byte(crc,static_cast<uint8_t>(magic[i]));
  crc=manifest_crc_u32(crc,generated::kArtifactVersion);
  crc=manifest_crc_byte(crc,generated::kFnnAccepted?1u:0u);
  crc=manifest_crc_byte(crc,generated::kRlAccepted?1u:0u);
  const float scalars[]={generated::kFallbackKp,generated::kFallbackKi,
    generated::kMinimumKp,generated::kMaximumKp,generated::kMinimumKi,
    generated::kMaximumKi,generated::kMaximumGainChangeFraction};
  for (float value:scalars) crc=manifest_crc_float(crc,value);
  for (float value:generated::kFnnErrorCenters) crc=manifest_crc_float(crc,value);
  for (float value:generated::kFnnErrorRateCenters) crc=manifest_crc_float(crc,value);
  for (int e=0;e<5;++e) for (int d=0;d<5;++d) for (int g=0;g<2;++g)
    crc=manifest_crc_float(crc,generated::kFnnRuleTable[e][d][g]);
  for (int g=0;g<2;++g) for (int feature=0;feature<4;++feature)
    crc=manifest_crc_float(crc,generated::kFnnContextCoefficients[g][feature]);
  for (float value:generated::kRlErrorEdges) crc=manifest_crc_float(crc,value);
  for (float value:generated::kRlErrorRateEdges) crc=manifest_crc_float(crc,value);
  for (float value:generated::kRlCommandEdges) crc=manifest_crc_float(crc,value);
  for (int a=0;a<9;++a) for (int g=0;g<2;++g)
    crc=manifest_crc_float(crc,generated::kRlTargetScales[a][g]);
  for (int e=0;e<5;++e) for (int d=0;d<5;++d) for (int c=0;c<3;++c)
    crc=manifest_crc_byte(crc,generated::kRlPolicy[e][d][c]);
  for (int e=0;e<5;++e) for (int d=0;d<5;++d) for (int c=0;c<3;++c)
    crc=manifest_crc_byte(crc,generated::kRlCovered[e][d][c]);
  return crc^0xFFFFFFFFu;
}

inline bool policy_manifest_valid() {
  if (generated::kArtifactVersion!=3u || policy_manifest_crc32()!=generated::kArtifactCrc32) return false;
  if (!std::isfinite(generated::kFallbackKp) || !std::isfinite(generated::kFallbackKi)) return false;
  if (generated::kFallbackKp<generated::kMinimumKp || generated::kFallbackKp>generated::kMaximumKp ||
      generated::kFallbackKi<generated::kMinimumKi || generated::kFallbackKi>generated::kMaximumKi) return false;
  for (int e=0;e<5;++e) for (int d=0;d<5;++d) for (int c=0;c<3;++c)
    if (generated::kRlPolicy[e][d][c]>=9u || generated::kRlCovered[e][d][c]>1u) return false;
  return true;
}

class SafePI {
 public:
  explicit SafePI(Gains fallback) : gains_(fallback), fallback_(fallback) {}

  void configure_fallback(Gains fallback) {
    fallback_ = fallback;
    reset();
  }

  void reset() {
    gains_ = fallback_; integral_ = 0.0f; output_ = 0.0f;
    fallback_active_ = false; output_limited_ = false;
  }

  float update(float error, float dt_seconds) {
    if (!std::isfinite(error) || !std::isfinite(dt_seconds) || dt_seconds <= 0.0f) {
      gains_ = fallback_; fallback_active_ = true; output_ = 0.0f; return output_;
    }
    const float dt_minutes = dt_seconds / 60.0f;
    const float candidate = integral_ + error * dt_minutes;
    const float raw = gains_.kp * error + gains_.ki * candidate;
    const float saturated = clamp_value(raw, 0.0f, 1.0f);
    if ((raw >= 0.0f && raw <= 1.0f) || (raw > 1.0f && error < 0.0f) ||
        (raw < 0.0f && error > 0.0f)) integral_ = candidate;
    output_limited_ = std::fabs(saturated - raw) > 1e-6f;
    output_ = saturated;
    return output_;
  }

  bool apply_proposal(Gains proposed, bool policy_valid = true) {
    if (!policy_valid || !std::isfinite(proposed.kp) || !std::isfinite(proposed.ki)) {
      gains_ = fallback_; fallback_active_ = true; return false;
    }
    const float fraction=generated::kMaximumGainChangeFraction;
    proposed.kp = clamp_value(proposed.kp, gains_.kp * (1.0f-fraction), gains_.kp * (1.0f+fraction));
    proposed.ki = clamp_value(proposed.ki, gains_.ki * (1.0f-fraction), gains_.ki * (1.0f+fraction));
    gains_.kp = clamp_value(proposed.kp, generated::kMinimumKp, generated::kMaximumKp);
    gains_.ki = clamp_value(proposed.ki, generated::kMinimumKi, generated::kMaximumKi);
    fallback_active_ = false;
    return true;
  }

  void force_fallback() {
    gains_ = fallback_;
    fallback_active_ = true;
  }

  Gains gains() const { return gains_; }
  float integral_state() const { return integral_; }
  Diagnostics diagnostics() const { return {gains_, output_, fallback_active_, output_limited_}; }

 private:
  Gains gains_;
  Gains fallback_;
  float integral_ = 0.0f;
  float output_ = 0.0f;
  bool fallback_active_ = false;
  bool output_limited_ = false;
};

class CompressorLimiter {
 public:
  CompressorLimiter(float minimum_running=0.25f, float slew_per_minute=0.05f,
                    float quantum=0.01f, float minimum_on_seconds=300.0f,
                    float minimum_off_seconds=180.0f)
      : minimum_running_(minimum_running), slew_per_second_(slew_per_minute/60.0f),
        quantum_(quantum), minimum_on_seconds_(minimum_on_seconds),
        minimum_off_seconds_(minimum_off_seconds) {}

  void reset() {
    continuous_=0.0f; output_=0.0f; on_=false; seconds_in_state_=1.0e9f;
  }

  float update(float requested, float dt_seconds) {
    requested=clamp_value(requested,0.0f,1.0f);
    bool wants_on=requested>=0.5f*minimum_running_;
    if (on_ && !wants_on) {
      if (seconds_in_state_>=minimum_on_seconds_) {
        on_=false; continuous_=0.0f; output_=0.0f; seconds_in_state_=0.0f;
      } else wants_on=true;
    } else if (!on_ && wants_on) {
      if (seconds_in_state_>=minimum_off_seconds_) {
        on_=true; continuous_=minimum_running_; output_=minimum_running_; seconds_in_state_=0.0f;
      } else wants_on=false;
    }
    if (on_ && wants_on) {
      const float target=clamp_value(requested<minimum_running_?minimum_running_:requested,minimum_running_,1.0f);
      const float step=slew_per_second_*dt_seconds;
      continuous_=clamp_value(target,continuous_-step,continuous_+step);
      output_=quantum_>0.0f?std::round(continuous_/quantum_)*quantum_:continuous_;
      output_=clamp_value(output_,minimum_running_,1.0f);
    }
    seconds_in_state_+=dt_seconds;
    return output_;
  }

  float command() const { return output_; }
  bool is_on() const { return on_; }

 private:
  float minimum_running_,slew_per_second_,quantum_,minimum_on_seconds_,minimum_off_seconds_;
  float continuous_=0.0f,output_=0.0f,seconds_in_state_=1.0e9f;
  bool on_=false;
};

inline void active_interval(float value, const float centers[5], int &low, int &high, float &weight) {
  value = clamp_value(value, centers[0], centers[4]);
  high = 1;
  while (high < 4 && value >= centers[high]) ++high;
  low = high - 1;
  weight = clamp_value((value - centers[low]) / (centers[high] - centers[low]), 0.0f, 1.0f);
}

inline Gains fnn_gains(float error, float delta_error, float applied_command=0.0f,
                       float integral_state=0.0f, float outdoor_delta_c=0.0f,
                       float load_fraction=0.0f) {
  int e0,e1,d0,d1; float ew,dw;
  active_interval(error,generated::kFnnErrorCenters,e0,e1,ew);
  active_interval(delta_error,generated::kFnnErrorRateCenters,d0,d1,dw);
  Gains result{0.0f,0.0f};
  const int es[2]={e0,e1}, ds[2]={d0,d1};
  const float ews[2]={1.0f-ew,ew}, dws[2]={1.0f-dw,dw};
  for (int ei=0;ei<2;++ei) for (int di=0;di<2;++di) {
    const float w=ews[ei]*dws[di];
    result.kp += w*generated::kFnnRuleTable[es[ei]][ds[di]][0];
    result.ki += w*generated::kFnnRuleTable[es[ei]][ds[di]][1];
  }
  const float features[4]={
    clamp_value(applied_command-0.5f,-2.0f,2.0f),
    clamp_value(integral_state/100.0f,-2.0f,2.0f),
    clamp_value(outdoor_delta_c/20.0f,-2.0f,2.0f),
    clamp_value(load_fraction-0.25f,-2.0f,2.0f)};
  float residual[2]={0.0f,0.0f};
  for (int g=0;g<2;++g) for (int feature=0;feature<4;++feature)
    residual[g]+=generated::kFnnContextCoefficients[g][feature]*features[feature];
  result.kp*=std::exp(clamp_value(residual[0],-0.12f,0.12f));
  result.ki*=std::exp(clamp_value(residual[1],-0.12f,0.12f));
  return result;
}

inline int edge_bin(float value, const float *edges, int edge_count) {
  int result=0;
  while (result<edge_count && value>=edges[result]) ++result;
  return result;
}

inline Gains rl_gains(float error, float error_rate, float applied_command,
                      Gains fallback, bool &covered) {
  const int e=edge_bin(error,generated::kRlErrorEdges,4);
  const int d=edge_bin(error_rate,generated::kRlErrorRateEdges,4);
  const int c=edge_bin(applied_command,generated::kRlCommandEdges,2);
  covered=policy_manifest_valid() && generated::kRlAccepted && generated::kRlCovered[e][d][c]!=0;
  const uint8_t action=covered?generated::kRlPolicy[e][d][c]:4;
  return {fallback.kp*generated::kRlTargetScales[action][0],
          fallback.ki*generated::kRlTargetScales[action][1]};
}

// Accelerated plant stub: 100 ms wall time = 1/3 simulated minute, so the
// five-hour demonstration lasts 90 seconds.  This is not a real-room claim.
class VirtualHVACPlant {
 public:
  void reset(float initial_temperature=30.0f) {
    temperature_=initial_temperature; actuator_=0.0f; index_=0;
    for (int i=0;i<kDelaySlots;++i) delay_[i]=0.0f;
  }
  float step(float command,bool door_open,float dt_sim_minutes=0.333333333f) {
    const float delayed=delay_[index_]; delay_[index_]=clamp_value(command,0.0f,1.0f);
    index_=(index_+1)%kDelaySlots;
    actuator_ += dt_sim_minutes*(delayed-actuator_)/6.0f;
    const float equilibrium=32.0f+(door_open?3.0f:0.0f)-14.0f*actuator_;
    temperature_ += dt_sim_minutes*(equilibrium-temperature_)/35.0f;
    return temperature_;
  }
  float temperature() const { return temperature_; }
  float actuator() const { return actuator_; }
 private:
  static const int kDelaySlots=14;
  float delay_[kDelaySlots]{};
  int index_=0;
  float temperature_=30.0f;
  float actuator_=0.0f;
};

}  // namespace hvac_mcu

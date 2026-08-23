#pragma once

#include <cmath>
#include <cstdint>

namespace hvac_mcu {

template <typename T>
inline T clamp_value(T value, T lower, T upper) {
  return value < lower ? lower : (value > upper ? upper : value);
}

struct Gains { float kp; float ki; };
struct Diagnostics { Gains gains; float output; bool fallback_active; bool output_limited; };

class SafePI {
 public:
  explicit SafePI(Gains fallback, float output_slew_per_second = 0.20f)
      : gains_(fallback), fallback_(fallback), output_slew_per_second_(output_slew_per_second) {}

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
    const float max_step = output_slew_per_second_ * dt_seconds;
    const float slewed = clamp_value(saturated, output_ - max_step, output_ + max_step);
    output_limited_ = std::fabs(slewed - raw) > 1e-6f;
    output_ = clamp_value(slewed, 0.0f, 1.0f);
    return output_;
  }

  bool apply_proposal(Gains proposed, bool policy_valid = true) {
    if (!policy_valid || !std::isfinite(proposed.kp) || !std::isfinite(proposed.ki)) {
      gains_ = fallback_; fallback_active_ = true; return false;
    }
    proposed.kp = clamp_value(proposed.kp, gains_.kp * 0.75f, gains_.kp * 1.25f);
    proposed.ki = clamp_value(proposed.ki, gains_.ki * 0.75f, gains_.ki * 1.25f);
    gains_.kp = clamp_value(proposed.kp, 0.002f, 1.5f);
    gains_.ki = clamp_value(proposed.ki, 1e-5f, 0.08f);
    fallback_active_ = false;
    return true;
  }

  Gains gains() const { return gains_; }
  Diagnostics diagnostics() const { return {gains_, output_, fallback_active_, output_limited_}; }

 private:
  Gains gains_;
  Gains fallback_;
  float integral_ = 0.0f;
  float output_ = 0.0f;
  float output_slew_per_second_ = 0.20f;
  bool fallback_active_ = false;
  bool output_limited_ = false;
};

// Exported from outputs/fnn_rule_table.npy. Only four neighbouring rules are read.
static const float kFnnRuleTable[5][5][2] = {
    {{0.051355902f,0.000056127f},{0.051355902f,0.000056127f},{0.051355902f,0.000056127f},{0.051355902f,0.000056127f},{0.051355902f,0.000056127f}},
    {{0.051355902f,0.000056127f},{0.051355902f,0.000056127f},{0.051355902f,0.000056127f},{0.051355902f,0.000056127f},{0.051355902f,0.000056127f}},
    {{0.051355902f,0.000056127f},{0.213310494f,0.001018775f},{0.606015206f,0.005166625f},{0.051355902f,0.000056127f},{0.051355902f,0.000056127f}},
    {{0.051355902f,0.000056127f},{0.213310494f,0.001018775f},{0.638474978f,0.008540036f},{0.051355902f,0.000056127f},{0.051355902f,0.000056127f}},
    {{0.051355902f,0.000056127f},{0.051355902f,0.000056127f},{0.621039394f,0.007309802f},{0.051355902f,0.000056127f},{0.051355902f,0.000056127f}}};

inline void active_interval(float value, const float centers[5], int &low, int &high, float &weight) {
  value = clamp_value(value, centers[0], centers[4]);
  high = 1;
  while (high < 4 && value >= centers[high]) ++high;
  low = high - 1;
  weight = clamp_value((value - centers[low]) / (centers[high] - centers[low]), 0.0f, 1.0f);
}

inline Gains fnn_gains(float error, float delta_error) {
  static const float e_centers[5] = {-5.0f,-2.5f,0.0f,2.5f,5.0f};
  static const float d_centers[5] = {-1.0f,-0.5f,0.0f,0.5f,1.0f};
  int e0,e1,d0,d1; float ew,dw;
  active_interval(error,e_centers,e0,e1,ew); active_interval(delta_error,d_centers,d0,d1,dw);
  Gains result{0.0f,0.0f};
  const int es[2]={e0,e1}, ds[2]={d0,d1};
  const float ews[2]={1.0f-ew,ew}, dws[2]={1.0f-dw,dw};
  for (int ei=0;ei<2;++ei) for (int di=0;di<2;++di) {
    const float w=ews[ei]*dws[di];
    result.kp += w*kFnnRuleTable[es[ei]][ds[di]][0];
    result.ki += w*kFnnRuleTable[es[ei]][ds[di]][1];
  }
  return result;
}

// Greedy RL export from the current quick run. All 25 coarse states were visited;
// this is coverage evidence, not proof that the noisy policy has converged.
static const uint8_t kRlPolicy[5][5] = {{8,1,0,0,3},{4,7,4,1,6},{4,6,2,5,1},{2,1,8,6,7},{5,1,1,6,7}};
static const uint8_t kRlCovered[5][5] = {{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}};
static const float kRlActions[9][2] = {{-0.10f,-0.10f},{-0.10f,0.10f},{0.0f,-0.10f},{0.0f,0.0f},{0.0f,0.10f},{0.10f,-0.10f},{0.10f,0.0f},{0.10f,0.10f},{0.20f,0.0f}};

inline int rl_bin(float value, float scale) {
  return clamp_value(static_cast<int>(std::floor(value/scale))+2,0,4);
}

inline Gains rl_gains(float error, float delta_error, Gains current, bool &covered) {
  const int e=rl_bin(error,2.5f), d=rl_bin(delta_error,0.5f);
  covered=kRlCovered[e][d]!=0;
  const uint8_t action=covered?kRlPolicy[e][d]:3;
  return {current.kp*(1.0f+kRlActions[action][0]),current.ki*(1.0f+kRlActions[action][1])};
}

// Accelerated plant stub: 100 ms real time = 0.6 simulated minutes.
class VirtualHVACPlant {
 public:
  void reset(float initial_temperature=30.0f) {
    temperature_=initial_temperature; actuator_=0.0f; index_=0;
    for (int i=0;i<kDelaySlots;++i) delay_[i]=0.0f;
  }
  float step(float command,bool door_open,float dt_sim_minutes=0.6f) {
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

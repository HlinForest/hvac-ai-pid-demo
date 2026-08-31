#include "hvac_pid_controller.hpp"
#include "generated_demo_profiles.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <limits>

int main() {
  using namespace hvac_mcu;
  constexpr float kPidDtSeconds=0.1f;
  constexpr int kAiDivider=20;
  const bool manifest_ok=policy_manifest_valid();
  const Gains fallback=manifest_ok?Gains{generated::kFallbackKp,generated::kFallbackKi}:kFactoryFallbackGains;
  SafePI controller(fallback);
  CompressorLimiter limiter;
  VirtualHVACPlant plant;
  float previous_ai_error=0.0f;
  float command=0.0f;
  long long worst_pi_ns=0,worst_ai_ns=0;
  int ai_calls=0,fallback_events=0;
  bool bounded=true;
  for (int tick=0;tick<4000;++tick) {
    const float setpoint=tick<180?24.0f:22.5f;
    const bool door_open=tick>=240&&tick<280;
    const float error=plant.temperature()-setpoint;
    if (tick%kAiDivider==0) {
      const float error_rate=ai_calls==0?0.0f:(error-previous_ai_error)/(kAiDivider*kPidDtSeconds/60.0f);
      const auto begin=std::chrono::steady_clock::now();
      if (tick<200) controller.apply_proposal(fnn_gains(error,error_rate),manifest_ok&&generated::kFnnAccepted);
      else {
        bool covered=false;
        const Gains proposed=rl_gains(error,error_rate,limiter.command(),fallback,covered);
        controller.apply_proposal(proposed,covered);
        if (!covered) ++fallback_events;
      }
      const auto end=std::chrono::steady_clock::now();
      worst_ai_ns=std::max(worst_ai_ns,std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
      previous_ai_error=error;
      ++ai_calls;
    }
    const auto begin=std::chrono::steady_clock::now();
    const float requested=controller.update(error,kPidDtSeconds);
    command=limiter.update(requested,kPidDtSeconds);
    const auto end=std::chrono::steady_clock::now();
    worst_pi_ns=std::max(worst_pi_ns,std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
    bounded=bounded&&command>=0.0f&&command<=1.0f;
    plant.step(command,door_open);
  }
  controller.apply_proposal({std::numeric_limits<float>::quiet_NaN(),0.01f});
  const Diagnostics d=controller.diagnostics();
  std::cout<<std::setprecision(9);
  const float parity_commands[3]={0.0f,0.35f,0.90f};
  for (float e:generated::kFnnErrorCenters) for (float de:generated::kFnnErrorRateCenters)
    for (float u:parity_commands) {
      const float integral=e*5.0f, outdoor_delta=10.0f, load_fraction=0.20f;
      const Gains fg=fnn_gains(e,de,u,integral,outdoor_delta,load_fraction);
      bool covered=false;
      const Gains rg=rl_gains(e,de,u,fallback,covered);
      std::cout<<"PARITY,"<<e<<","<<de<<","<<u<<","<<integral<<","<<outdoor_delta<<","<<load_fraction
               <<","<<fg.kp<<","<<fg.ki<<","<<(covered?1:0)<<","<<rg.kp<<","<<rg.ki<<"\n";
    }
  int valid_profiles=0, stable_profiles=0;
  constexpr float kAcceleratedPhysicalDtSeconds=20.0f;  // 100 ms wall clock at 200x
  for (const auto &profile : demo::kProfiles) {
    const bool flags_valid=profile.accepted!=profile.fallback_required;
    const bool gains_valid=profile.kp>=0.002f&&profile.kp<=1.5f&&profile.ki>=1e-5f&&profile.ki<=0.08f;
    if (flags_valid&&gains_valid) ++valid_profiles;
    SafePI profile_controller({profile.kp,profile.ki});
    CompressorLimiter profile_limiter;
    VirtualHVACPlant profile_plant;
    for (int profile_tick=0;profile_tick<900;++profile_tick) {
      const float profile_error=profile_plant.temperature()-24.0f;
      const float request=profile_controller.update(profile_error,kAcceleratedPhysicalDtSeconds);
      const float limited=profile_limiter.update(request,kAcceleratedPhysicalDtSeconds);
      profile_plant.step(limited,profile_tick>=567&&profile_tick<612);
    }
    if (std::fabs(profile_plant.temperature()-24.0f)<=0.75f) ++stable_profiles;
  }
  const bool passed=manifest_ok&&bounded&&ai_calls==200&&d.fallback_active&&valid_profiles==7&&stable_profiles==7;
  std::cout<<"MCU_SIL "<<(passed?"PASS":"FAIL")<<"\n"
           <<"manifest v3="<<manifest_ok<<"; CRC32="<<std::hex<<generated::kArtifactCrc32
           <<"; computed="<<policy_manifest_crc32()<<std::dec<<"\n"
           <<"PID period=100ms; AI period=2s; AI calls="<<ai_calls<<"\n"
           <<"sizeof(SafePI)="<<sizeof(SafePI)<<" bytes; sizeof(VirtualHVACPlant)="<<sizeof(VirtualHVACPlant)<<" bytes\n"
           <<"worst PI="<<worst_pi_ns<<" ns; worst AI="<<worst_ai_ns<<" ns\n"
           <<"seven profiles valid="<<valid_profiles<<"; stable after 90s demo="<<stable_profiles<<"\n"
           <<"final temperature="<<plant.temperature()<<" C; u="<<command<<"; Kp="<<d.gains.kp<<"; Ki="<<d.gains.ki<<"\n"
           <<"RL uncovered-state fallbacks="<<fallback_events<<"; NaN fallback="<<d.fallback_active<<"\n";
  return passed?0:1;
}

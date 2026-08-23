#include "hvac_pid_controller.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>

int main() {
  using namespace hvac_mcu;
  constexpr float kPidDtSeconds=0.1f;
  constexpr int kAiDivider=20;
  SafePI controller({0.12f,0.004f});
  VirtualHVACPlant plant;
  float previous_error=plant.temperature()-24.0f;
  float command=0.0f;
  long long worst_pi_ns=0,worst_ai_ns=0;
  int ai_calls=0,fallback_events=0;
  bool bounded=true;
  for (int tick=0;tick<400;++tick) {
    const float setpoint=tick<180?24.0f:22.5f;
    const bool door_open=tick>=240&&tick<280;
    const float error=plant.temperature()-setpoint;
    const float delta_error=error-previous_error;
    if (tick%kAiDivider==0) {
      const auto begin=std::chrono::steady_clock::now();
      if (tick<200) controller.apply_proposal(fnn_gains(error,delta_error));
      else {
        bool covered=false;
        const Gains proposed=rl_gains(error,delta_error,controller.gains(),covered);
        controller.apply_proposal(proposed,covered);
        if (!covered) ++fallback_events;
      }
      const auto end=std::chrono::steady_clock::now();
      worst_ai_ns=std::max(worst_ai_ns,std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
      ++ai_calls;
    }
    const auto begin=std::chrono::steady_clock::now();
    command=controller.update(error,kPidDtSeconds);
    const auto end=std::chrono::steady_clock::now();
    worst_pi_ns=std::max(worst_pi_ns,std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
    bounded=bounded&&command>=0.0f&&command<=1.0f;
    plant.step(command,door_open);
    previous_error=error;
  }
  controller.apply_proposal({std::numeric_limits<float>::quiet_NaN(),0.01f});
  const Diagnostics d=controller.diagnostics();
  const bool passed=bounded&&ai_calls==20&&d.fallback_active;
  std::cout<<"MCU_SIL "<<(passed?"PASS":"FAIL")<<"\n"
           <<"PID period=100ms; AI period=2s; AI calls="<<ai_calls<<"\n"
           <<"sizeof(SafePI)="<<sizeof(SafePI)<<" bytes; sizeof(VirtualHVACPlant)="<<sizeof(VirtualHVACPlant)<<" bytes\n"
           <<"worst PI="<<worst_pi_ns<<" ns; worst AI="<<worst_ai_ns<<" ns\n"
           <<"final temperature="<<plant.temperature()<<" C; u="<<command<<"; Kp="<<d.gains.kp<<"; Ki="<<d.gains.ki<<"\n"
           <<"RL uncovered-state fallbacks="<<fallback_events<<"; NaN fallback="<<d.fallback_active<<"\n";
  return passed?0:1;
}

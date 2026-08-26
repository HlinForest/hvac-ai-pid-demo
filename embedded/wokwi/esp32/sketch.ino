#include "hvac_pid_controller.hpp"

using namespace hvac_mcu;

#if defined(ARDUINO_ARCH_ESP32)
const int PIN_SETPOINT=34, PIN_DOOR=15, PIN_PWM=5, PIN_FALLBACK=4;
#elif defined(ARDUINO_ARCH_STM32)
const int PIN_SETPOINT=PA0, PIN_DOOR=PB11, PIN_PWM=PB0, PIN_FALLBACK=PB1;
#else
#error "Select ESP32 DevKit or STM32F103C8 Blue Pill"
#endif

// Change to true in the left editor to exercise the exported RL policy.
constexpr bool USE_RL_POLICY=false;
constexpr uint32_t PID_PERIOD_MS=100;
constexpr uint32_t AI_DIVIDER=20;  // 20 * 100 ms = 2 s

const Gains fallback_gains{generated::kFallbackKp,generated::kFallbackKi};
SafePI controller(fallback_gains);
CompressorLimiter limiter;
VirtualHVACPlant plant;
uint32_t next_pid_ms=0, tick_count=0;
float previous_ai_error=0.0f;
uint32_t previous_ai_ms=0;
bool ai_state_initialized=false;
uint32_t worst_pi_us=0, worst_ai_us=0;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  pinMode(PIN_DOOR,INPUT_PULLUP);
  pinMode(PIN_PWM,OUTPUT);
  pinMode(PIN_FALLBACK,OUTPUT);
  controller.reset();
  limiter.reset();
  plant.reset(30.0f);
  next_pid_ms=millis();
}

void control_tick() {
  const int adc=analogRead(PIN_SETPOINT);
  const float setpoint=22.0f+4.0f*static_cast<float>(adc)/4095.0f;
  const bool door_open=digitalRead(PIN_DOOR)==LOW;
  const float error=plant.temperature()-setpoint;

  if (tick_count%AI_DIVIDER==0) {
    const uint32_t now_ms=millis();
    const float elapsed_minutes=ai_state_initialized?static_cast<float>(now_ms-previous_ai_ms)/60000.0f:0.0f;
    const float error_rate=ai_state_initialized?(error-previous_ai_error)/fmaxf(elapsed_minutes,1.0e-6f):0.0f;
    const uint32_t started=micros();
    if (USE_RL_POLICY) {
      bool covered=false;
      const Gains proposed=rl_gains(error,error_rate,limiter.command(),fallback_gains,covered);
      controller.apply_proposal(proposed,covered);
    } else {
      controller.apply_proposal(fnn_gains(error,error_rate),generated::kFnnAccepted);
    }
    const uint32_t elapsed=micros()-started;
    if (elapsed>worst_ai_us) worst_ai_us=elapsed;
    previous_ai_error=error;
    previous_ai_ms=now_ms;
    ai_state_initialized=true;
  }

  const uint32_t started=micros();
  const float requested=controller.update(error,PID_PERIOD_MS/1000.0f);
  const float command=limiter.update(requested,PID_PERIOD_MS/1000.0f);
  const uint32_t elapsed=micros()-started;
  if (elapsed>worst_pi_us) worst_pi_us=elapsed;
  plant.step(command,door_open);
  analogWrite(PIN_PWM,static_cast<int>(255.0f*command));
  const Diagnostics d=controller.diagnostics();
  digitalWrite(PIN_FALLBACK,d.fallback_active?HIGH:LOW);

  if (tick_count%5==0) {
    Serial.print("temp:"); Serial.print(plant.temperature(),3);
    Serial.print(" setpoint:"); Serial.print(setpoint,3);
    Serial.print(" pwm_pct:"); Serial.print(command*100.0f,2);
    Serial.print(" kp_x100:"); Serial.print(d.gains.kp*100.0f,3);
    Serial.print(" ki_x1000:"); Serial.print(d.gains.ki*1000.0f,3);
    Serial.print(" pi_us:"); Serial.print(worst_pi_us);
    Serial.print(" ai_us:"); Serial.print(worst_ai_us);
    Serial.print(" fallback:"); Serial.println(d.fallback_active?1:0);
  }
  ++tick_count;
}

void loop() {
  const uint32_t now=millis();
  if (static_cast<int32_t>(now-next_pid_ms)>=0) {
    next_pid_ms+=PID_PERIOD_MS;
    control_tick();
  }
}

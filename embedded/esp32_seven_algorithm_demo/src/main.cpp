#include <Arduino.h>

#include "generated_demo_profiles.hpp"
#include "hvac_pid_controller.hpp"
#include "modbus_config.hpp"
#include "modbus_rtu_adapter.hpp"

using namespace hvac_mcu;

namespace {

constexpr int kSetpointPin = 34;
constexpr int kDoorPin = 15;
constexpr int kPwmPin = 5;
constexpr int kFallbackPin = 4;
constexpr int kPwmChannel = 0;
constexpr float kDefaultSetpoint = 24.0f;
constexpr float kDemoPhysicalDtSeconds = 20.0f;  // 100 ms wall clock at 200×

enum class PlantMode : uint8_t { DEMO, MODBUS };

PlantMode mode = PlantMode::DEMO;
uint8_t algorithm_index = 1;  // IMC
float serial_setpoint = kDefaultSetpoint;
bool use_knob = true;
bool manual_door = false;
bool fault = false;
bool profile_crc_valid = false;
uint32_t next_pid_ms = 0, next_telemetry_ms = 0, next_modbus_ms = 0;
uint32_t tick_count = 0, worst_pi_us = 0, worst_ai_us = 0, missed_periods = 0;
uint32_t last_sensor_ms = 0;
float measured_temperature = 30.0f, actual_frequency_hz = 0.0f;
float previous_ai_error = 0.0f;

SafePI controller({demo::kProfiles[1].kp, demo::kProfiles[1].ki});
CompressorLimiter limiter;
VirtualHVACPlant plant;
HardwareSerial modbus_serial(2);
ModbusRTUAdapter modbus(modbus_serial);

uint32_t crc32(const char *text) {
  uint32_t crc = 0xFFFFFFFFu;
  while (*text) {
    crc ^= static_cast<uint8_t>(*text++);
    for (uint8_t bit = 0; bit < 8; ++bit)
      crc = (crc & 1u) ? (crc >> 1u) ^ 0xEDB88320u : crc >> 1u;
  }
  return crc ^ 0xFFFFFFFFu;
}

const demo::ControllerProfile &profile() { return demo::kProfiles[algorithm_index]; }

void activate_profile(uint8_t index) {
  if (index >= 7) return;
  algorithm_index = index;
  const auto &selected = profile();
  controller.configure_fallback({selected.kp, selected.ki});
  if (selected.fallback_required || !selected.accepted) controller.force_fallback();
  limiter.reset();
}

int find_profile(const String &name) {
  for (int i = 0; i < 7; ++i) if (name.equalsIgnoreCase(demo::kProfiles[i].name)) return i;
  return -1;
}

void process_command(String line) {
  line.trim();
  const int separator = line.indexOf(' ');
  const String command = separator < 0 ? line : line.substring(0, separator);
  String argument = separator < 0 ? "" : line.substring(separator + 1);
  argument.trim();
  if (command.equalsIgnoreCase("ALGO")) {
    const int index = find_profile(argument);
    if (index >= 0) activate_profile(static_cast<uint8_t>(index));
    else Serial.println("{\"error\":\"unknown algorithm\"}");
  } else if (command.equalsIgnoreCase("MODE")) {
    if (argument.equalsIgnoreCase("DEMO")) { mode = PlantMode::DEMO; fault = false; plant.reset(30.0f); }
    else if (argument.equalsIgnoreCase("MODBUS")) mode = PlantMode::MODBUS;
  } else if (command.equalsIgnoreCase("SETPOINT")) {
    serial_setpoint = constrain(argument.toFloat(), 18.0f, 30.0f);
    use_knob = false;
  } else if (command.equalsIgnoreCase("KNOB")) {
    use_knob = argument != "0";
  } else if (command.equalsIgnoreCase("DISTURB")) {
    manual_door = argument != "0";
  } else if (command.equalsIgnoreCase("RESET")) {
    plant.reset(30.0f); limiter.reset(); activate_profile(algorithm_index); fault = false;
  } else if (!command.equalsIgnoreCase("STATUS")) {
    Serial.println("{\"error\":\"commands: ALGO/MODE/SETPOINT/KNOB/DISTURB/RESET/STATUS\"}");
  }
}

void read_modbus_inputs() {
  uint16_t raw_temperature = 0, raw_frequency = 0, raw_alarm = 0;
  const bool ok_temperature = modbus.read_holding(modbus_config::kTemperatureRegister, raw_temperature);
  const bool ok_frequency = modbus.read_holding(modbus_config::kActualFrequencyRegister, raw_frequency);
  const bool ok_alarm = modbus.read_holding(modbus_config::kAlarmRegister, raw_alarm);
  if (ok_temperature && ok_frequency && ok_alarm && raw_alarm == 0) {
    measured_temperature = raw_temperature * modbus_config::kTemperatureScale;
    actual_frequency_hz = raw_frequency * modbus_config::kFrequencyScale;
    last_sensor_ms = millis();
    fault = !isfinite(measured_temperature) || measured_temperature < -20.0f || measured_temperature > 80.0f;
  } else {
    fault = true;
  }
}

void schedule_adaptive(float error) {
  const uint32_t started = micros();
  const float error_rate = (error - previous_ai_error) / (demo::kAdaptivePeriodMs / 60000.0f);
  if (algorithm_index == static_cast<uint8_t>(demo::AlgorithmId::FNN)) {
    controller.apply_proposal(fnn_gains(error, error_rate), generated::kFnnAccepted && profile().accepted);
  } else if (algorithm_index == static_cast<uint8_t>(demo::AlgorithmId::RL)) {
    bool covered = false;
    const Gains proposed = rl_gains(error, error_rate, limiter.command(),
                                    {profile().kp, profile().ki}, covered);
    controller.apply_proposal(proposed, covered && profile().accepted);
  }
  if (!profile().accepted || profile().fallback_required) controller.force_fallback();
  previous_ai_error = error;
  const uint32_t elapsed_us = static_cast<uint32_t>(micros() - started);
  if (elapsed_us > worst_ai_us) worst_ai_us = elapsed_us;
}

void control_tick() {
  const float setpoint = use_knob ? 22.0f + 4.0f * analogRead(kSetpointPin) / 4095.0f : serial_setpoint;
  const bool automatic_door = mode == PlantMode::DEMO && millis() > 56700u && millis() < 61200u;
  const bool door_open = manual_door || automatic_door || digitalRead(kDoorPin) == LOW;
  if (mode == PlantMode::DEMO) measured_temperature = plant.temperature();
  const float error = measured_temperature - setpoint;

  if (tick_count % (demo::kAdaptivePeriodMs / demo::kPidPeriodMs) == 0) schedule_adaptive(error);
  const uint32_t started = micros();
  const float physical_dt_seconds = mode == PlantMode::DEMO
                                        ? kDemoPhysicalDtSeconds
                                        : demo::kPidPeriodMs / 1000.0f;
  float requested = fault || !profile_crc_valid ? controller.update(NAN, physical_dt_seconds)
                                                : controller.update(error, physical_dt_seconds);
  float command = limiter.update(requested, physical_dt_seconds);
  if (fault || !profile_crc_valid) command = 0.0f;
  const uint32_t elapsed_us = static_cast<uint32_t>(micros() - started);
  if (elapsed_us > worst_pi_us) worst_pi_us = elapsed_us;

  if (mode == PlantMode::DEMO) {
    measured_temperature = plant.step(command, door_open);
  } else if (modbus_config::kWritesEnabled) {
    const uint16_t raw = static_cast<uint16_t>(constrain(command * 100.0f * modbus_config::kCapacityWriteScale, 0.0f, 65535.0f));
    if (!modbus.write_single(modbus_config::kCapacityRequestRegister, raw)) fault = true;
  }
  ledcWrite(kPwmChannel, static_cast<uint32_t>(255.0f * command));
  digitalWrite(kFallbackPin, controller.diagnostics().fallback_active || fault ? HIGH : LOW);
  ++tick_count;
}

void send_telemetry() {
  const Diagnostics d = controller.diagnostics();
  const float setpoint = use_knob ? 22.0f + 4.0f * analogRead(kSetpointPin) / 4095.0f : serial_setpoint;
  Serial.printf(
      "{\"temperature_c\":%.3f,\"setpoint_c\":%.3f,\"capacity_pct\":%.2f,"
      "\"kp\":%.7f,\"ki\":%.7f,\"algorithm\":\"%s\",\"mode\":\"%s\","
      "\"accepted\":%s,\"fallback\":%s,\"fault\":%s,\"frequency_hz\":%.2f,"
      "\"pi_wcet_us\":%lu,\"ai_wcet_us\":%lu,\"missed_periods\":%lu}\n",
      measured_temperature, setpoint, limiter.command() * 100.0f, d.gains.kp, d.gains.ki,
      profile().name, mode == PlantMode::DEMO ? "demo" : "modbus",
      profile().accepted ? "true" : "false",
      (d.fallback_active || profile().fallback_required) ? "true" : "false",
      fault ? "true" : "false", actual_frequency_hz,
      static_cast<unsigned long>(worst_pi_us), static_cast<unsigned long>(worst_ai_us),
      static_cast<unsigned long>(missed_periods));
}

}  // namespace

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  pinMode(kDoorPin, INPUT_PULLUP);
  pinMode(kFallbackPin, OUTPUT);
  ledcSetup(kPwmChannel, 1000, 8);
  ledcAttachPin(kPwmPin, kPwmChannel);
  modbus.begin();
  profile_crc_valid = crc32(demo::kProfileManifest) == demo::kProfileCrc32;
  activate_profile(algorithm_index);
  plant.reset(30.0f);
  next_pid_ms = next_telemetry_ms = next_modbus_ms = millis();
}

void loop() {
  while (Serial.available()) process_command(Serial.readStringUntil('\n'));
  const uint32_t now = millis();
  if (mode == PlantMode::MODBUS && static_cast<int32_t>(now - next_modbus_ms) >= 0) {
    next_modbus_ms += demo::kTelemetryPeriodMs;
    read_modbus_inputs();
    if (millis() - last_sensor_ms > modbus_config::kSensorStaleTimeoutMs) fault = true;
  }
  if (static_cast<int32_t>(now - next_pid_ms) >= 0) {
    if (now - next_pid_ms >= demo::kPidPeriodMs) ++missed_periods;
    next_pid_ms += demo::kPidPeriodMs;
    control_tick();
  }
  if (static_cast<int32_t>(now - next_telemetry_ms) >= 0) {
    next_telemetry_ms += demo::kTelemetryPeriodMs;
    send_telemetry();
  }
}

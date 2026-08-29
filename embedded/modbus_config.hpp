#pragma once

#include <cstdint>

namespace hvac_mcu { namespace modbus_config {

// These values are placeholders.  Replace them only from the air-conditioner
// or BMS vendor's signed register map.  Writes remain disabled by default.
static constexpr bool kWritesEnabled = false;
static constexpr uint32_t kBaud = 9600u;
static constexpr uint8_t kSlaveAddress = 1u;
static constexpr int kRxPin = 16;
static constexpr int kTxPin = 17;
static constexpr int kDriverEnablePin = 21;

static constexpr uint16_t kTemperatureRegister = 0x0000u;
static constexpr uint16_t kActualFrequencyRegister = 0x0001u;
static constexpr uint16_t kAlarmRegister = 0x0002u;
static constexpr uint16_t kCapacityRequestRegister = 0x0100u;

static constexpr float kTemperatureScale = 0.1f;  // raw 241 -> 24.1 °C
static constexpr float kFrequencyScale = 0.1f;    // raw 503 -> 50.3 Hz
static constexpr float kCapacityWriteScale = 10.0f;  // 75.0% -> raw 750
static constexpr bool kSwapRegisterBytes = false;
static constexpr uint32_t kResponseTimeoutMs = 250u;
static constexpr uint32_t kSensorStaleTimeoutMs = 3000u;

} }  // namespace hvac_mcu::modbus_config

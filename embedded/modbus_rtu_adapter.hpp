#pragma once

#if !defined(ARDUINO_ARCH_ESP32)
#error "ModbusRTUAdapter currently targets ESP32 HardwareSerial."
#endif

#include <Arduino.h>

#include "modbus_config.hpp"

namespace hvac_mcu {

inline uint16_t modbus_crc16(const uint8_t *data, size_t length) {
  uint16_t crc = 0xFFFFu;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (uint8_t bit = 0; bit < 8; ++bit)
      crc = (crc & 1u) ? static_cast<uint16_t>((crc >> 1u) ^ 0xA001u) : static_cast<uint16_t>(crc >> 1u);
  }
  return crc;
}

class ModbusRTUAdapter {
 public:
  explicit ModbusRTUAdapter(HardwareSerial &serial) : serial_(serial) {}

  void begin() {
    pinMode(modbus_config::kDriverEnablePin, OUTPUT);
    set_transmit(false);
    serial_.begin(
        modbus_config::kBaud, SERIAL_8N1,
        modbus_config::kRxPin, modbus_config::kTxPin);
  }

  bool read_holding(uint16_t address, uint16_t &value) {
    uint8_t request[8] = {
      modbus_config::kSlaveAddress, 0x03,
      static_cast<uint8_t>(address >> 8), static_cast<uint8_t>(address),
      0x00, 0x01, 0x00, 0x00
    };
    append_crc(request, 6);
    uint8_t response[7]{};
    if (!transaction(request, sizeof(request), response, sizeof(response))) return false;
    if (response[0] != modbus_config::kSlaveAddress || response[1] != 0x03 || response[2] != 0x02) return false;
    value = static_cast<uint16_t>((response[3] << 8) | response[4]);
    if (modbus_config::kSwapRegisterBytes) value = static_cast<uint16_t>((value >> 8) | (value << 8));
    return true;
  }

  bool write_single(uint16_t address, uint16_t value) {
    if (!modbus_config::kWritesEnabled) return false;
    if (modbus_config::kSwapRegisterBytes) value = static_cast<uint16_t>((value >> 8) | (value << 8));
    uint8_t request[8] = {
      modbus_config::kSlaveAddress, 0x06,
      static_cast<uint8_t>(address >> 8), static_cast<uint8_t>(address),
      static_cast<uint8_t>(value >> 8), static_cast<uint8_t>(value), 0x00, 0x00
    };
    append_crc(request, 6);
    uint8_t response[8]{};
    return transaction(request, sizeof(request), response, sizeof(response));
  }

 private:
  void set_transmit(bool enabled) {
    digitalWrite(modbus_config::kDriverEnablePin, enabled ? HIGH : LOW);
  }

  static void append_crc(uint8_t *frame, size_t payload_length) {
    const uint16_t crc = modbus_crc16(frame, payload_length);
    frame[payload_length] = static_cast<uint8_t>(crc);
    frame[payload_length + 1] = static_cast<uint8_t>(crc >> 8);
  }

  bool transaction(const uint8_t *request, size_t request_length, uint8_t *response, size_t response_length) {
    while (serial_.available()) serial_.read();
    set_transmit(true);
    delayMicroseconds(100);
    serial_.write(request, request_length);
    serial_.flush();
    delayMicroseconds(100);
    set_transmit(false);

    const uint32_t started = millis();
    size_t received = 0;
    while (received < response_length && millis() - started < modbus_config::kResponseTimeoutMs) {
      if (serial_.available()) response[received++] = static_cast<uint8_t>(serial_.read());
      else delay(1);
    }
    if (received != response_length) return false;
    const uint16_t received_crc = static_cast<uint16_t>(response[response_length - 2]) |
                                  static_cast<uint16_t>(response[response_length - 1] << 8);
    return received_crc == modbus_crc16(response, response_length - 2);
  }

  HardwareSerial &serial_;
};

}  // namespace hvac_mcu

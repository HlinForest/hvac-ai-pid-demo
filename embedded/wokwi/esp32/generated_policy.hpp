#pragma once

#include <cstdint>

namespace hvac_mcu { namespace generated {

static constexpr uint32_t kArtifactVersion = 3u;
static constexpr uint32_t kArtifactCrc32 = 0xFC5BB753u;
static constexpr bool kFnnAccepted = true;
static constexpr bool kRlAccepted = true;
static constexpr float kFallbackKp = 0.517486274f;
static constexpr float kFallbackKi = 0.00405362574f;
static constexpr float kMinimumKp = 0.002f;
static constexpr float kMaximumKp = 1.5f;
static constexpr float kMinimumKi = 0.00001f;
static constexpr float kMaximumKi = 0.08f;
static constexpr float kMaximumGainChangeFraction = 0.10f;
static const float kFnnErrorCenters[5] = {-3.0f,-0.75f,0.0f,1.5f,5.0f};
static const float kFnnErrorRateCenters[5] = {-0.300000012f,-0.0500000007f,0.0f,0.0500000007f,0.300000012f};
static const float kFnnRuleTable[5][5][2] = {{{0.491611958f,0.00486435136f},{0.491611958f,0.00486435136f},{0.491611958f,0.00486435136f},{0.491611958f,0.00486435136f},{0.491611958f,0.00486435136f}},
    {{0.491611958f,0.00486435136f},{0.491611958f,0.00486435136f},{0.491611958f,0.00486435136f},{0.491611958f,0.00486435136f},{0.491611958f,0.00486435136f}},
    {{0.517486274f,0.00486435136f},{0.517486274f,0.00486435136f},{0.517486274f,0.00486435136f},{0.517486274f,0.00486435136f},{0.517486274f,0.00486435136f}},
    {{0.527835965f,0.00486435136f},{0.527835965f,0.00486435136f},{0.527835965f,0.00486435136f},{0.527835965f,0.00486435136f},{0.527835965f,0.00486435136f}},
    {{0.527835965f,0.00486435136f},{0.527835965f,0.00486435136f},{0.527835965f,0.00486435136f},{0.527835965f,0.00486435136f},{0.527835965f,0.00486435136f}}};
static const float kFnnContextCoefficients[2][4] = {{-0.0f,-0.0f,0.0f,0.0f},{-0.0f,-0.0f,0.0f,0.0f}};
static const float kRlErrorEdges[4] = {-2.0f,-0.5f,0.5f,2.0f};
static const float kRlErrorRateEdges[4] = {-0.150000006f,-0.0299999993f,0.0299999993f,0.150000006f};
static const float kRlCommandEdges[2] = {0.0500000007f,0.699999988f};
static const float kRlTargetScales[9][2] = {{0.75f,0.75f},{0.75f,1.0f},{0.75f,1.29999995f},{1.0f,0.75f},{1.0f,1.0f},{1.0f,1.29999995f},{1.29999995f,0.75f},{1.29999995f,1.0f},{1.29999995f,1.29999995f}};
static const uint8_t kRlPolicy[5][5][3] = {{{4,4,4},{4,4,4},{4,4,4},{4,4,4},{4,4,4}},
    {{4,4,4},{4,4,4},{4,4,4},{4,4,4},{4,4,4}},
    {{4,4,4},{4,4,4},{4,4,4},{4,4,4},{4,4,4}},
    {{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2}},
    {{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2}}};
static const uint8_t kRlCovered[5][5][3] = {{{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0}},
    {{1,1,0},{1,1,0},{1,0,0},{1,0,0},{1,0,0}},
    {{1,1,1},{1,1,1},{1,1,1},{1,1,0},{1,1,0}},
    {{0,1,1},{0,1,1},{1,1,1},{0,1,1},{1,1,1}},
    {{0,0,1},{0,0,1},{1,1,1},{0,1,1},{0,1,1}}};

} }  // namespace hvac_mcu::generated

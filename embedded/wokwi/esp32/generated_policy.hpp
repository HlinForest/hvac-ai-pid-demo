#pragma once

#include <cstdint>

namespace hvac_mcu { namespace generated {

static constexpr uint32_t kArtifactVersion = 2u;
static constexpr uint32_t kArtifactCrc32 = 0x76E995D6u;
static constexpr bool kFnnAccepted = false;
static constexpr bool kRlAccepted = true;
static constexpr float kFallbackKp = 0.450823722f;
static constexpr float kFallbackKi = 0.00307651849f;
static const float kFnnErrorCenters[5] = {-3.0f,-0.75f,0.0f,1.5f,5.0f};
static const float kFnnErrorRateCenters[5] = {-0.3f,-0.05f,0.0f,0.05f,0.3f};
static const float kFnnRuleTable[5][5][2] = {{{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f}},
    {{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f}},
    {{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f}},
    {{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f}},
    {{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f},{0.450823722f,0.00307651849f}}};
static const float kRlErrorEdges[4] = {-2.0f,-0.5f,0.5f,2.0f};
static const float kRlErrorRateEdges[4] = {-0.15f,-0.03f,0.03f,0.15f};
static const float kRlCommandEdges[2] = {0.05f,0.7f};
static const float kRlTargetScales[9][2] = {{0.75f,0.75f},{0.75f,1.0f},{0.75f,1.3f},{1.0f,0.75f},{1.0f,1.0f},{1.0f,1.3f},{1.3f,0.75f},{1.3f,1.0f},{1.3f,1.3f}};
static const uint8_t kRlPolicy[5][5][3] = {{{3,4,4},{1,4,4},{1,4,4},{4,4,4},{0,4,4}},
    {{3,4,4},{4,0,4},{0,0,4},{1,4,4},{4,4,4}},
    {{1,0,1},{6,6,4},{6,6,3},{7,6,4},{1,0,4}},
    {{4,4,8},{4,3,8},{2,5,2},{4,0,6},{1,6,4}},
    {{4,4,1},{4,4,0},{6,3,5},{4,2,8},{4,1,2}}};
static const uint8_t kRlCovered[5][5][3] = {{{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0}},
    {{1,1,0},{1,1,0},{1,1,0},{1,0,0},{1,0,0}},
    {{1,1,1},{1,1,1},{1,1,1},{1,1,0},{1,1,0}},
    {{0,1,1},{0,1,1},{1,1,1},{0,1,1},{1,1,1}},
    {{0,0,1},{0,0,1},{1,1,1},{0,1,1},{0,1,1}}};

} }  // namespace hvac_mcu::generated

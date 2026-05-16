#pragma once

#include <cstdint>

static constexpr uint32_t PTO_HCCL_MAX_RANKS = 64;

struct HcclDeviceContext {
    uint64_t workSpace = 0;
    uint64_t workSpaceSize = 0;
    uint32_t rankId = 0;
    uint32_t rankNum = 0;
    uint64_t winSize = 0;
    uint64_t windowsIn[PTO_HCCL_MAX_RANKS] = {};
    uint64_t windowsOut[PTO_HCCL_MAX_RANKS] = {};
};

namespace pto_hccl_compat {

constexpr uint32_t MAX_CC_TILING_NUM = 8U;
constexpr uint32_t GROUP_NAME_SIZE = 128U;
constexpr uint32_t ALG_CONFIG_SIZE = 128U;
constexpr uint32_t LOCAL_NOTIFY_MAX_NUM = 64U;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 19U;
constexpr uint32_t AICPU_OP_NOTIFY_MAX_NUM = 2U;

struct CommResourceInitV2 {
    uint32_t version = 0;
    uint32_t hcommCount = 0;
    uint32_t offset[MAX_CC_TILING_NUM] = {};
    uint8_t debugMode = 0;
    uint8_t preparePosition = 0;
    uint16_t queueNum = 0;
    uint16_t commBlockNum = 0;
    uint8_t devType = 0;
    char reserved[17] = {};
};

struct CommResourceConfigV2 {
    uint8_t skipLocalRankCopy = 0;
    uint8_t skipBufferWindowCopy = 0;
    uint8_t stepSize = 0;
    uint8_t version = 0;
    char reserved[9] = {};
    uint8_t commEngine = 0;
    uint8_t srcDataType = 0;
    uint8_t dstDataType = 0;
    char groupName[GROUP_NAME_SIZE] = {};
    char algConfig[ALG_CONFIG_SIZE] = {};
    uint32_t opType = 0;
    uint32_t reduceType = 0;
};

struct CommResourceTilingV2 {
    CommResourceInitV2 init{};
    CommResourceConfigV2 inner{};
};

struct HcclSignalInfo {
    uint64_t resId = 0;
    uint64_t addr = 0;
    uint32_t devId = 0;
    uint32_t tsId = 0;
    uint32_t rankId = 0;
    uint32_t flag = 0;
};

struct HcclStreamInfo {
    int32_t streamIds = 0;
    uint32_t sqIds = 0;
    uint32_t cqIds = 0;
    uint32_t logicCqids = 0;
};

struct ListCommon {
    uint64_t nextHost = 0;
    uint64_t preHost = 0;
    uint64_t nextDevice = 0;
    uint64_t preDevice = 0;
};

struct LocalResInfoV2 {
    uint32_t streamNum = 0;
    uint32_t signalNum = 0;
    HcclSignalInfo localSignals[LOCAL_NOTIFY_MAX_NUM] = {};
    HcclStreamInfo streamInfo[LOCAL_STREAM_MAX_NUM] = {};
    HcclStreamInfo mainStreamInfo{};
    HcclSignalInfo aicpuOpNotify[AICPU_OP_NOTIFY_MAX_NUM] = {};
    ListCommon nextTagRes{};
};

struct AlgoTopoInfo {
    uint32_t userRank = 0;
    uint32_t userRankSize = 0;
    int32_t deviceLogicId = 0;
    bool isSingleMeshAggregation = false;
    uint32_t deviceNumPerAggregation = 0;
    uint32_t superPodNum = 0;
    uint32_t devicePhyId = 0;
    uint32_t topoType = 0;
    uint32_t deviceType = 0;
    uint32_t serverNum = 0;
    uint32_t meshAggregationRankSize = 0;
    uint32_t multiModuleDiffDeviceNumMode = 0;
    uint32_t multiSuperPodDiffServerNumMode = 0;
    uint32_t realUserRank = 0;
    bool isDiffDeviceModule = false;
    bool isDiffDeviceType = false;
    uint32_t gcdDeviceNumPerAggregation = 0;
    uint32_t moduleNum = 0;
    uint32_t isUsedRdmaRankPairNum = 0;
    uint64_t isUsedRdmaRankPair = 0;
    uint32_t pairLinkCounterNum = 0;
    uint64_t pairLinkCounter = 0;
    uint32_t nicNum = 0;
    uint64_t nicList = 0;
    uint64_t complanRankLength = 0;
    uint64_t complanRank = 0;
    uint64_t bridgeRankNum = 0;
    uint64_t bridgeRank = 0;
    uint64_t serverAndsuperPodRankLength = 0;
    uint64_t serverAndsuperPodRank = 0;
};

struct HcclOpConfig {
    uint8_t deterministic = 0;
    uint8_t retryEnable = 0;
    uint8_t highPerfEnable = 0;
    uint8_t padding[5] = {};
    uint8_t linkTimeOut[8] = {};
    uint64_t notifyWaitTime = 0;
    uint32_t retryHoldTime = 0;
    uint32_t retryIntervalTime = 0;
    bool interXLinkDisable = false;
    uint32_t floatOverflowMode = 0;
    uint32_t multiQpThreshold = 0;
};

struct RemoteResPtr {
    uint64_t nextHostPtr = 0;
    uint64_t nextDevicePtr = 0;
};

struct HcclWorkspaceInfo {
    uint64_t workspace = 0;
    uint64_t workspaceSize = 0;
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId = 0;
    uint32_t remoteWorldRank = 0;
    uint64_t windowsIn = 0;
    uint64_t windowsOut = 0;
    uint64_t windowsExp = 0;
    ListCommon nextTagRes{};
};

struct HcclOpResParamHead {
    uint32_t localUsrRankId = 0;
    uint32_t rankSize = 0;
    uint64_t winSize = 0;
    uint64_t localWindowsIn = 0;
    uint64_t localWindowsOut = 0;
    char hcomId[128] = {};
    uint64_t winExpSize = 0;
    uint64_t localWindowsExp = 0;
};

struct HcclOpResParam {
    HcclWorkspaceInfo workSpaceInfo{};
    uint32_t localUsrRankId = 0;
    uint32_t rankSize = 0;
    uint64_t winSize = 0;
    uint64_t localWindowsIn = 0;
    uint64_t localWindowsOut = 0;
    char hcomId[128] = {};
    uint64_t winExpSize = 0;
    uint64_t localWindowsExp = 0;
    uint32_t rWinStart = 0;
    uint32_t rWinOffset = 0;
    uint64_t version = 0;
    LocalResInfoV2 localRes{};
    AlgoTopoInfo topoInfo{};
    HcclOpConfig config{};
    uint64_t hostStateInfo = 0;
    uint64_t aicpuStateInfo = 0;
    uint64_t lockAddr = 0;
    uint32_t rsv[16] = {};
    uint32_t notifysize = 0;
    uint32_t remoteResNum = 0;
    RemoteResPtr remoteRes[1] = {};
};

} // namespace pto_hccl_compat

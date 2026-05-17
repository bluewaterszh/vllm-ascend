#include "tiling_builder.hpp"

namespace mc2::v4 {

WorkspaceLayoutConfig BuildDefaultWorkspaceLayout() {
    return WorkspaceLayoutConfig{};
}

KernelLaunchConfig BuildDefaultLaunchConfig() {
    KernelLaunchConfig cfg{};
    cfg.expertsPerRank = 2;
    cfg.numExperts = 2;
    cfg.worldSize = 2;
    cfg.numExpertGroups = 2;
    cfg.maxOutputSize = 8;
    return cfg;
}

}  // namespace mc2::v4

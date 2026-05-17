#pragma once

#include "op_kernel/dispatch_ffn_combine_tiling.h"

namespace mc2::v4 {

WorkspaceLayoutConfig BuildDefaultWorkspaceLayout();
KernelLaunchConfig BuildDefaultLaunchConfig();

}  // namespace mc2::v4

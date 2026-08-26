from pathlib import Path
import shutil

root=Path(__file__).resolve().parent
source_header=root.parent/"hvac_pid_controller.hpp"
source_generated=root.parent/"generated_policy.hpp"
source_sketch=root/"common"/"sketch.ino"
for target in (root/"esp32",root/"stm32f103"):
    shutil.copy2(source_header,target/"hvac_pid_controller.hpp")
    shutil.copy2(source_generated,target/"generated_policy.hpp")
    shutil.copy2(source_sketch,target/"sketch.ino")
    print(target)

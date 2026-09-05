# 在 OMEdit 图形界面查看仿真

## 为什么命令行运行后没有弹出模拟器

`run_openmodelica_validation.ps1` 调用的是无界面的 `omc` 编译器。它完成编译、DASSL 数值求解、CSV 导出和报告刷新，但不会启动 OMEdit，也不会显示 3D 动画。

本项目是集中参数热网络模型。可视内容主要是热容、热阻、热源组件及温度/负荷/制冷指令曲线，不是空调外观动画。

## 打开模型

双击项目根目录下的 `open_modelica_gui.bat`。脚本会打开：

`HVACAI > PrecisionCabinetCooling`

该启动器只对本次 OMEdit 子进程设置一个纯英文的 AppData 目录，位置是
`modelica_runtime/omedit_appdata`。这是为了绕过 OpenModelica 1.27 在当前中文
Windows 用户名上的编码问题；不会修改系统级环境变量，也不会改变其他软件的
AppData。启动器还会复制 OpenModelica 安装目录自带的离线库索引，因此加载
Modelica Standard Library 不依赖联网下载。

如果左侧库树只显示 `HVACAI`，展开它并双击 `PrecisionCabinetCooling`。

## 在 OMEdit 中重新运行

1. 选择 `Simulation > Simulation Setup`。
2. 设置 `Start Time = 0`、`Stop Time = 43200`、`Interval = 6`、`Tolerance = 1e-8`、求解器 `dassl`。
3. 点击工具栏的绿色“Simulate”按钮。
4. 运行结束后进入 `Plotting`，在变量树中勾选：
   - `zone.T`：室内温度，单位 K；换算摄氏度需减 273.15。
   - `wall.T`：墙体/机柜慢热质温度，单位 K。
   - `outdoorK.y`：室外昼夜温度，单位 K。
   - `commandInput.y`：给定的制冷容量指令，0.45 表示 45%。
   - `actuator.y`：经过 8 min 延迟和 6 min 惯性后的实际容量比例。
   - `totalLoad.y`：设备、人员和开门造成的总内部热负荷，单位 W。

建议先单独勾选 `zone.T`、`outdoorK.y`，再新建图窗查看 `commandInput.y`、`actuator.y` 和 `totalLoad.y`，避免不同单位共用纵轴。

已经由命令行生成的中文摄氏度对比图位于 `outputs_review_v3/openmodelica_cross_validation.png`，原始 OpenModelica 结果位于 `outputs_review_v3/modelica/PrecisionCabinetCooling_res.csv`。

within HVACAI;
model PrecisionCabinetCooling
  "Two-state single-zone variable-speed DX cooling reference"
  import Modelica.Units.SI;
  parameter SI.HeatCapacity CZone=2.45e6;
  parameter SI.HeatCapacity CWall=16e6;
  parameter SI.ThermalResistance ROutZone=0.014;
  parameter SI.ThermalResistance RZoneWall=0.008;
  parameter SI.ThermalResistance ROutWall=0.024;
  parameter SI.Power equipmentLoad=550;
  parameter SI.Power coolingCapacity=7600;
  parameter SI.Time actuatorDelay=480;
  parameter SI.Time actuatorTimeConstant=360;
  parameter Real initialCommand(min=0, max=1)=0.45 "Compressor capacity before step";
  parameter Real commandStep(min=0, max=1)=0.12 "Identification step magnitude";
  parameter SI.Temperature TZoneInitial=301.65 "28.5 degC";
  parameter SI.Temperature TWallInitial=303.30 "Python reset: 0.7*Tzone + 0.3*ToutBase";
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor zone(C=CZone, T(start=TZoneInitial, fixed=true))
    annotation(Placement(transformation(extent={{20,10},{40,30}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor wall(C=CWall, T(start=TWallInitial, fixed=true))
    annotation(Placement(transformation(extent={{20,65},{40,85}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor outZone(G=1/ROutZone)
    annotation(Placement(transformation(extent={{-35,10},{-15,30}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor zoneWall(G=1/RZoneWall)
    annotation(Placement(transformation(origin={30,48}, extent={{-10,-10},{10,10}}, rotation=90)));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor outWall(G=1/ROutWall)
    annotation(Placement(transformation(extent={{-35,65},{-15,85}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature outdoorBoundary
    annotation(Placement(transformation(extent={{-65,38},{-45,58}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow equipment
    annotation(Placement(transformation(extent={{0,-20},{20,0}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow cooling
    annotation(Placement(transformation(extent={{45,-20},{25,0}})));
  Modelica.Blocks.Sources.Sine outdoorK(amplitude=4, f=1/86400, offset=307.15, phase=-Modelica.Constants.pi/12)
    annotation(Placement(transformation(extent={{-95,38},{-75,58}})));
  Modelica.Blocks.Sources.TimeTable occupiedLoad(table=[0,0;5399,0;5400,1100;34199,1100;34200,0;43200,0])
    annotation(Placement(transformation(extent={{-95,-25},{-75,-5}})));
  Modelica.Blocks.Sources.TimeTable doorLoad(table=[0,0;17999,0;18000,2600;18720,2600;18721,0;43200,0])
    annotation(Placement(transformation(extent={{-95,-55},{-75,-35}})));
  Modelica.Blocks.Math.Add3 totalLoad
    annotation(Placement(transformation(extent={{-45,-30},{-25,-10}})));
  Modelica.Blocks.Sources.Constant baseLoad(k=equipmentLoad)
    annotation(Placement(transformation(extent={{-95,5},{-75,25}})));
  Modelica.Blocks.Sources.Step commandInput(height=commandStep, offset=initialCommand, startTime=7200)
    annotation(Placement(transformation(extent={{-95,-90},{-75,-70}})));
  Modelica.Blocks.Nonlinear.FixedDelay delay(delayTime=actuatorDelay)
    annotation(Placement(transformation(extent={{-65,-90},{-45,-70}})));
  Modelica.Blocks.Continuous.FirstOrder actuator(
    T=actuatorTimeConstant,
    initType=Modelica.Blocks.Types.Init.InitialOutput,
    y_start=initialCommand)
    annotation(Placement(transformation(extent={{-35,-90},{-15,-70}})));
  Modelica.Blocks.Math.Gain coolingGain(k=-coolingCapacity)
    annotation(Placement(transformation(extent={{-5,-90},{15,-70}})));
equation
  connect(outdoorK.y, outdoorBoundary.T);
  connect(outdoorBoundary.port, outZone.port_a);
  connect(outZone.port_b, zone.port);
  connect(outdoorBoundary.port, outWall.port_a);
  connect(outWall.port_b, wall.port);
  connect(zone.port, zoneWall.port_a);
  connect(zoneWall.port_b, wall.port);
  connect(baseLoad.y, totalLoad.u1);
  connect(occupiedLoad.y, totalLoad.u2);
  connect(doorLoad.y, totalLoad.u3);
  connect(totalLoad.y, equipment.Q_flow);
  connect(equipment.port, zone.port);
  connect(commandInput.y, delay.u);
  connect(delay.y, actuator.u);
  connect(actuator.y, coolingGain.u);
  connect(coolingGain.y, cooling.Q_flow);
  connect(cooling.port, zone.port);
  annotation(
    experiment(StartTime=0, StopTime=43200, Interval=6),
    Diagram(coordinateSystem(extent={{-100,-100},{100,100}})));
end PrecisionCabinetCooling;

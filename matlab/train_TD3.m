

clear
close all
clc

%% TD3 training script for the nonlinear MIMO coupled-tank system
% This script trains a Twin-Delayed Deep Deterministic Policy Gradient (TD3)
% agent using the Simulink environment defined in rldepositos.slx.
%
% State vector:
% s = [-e1, -e2, dy1/dt, dy2/dt]'
%
% Action vector:
% a = [dp/dt, dv/dt]'
%
% The plant parameters are numerically included in rldepositos.slx.

observationInfo = rlNumericSpec([4 1],...
    'LowerLimit',[-100 -100  -inf   -inf]',...
    'UpperLimit',[ 100  100 inf  inf]');
observationInfo.Name = 'observations';
observationInfo.Description = '-e1, -e2, derivative of measured level 1, derivative of measured level 2';

actionInfo = rlNumericSpec([2 1],'LowerLimit',[-inf  -inf]','UpperLimit',[inf inf]');
actionInfo.Name = 'actions';
actionInfo.Description = 'pump command derivative and valve command derivative';

% Simulink reinforcement learning environment.

env = rlSimulinkEnv("rldepositos","rldepositos/RL Agent",observationInfo,actionInfo);

% Randomize the reference levels and reset the initial tank levels at the
% beginning of each episode.
env.ResetFcn = @(in)localResetFcn(in);

% Simulation settings.
Ts = 1.0; % Agent sample time [s]
Tf = 150; % Episode duration [s]

% Random seed for reproducibility.
rng(0)


%% Critic networks
% TD3 uses two independent critics to reduce overestimation of the
% action-value function. Both critics receive the observation and action
% vectors as inputs. A quadratic layer is used to match the quadratic reward
% formulation used in the paper.

obsPath = featureInputLayer(observationInfo.Dimension(1),Name="netObsIn");
actPath = featureInputLayer(actionInfo.Dimension(1),Name="netActIn");

commonPath = [
    concatenationLayer(1,2,Name="concat")
    quadraticLayer
    fullyConnectedLayer(1,Name="value", ...
        BiasLearnRateFactor=0,Bias=0)
    ];

criticNetwork = layerGraph(obsPath);
criticNetwork = addLayers(criticNetwork,actPath);
criticNetwork = addLayers(criticNetwork,commonPath);

criticNetwork = connectLayers(criticNetwork,"netObsIn","concat/in1");
criticNetwork = connectLayers(criticNetwork,"netActIn","concat/in2");

criticNetwork = dlnetwork(criticNetwork);


critic1 = rlQValueFunction( initialize(criticNetwork),observationInfo,actionInfo, ObservationInputNames="netObsIn", ActionInputNames="netActIn");
critic2 = rlQValueFunction( initialize(criticNetwork),observationInfo,actionInfo, ObservationInputNames="netObsIn", ActionInputNames="netActIn");

% Initialize the second critic differently from the first one.
critic2.Learnables{1}=dlarray(rand(1,21));

%% Actor network
% The TD3 actor is constrained to a linear deterministic policy. The weights
% of the fully connected layer define the feedback gain matrix used by the
% final controller.
actorNetwork = [
    featureInputLayer(observationInfo.Dimension(1))
    fullyConnectedLayer(actionInfo.Dimension(1), BiasLearnRateFactor=0,Bias=[0;0])
     ];

actorNetwork = dlnetwork(actorNetwork);
actorNetwork = initialize(actorNetwork);

actor = rlContinuousDeterministicActor(actorNetwork,observationInfo,actionInfo);

%% Optimizer and agent options

% Critic optimizer options.
criticOptions = rlOptimizerOptions( ...
    Optimizer="adam", ...
    LearnRate=1e-1,... 
    GradientThreshold=1, ...
    L2RegularizationFactor=2e-3);

% Actor optimizer options.
actorOptions = rlOptimizerOptions( ...
    Optimizer="adam", ...
    LearnRate=1e-1,...
    GradientThreshold=1, ...
    L2RegularizationFactor=1e-4);

agentOptions = rlTD3AgentOptions;
agentOptions.SampleTime = Ts;
agentOptions.DiscountFactor = 0.99;
agentOptions.TargetSmoothFactor = 1e-3;
agentOptions.TargetPolicySmoothModel.Variance = 0.2;
agentOptions.TargetPolicySmoothModel.LowerLimit = -0.5;
agentOptions.TargetPolicySmoothModel.UpperLimit = 0.5;
agentOptions.CriticOptimizerOptions = criticOptions;
agentOptions.ActorOptimizerOptions = actorOptions;

agent = rlTD3Agent(actor,[critic1 critic2],agentOptions);

%% Actor initialization
% Initialize the actor with negative weights to facilitate convergence of
% the negative-feedback controller.
params = getLearnableParameters(agent);
params.Actor{1} = -single([1 1 1 1; 1 1 1 1]);
agent = setLearnableParameters(agent,params);

% Reward weights used in the Simulink model to balance tracking speed and
% control effort.
Q11=1;      Q22=Q11;
Q33=200;   Q44=Q33;
R11=6;      R22=6;

%% Training options
% Each episode lasts Tf/Ts time steps. Agents are saved when the average
% reward reaches the specified threshold.

trainOpts = rlTrainingOptions(...
    MaxEpisodes=42000, ...
    MaxStepsPerEpisode=ceil(Tf/Ts), ...
    ScoreAveragingWindowLength=20, ...
    Verbose=false, ...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=-10,...
    SaveAgentCriteria= "AverageReward",...
    SaveAgentValue= -3e4);


logger=rlDataLogger();
logger.EpisodeFinishedFcn=@myEpisodeFinishedFcn;


%% Train or load agent
doTraining = true;

if doTraining
trainingStats = train(agent,env,trainOpts,Logger=logger);
save("trained_TD3_agent.mat","agent","trainingStats")
else
load("trained_TD3_agent.mat","agent")
end

%% Local reset function
function in = localResetFcn(in)

% Randomize the reference levels while avoiding very small or excessively
% large combined reference values.

h1 =70*randn; 
h4 =70*randn;   
h14=h1+h4;

while h1 <= 10 || h4 <= 10 || h14 >= 80
            h4 =70*randn;
            h1 =70*randn; 
            h14=h1+h4;
end

% Reference level for tank 1.
blk = sprintf('rldepositos/Desired \nWater Level 1');
in = setBlockParameter(in,blk,'Value',num2str(h1));

% Reference level for tank 2 in the paper, internally named tank 4 in the
% Simulink model.
blk = sprintf('rldepositos/Desired \nWater Level 4');
in = setBlockParameter(in,blk,'Value',num2str(h4));

% Initial liquid level in tank 1.
  h1 = 0;
 blk = sprintf('rldepositos/Depositos/H1'); 
 in = setBlockParameter(in,blk,'InitialCondition',num2str(h1));


% Initial liquid level in tank 2 in the paper, internally named tank 4 in
% the Simulink model.
 h4 = 0;
blk = sprintf('rldepositos/Depositos/H4');
in = setBlockParameter(in,blk,'InitialCondition',num2str(h4));


end


%% Gain-matrix logger

function dataToLog = myEpisodeFinishedFcn(data)

% Extract the actor weights. Since the paper defines u = -Kx and the actor
% implements u = W_actor x, the feedback gain matrix is K = -W_actor.

params = getLearnableParameters(data.Agent);
K = -double(params.Actor{1})

% Log each gain separately for easier visualization and post-processing.
dataToLog.K11 = K(1,1);
dataToLog.K12 = K(1,2);
dataToLog.K13 = K(1,3);
dataToLog.K14 = K(1,4);
dataToLog.K21 = K(2,1);
dataToLog.K22 = K(2,2);
dataToLog.K23 = K(2,3);
dataToLog.K24 = K(2,4);
end

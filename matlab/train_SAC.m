

clear; close all; clc

%% SAC training script for the nonlinear MIMO coupled-tank system
% This script trains a Soft Actor-Critic (SAC) agent using the Simulink
% environment defined in rldepositos.slx.
%
% State vector:
% s = [-e1, -e2, dy1/dt, dy2/dt]'
%
% Action vector:
% a = [dp/dt, dv/dt]'
%
% The plant parameters are numerically included in rldepositos.slx.
%% Observation and action specifications
observationInfo = rlNumericSpec([4 1],...
    'LowerLimit',[-100 -100  -10   -10]',...
    'UpperLimit',[ 100  100 10  10]');
observationInfo.Name = 'observations';
observationInfo.Description = '-e1, -e2, derivative of measured level 1, derivative of measured level 2';

actionInfo = rlNumericSpec([2 1],'LowerLimit',[-10  -10]','UpperLimit',[10 10]');
actionInfo.Name = 'actions';
actionInfo.Description = 'derivative pump and derivative valve';

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


%% Critic network
% The SAC critic approximates the action-value function Q(s,a). The critic
% receives the observation and action vectors as inputs. A quadratic layer is
% used to match the quadratic reward formulation used in the paper.

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

critic = rlQValueFunction( initialize(criticNetwork),observationInfo,actionInfo, ObservationInputNames="netObsIn", ActionInputNames="netActIn");

%% Actor network

% The SAC actor implements a stochastic Gaussian policy. The network returns
% the mean and standard deviation of the action distribution.

inPath = [ 
    featureInputLayer( ...
        prod(observationInfo.Dimension), ...
        Name="netOin")
    fullyConnectedLayer( ...
        prod(actionInfo.Dimension), ...
        Name="infc") 
    ];

meanPath = [ 
    tanhLayer(Name="tanhMean");
    fullyConnectedLayer(prod(actionInfo.Dimension));
    scalingLayer(Name="scale", ...
    Scale=actionInfo.UpperLimit) 
    ];

sdevPath = [ 
    tanhLayer(Name="tanhStdv");
    fullyConnectedLayer(prod(actionInfo.Dimension));
    softplusLayer(Name="splus") 
    ];


actorNetwork = dlnetwork();
actorNetwork = addLayers(actorNetwork,inPath);
actorNetwork = addLayers(actorNetwork,meanPath);
actorNetwork = addLayers(actorNetwork,sdevPath);


actorNetwork = connectLayers(actorNetwork,"infc","tanhMean/in");
actorNetwork = connectLayers(actorNetwork,"infc","tanhStdv/in");

actorNetwork = initialize(actorNetwork);

actor = rlContinuousGaussianActor(actorNetwork, observationInfo, actionInfo, ...
    ActionMeanOutputNames="scale",...
    ActionStandardDeviationOutputNames="splus",...
    ObservationInputNames="netOin");


%% Optimizer and agent options

criticOptions = rlOptimizerOptions( ...
    Optimizer="adam", ...
    LearnRate=1e-2,... 
    GradientThreshold=1, ...
    L2RegularizationFactor=2e-4);


actorOptions = rlOptimizerOptions( ...
    Optimizer="adam", ...
    LearnRate=1e-2,...
    GradientThreshold=1, ...
    L2RegularizationFactor=1e-5);

% SAC agent options.

agentOptions = rlSACAgentOptions;
agentOptions.SampleTime = Ts;
agentOptions.DiscountFactor = 0.99;
agentOptions.TargetSmoothFactor = 1e-3;
agentOptions.ExperienceBufferLength = 1e6;
agentOptions.MiniBatchSize = 128;

agentOptions.CriticOptimizerOptions = criticOptions;
agentOptions.ActorOptimizerOptions = actorOptions;


agent = rlSACAgent(actor,critic,agentOptions);


%% Critic initialization
% The critic is initialized using a negative-definite quadratic form
% consistent with the reward function used in the paper.

Q11=1;      Q22=Q11;
Q33=200;   Q44=Q33;
R11=6;      R22=6;

W = -single(diag([Q11 Q22 Q33 Q44 R11 R22]));
 
% The quadratic layer stores only the upper triangular coefficients.
 idx = triu(true(6));

params = getLearnableParameters(agent);
params.critic{1} = W(idx)';
agent = setLearnableParameters(agent,params);

%% Training options

trainOpts = rlTrainingOptions(...
    MaxEpisodes=8000, ...
    MaxStepsPerEpisode=ceil(Tf/Ts), ...
    ScoreAveragingWindowLength=20, ...
    Verbose=false, ...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=-10,...
    SaveAgentCriteria= "AverageReward",...
    SaveAgentValue= -4e4);

%% Train or load agent
doTraining = true;

if doTraining
trainingStats = train(agent,env,trainOpts);
save("trained_SAC_agent.mat","agent","trainingStats")
else
load("trained_SAC_agent.mat","agent")
end


%Local reset function
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

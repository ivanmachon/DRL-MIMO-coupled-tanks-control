

clear
close all
clc

%% DDPG training script for the nonlinear MIMO coupled-tank system
% This script trains a Deep Deterministic Policy Gradient (DDPG) agent using
% the Simulink environment defined in rldepositos.slx.
%
% State vector:
% s = [-e1, -e2, dy1/dt, dy2/dt]'
%
% Action vector:
% a = [dp/dt, dv/dt]'
%
% The plant parameters are numerically included in rldepositos.slx.

observationInfo = rlNumericSpec([4 1],...
    'LowerLimit',[-100 -100 -100 -100]',...
    'UpperLimit',[ 100  100 100  100]');
observationInfo.Name = 'observations';
observationInfo.Description = '-e1, -e2, derivative of measured level 1, derivative of measured level 2';

actionInfo = rlNumericSpec([2 1],'LowerLimit',[-100  -100]','UpperLimit',[100 100]');
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


%% Critic network
% The DDPG critic approximates the action-value function Q(s,a). The critic
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

critic = rlQValueFunction(criticNetwork,observationInfo,actionInfo, ...
    ObservationInputNames="netObsIn", ...
    ActionInputNames="netActIn");

%% Actor network
% The DDPG actor is constrained to a linear deterministic policy. The weights
% of the fully connected layer define the feedback gain matrix used by the
% final controller.

actorNetwork = [
    featureInputLayer(observationInfo.Dimension(1))
    fullyConnectedLayer(actionInfo.Dimension(1), BiasLearnRateFactor=0,Bias=[0;0])
     ];

actorNetwork = dlnetwork(actorNetwork);

actor = rlContinuousDeterministicActor(actorNetwork,observationInfo,actionInfo);


agent = rlDDPGAgent(actor,critic);


agent.SampleTime = Ts;

agent.AgentOptions.TargetSmoothFactor = 1e-2;%low values to update slowly the target networks in order to stabilize the training
agent.AgentOptions.DiscountFactor = 1.0;%High discount factor values tend to destabilize the algorithm
agent.AgentOptions.MiniBatchSize = 128;%It must be large enough to achieve deterministic results with low variance but not excessively large, requiring high memory usage.
agent.AgentOptions.ExperienceBufferLength = 1e6; 

agent.AgentOptions.NoiseOptions.Variance = 0.3;
agent.AgentOptions.NoiseOptions.VarianceDecayRate = 1e-3;

agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-01;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 10;
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-01;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 10;


%% Critic initialization
% The critic is initialized using a negative-definite quadratic form
% consistent with the reward function used in the paper.

Q11=1;       Q22=Q11;
Q33=200;    Q44=Q33;
R11=6;        R22=6;

W = -single(diag([Q11 Q22 Q33 Q44 R11 R22]));


% The quadratic layer stores only the upper triangular coefficients.
idx = triu(true(6));
% Update parameters in the actor and critic
par = getLearnableParameters(agent);
% Initialize the actor with negative weights to facilitate convergence of
% the negative-feedback controller.
par.Actor{1} = -single([1 1 1 1; 1 1 1 1;]);


par.Critic{1} = W(idx)';
agent = setLearnableParameters(agent,par);

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
    StopTrainingValue=-1e3,...
    SaveAgentCriteria= "AverageReward",...
    SaveAgentValue= -2e4);

logger=rlDataLogger();
logger.EpisodeFinishedFcn=@myEpisodeFinishedFcn;


%Train the agent using the train function. Training is a computationally intensive process that takes several minutes to complete. To save time while running this example, load a pretrained agent by setting doTraining to false. To train the agent yourself, set doTraining to true.

doTraining = true;

if doTraining
trainingStats = train(agent,env,trainOpts,Logger=logger);
save("trained_DDPG_agent.mat","agent","trainingStats")
else
load("trained_DDPG_agent.mat","agent")
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

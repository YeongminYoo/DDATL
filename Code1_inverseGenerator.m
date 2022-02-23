%% Train inverse generator %%
clc; clear all; close all; warning off;


%% Load input file
% Design variables
load("designVariable_CBOX.mat");

% Responses
load("response_CBOX.mat");

for i = 1:(size(response,2)/2)-1
    response_4D(:,:,1,i) = [response(:,2*i)];
end

augimds = augmentedImageDatastore([size(response,1) 1],response_4D);


%% Define inverse generator network
filterSize = [35 1];

layersInverseGenerator = [
    
    imageInputLayer([size(response,1) 1 1],'Normalization','none','Name','in')
 
    convolution2dLayer(filterSize,8,'Stride',2,'Padding',0,'Name','conv1')
    batchNormalizationLayer('Name','batch1')
    reluLayer('Name','relu1')
        
    convolution2dLayer(filterSize,16,'Stride',2,'Padding',0,'Name','conv2')
    batchNormalizationLayer('Name','batch2')
    reluLayer('Name','relu2')
        
    convolution2dLayer(filterSize,32,'Stride',2,'Padding',0,'Name','conv3')
    batchNormalizationLayer('Name','batch3')
    reluLayer('Name','relu3')
        
    convolution2dLayer(filterSize,64,'Stride',2,'Padding',0,'Name','conv4')
    batchNormalizationLayer('Name','batch4')
    reluLayer('Name','relu4')
        
    convolution2dLayer(filterSize,8,'Stride',2,'Padding',0,'Name','conv5')
    batchNormalizationLayer('Name','batch5')
    reluLayer('Name','relu5')
    
    fullyConnectedLayer(8,'Name','fc') 
    regressionLayer('Name','reg')
    
    ];


%% Train model
% Design variables
x1_training = designVariable(:,1);
x2_training = designVariable(:,2);
x3_training = designVariable(:,3);
x4_training = designVariable(:,4);
x5_training = designVariable(:,5);
x6_training = designVariable(:,6);
x7_training = designVariable(:,7);
x8_training = designVariable(:,8);

for j = 1:size(designVariable,1)
    x_training(j,1) = [x1_training(j,1)];
    x_training(j,2) = [x2_training(j,1)];
    x_training(j,3) = [x3_training(j,1)];
    x_training(j,4) = [x4_training(j,1)];
    x_training(j,5) = [x5_training(j,1)];
    x_training(j,6) = [x6_training(j,1)];
    x_training(j,7) = [x7_training(j,1)];
    x_training(j,8) = [x8_training(j,1)];
end

x_train = x_training;
x_validation = x_training;
response_y = read(augimds);
y_training = cat(4,response_y{:,1}{:});
y_train = y_training;
y_validation = y_training;

miniBatchSize = 93;
validationFrequency = floor(numel(x_train)/miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10000, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.01, ...
    'LearnRateDropPeriod',10000, ...
    'Shuffle','never', ...
    'ValidationData',{y_validation,x_validation}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',false);

[IGnet,info] = trainNetwork(y_train,x_train,layersInverseGenerator,options);
x_prediction = predict(IGnet,y_validation);

predictionError = x_validation - x_prediction;
squares = predictionError.^2;
rmse = sqrt(mean(squares));

f = figure(1);
f.Position;
f.Position(4) = [f.Position(4)/2];
plot(1:size(info.TrainingLoss,2),info.TrainingLoss,'k','LineWidth',2);
xlim([0 10000]);
set(gca,'fontsize',15,'fontname','times new roman');
xlabel('Epoch','fontsize',25,'fontname','times new roman'); 
ylabel('Loss','fontsize',25,'fontname','times new roman');
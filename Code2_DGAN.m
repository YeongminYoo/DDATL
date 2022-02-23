%% Designable generative adversarial network (DGAN) %%
clc; clear all; close all; warning off;


%% Load input file
% Responses
load("response_CBOX");

for i = 1:(size(response,2)/2)-1
    response_y1(:,i) = [response(:,2*i-1)];
    response_y2(:,i) = [response(:,2*i)];
    response_4D(:,:,1,i) = [response(:,2*i)];
end

augimds = augmentedImageDatastore([size(response,1) 1],response_4D);


%% Define generator network
filterSize = [35 1];
latentDim = 14;

layersGenerator = [
    
    imageInputLayer([1 1 latentDim],'Normalization','none','Name','in')
    
    transposedConv2dLayer(filterSize,64,'Name','tconv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    
    transposedConv2dLayer(filterSize,32,'Stride',2,'Cropping',0,'Name','tconv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    
    transposedConv2dLayer(filterSize,16,'Stride',2,'Cropping',0,'Name','tconv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    
    transposedConv2dLayer(filterSize,8,'Stride',2,'Cropping',0,'Name','tconv4')
    batchNormalizationLayer('Name','bn4')
    reluLayer('Name','relu4')
    
    transposedConv2dLayer(filterSize,1,'Stride',2,'Cropping',0,'Name','tconv5')
    tanhLayer('Name','tanh')
    
    ];

% lgraphGenerator = layerGraph(layersGenerator);
% dlnetGenerator = dlnetwork(lgraphGenerator);

load('dlnetGenerator_SMBR.mat');
dlnetGenerator = dlnetGenerator;


%% Define discriminator network
dropoutProb = 0.5;
scale = 0.2;

layersDiscriminator = [
    
    imageInputLayer([size(response,1) 1 1],'Normalization','none','Name','in')
    
    dropoutLayer(dropoutProb,'Name','dropout')
    
    convolution2dLayer(filterSize,8,'Stride',2,'Padding',0,'Name','conv1')
    leakyReluLayer(scale,'Name','lrelu1')
    
    convolution2dLayer(filterSize,16,'Stride',2,'Padding',0,'Name','conv2')
    batchNormalizationLayer('Name','bn2')
    leakyReluLayer(scale,'Name','lrelu2')
    
    convolution2dLayer(filterSize,32,'Stride',2,'Padding',0,'Name','conv3')
    batchNormalizationLayer('Name','bn3')
    leakyReluLayer(scale,'Name','lrelu3')
    
    convolution2dLayer(filterSize,64,'Stride',2,'Padding',0,'Name','conv4')
    batchNormalizationLayer('Name','bn4')
    leakyReluLayer(scale,'Name','lrelu4')
    
    convolution2dLayer(filterSize,1,'Stride',2,'Padding',0,'Name','conv5')
    
    ];

lgraphDiscriminator = layerGraph(layersDiscriminator);
dlnetDiscriminator = dlnetwork(lgraphDiscriminator);


%% Define inverse generator network
load("IGnet.mat");
dlnetInverseGenerator = IGnet;


%% Domain knowledge
domainValue = trapz(response(:,end));


%% Specify training options
epoch = 1000;
miniBatchSize = 18;
augimds.MiniBatchSize = miniBatchSize;
learnRateGenerator = 0.001;
learnRateDiscriminator = 0.001;
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;


%% Train model
global probReal probGenerated scoreGenerator scoreDiscriminator

iteration = 0;
start = tic;
figure(1);
figure(2);
scoreAxes = subplot(1,1,1);
lineScoreGenerator = animatedline(scoreAxes,'Color',[0 0.447 0.741]);
lineScoreDiscriminator = animatedline(scoreAxes,'Color',[0.85 0.325 0.098]);

trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];
result_generatorScore = [];
result_discriminatorScore = [];

% Loop over epochs
for i = 1:epoch    
    
    augimds = shuffle(augimds);
    
    % Loop over mini-batches
    while hasdata(augimds)
        iteration = iteration + 1;

        % Read mini-batch of data
        data = read(augimds);
        
        % Ignore last partial mini-batch of epoch
        if size(data,1) < miniBatchSize
            continue
        end        

        y_training = cat(4,data{:,1}{:});
        min_y2 = min(response_y2(:)); max_y2 = max(response_y2(:));
        y_training_normalize = rescale(y_training,-1,1,'InputMin',min_y2,'InputMax',max_y2);
                
        x1_training = randn(1,1,1,size(y_training,4));
        x2_training = randn(1,1,1,size(y_training,4));
        x3_training = randn(1,1,1,size(y_training,4));
        x4_training = randn(1,1,1,size(y_training,4));
        x5_training = randn(1,1,1,size(y_training,4));
        x6_training = randn(1,1,1,size(y_training,4));
        x7_training = randn(1,1,1,size(y_training,4));
        x8_training = randn(1,1,1,size(y_training,4));
        x9_training = randn(1,1,1,size(y_training,4));
        x10_training = randn(1,1,1,size(y_training,4));
        x11_training = randn(1,1,1,size(y_training,4));
        x12_training = randn(1,1,1,size(y_training,4));
        x13_training = randn(1,1,1,size(y_training,4));
        x14_training = randn(1,1,1,size(y_training,4));
        
        for j = 1:size(y_training,4)
            x_training(:,:,1,j) = [x1_training(:,:,1,j)];
            x_training(:,:,2,j) = [x2_training(:,:,1,j)];
            x_training(:,:,3,j) = [x3_training(:,:,1,j)];
            x_training(:,:,4,j) = [x4_training(:,:,1,j)];
            x_training(:,:,5,j) = [x5_training(:,:,1,j)];
            x_training(:,:,6,j) = [x6_training(:,:,1,j)];
            x_training(:,:,7,j) = [x7_training(:,:,1,j)];
            x_training(:,:,8,j) = [x8_training(:,:,1,j)];
            x_training(:,:,9,j) = [x9_training(:,:,1,j)];
            x_training(:,:,10,j) = [x10_training(:,:,1,j)];
            x_training(:,:,11,j) = [x11_training(:,:,1,j)];
            x_training(:,:,12,j) = [x12_training(:,:,1,j)];
            x_training(:,:,13,j) = [x13_training(:,:,1,j)];
            x_training(:,:,14,j) = [x14_training(:,:,1,j)];
        end
        
        % Convert mini-batch of data to dlarray specify the dimension labels 'SSCB' (spatial, spatial, channel, batch)
        dly_training = dlarray(y_training_normalize,'SSCB');
        dlx_training = dlarray(x_training,'SSCB');
        
        % Evaluate the model gradients and the generator state using dlfeval and the modelGradients function
        [gradientsGenerator,gradientsDiscriminator,stateGenerator,scoreGenerator,scoreDiscriminator] = ...
            dlfeval(@modelGradients,dlnetGenerator,dlnetDiscriminator,dly_training,dlx_training);
        dlnetGenerator.State = stateGenerator;
        
        % Update the discriminator network parameters
        [dlnetDiscriminator.Learnables,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator.Learnables,gradientsDiscriminator,trailingAvgDiscriminator,trailingAvgSqDiscriminator,iteration,learnRateDiscriminator,gradientDecayFactor,squaredGradientDecayFactor);
        
        % Update the generator network parameters
        [dlnetGenerator.Learnables,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator.Learnables,gradientsGenerator,trailingAvgGenerator,trailingAvgSqGenerator,iteration,learnRateGenerator,gradientDecayFactor,squaredGradientDecayFactor);        
        
        % Every 1 iterations, display batch of generated images using the held-out generator input
        if mod(iteration,1) == 0 || iteration == 1
            
            % Generate data using the held-out generator input
            dly_training_generated = predict(dlnetGenerator,dlx_training);
                        
            % Rescale the data and display the data                  
            y_training_generated_normalize = extractdata(dly_training_generated(:,1));            
            y_training_deNormalize = rescale(y_training_generated_normalize,min_y2,max_y2,'InputMin',-1,'InputMax',1);  
                       
            figure(1)
            movegui([350 350]);
            plot(response_y1(:,1),y_training_deNormalize,'k');
            set(gca,'fontsize',15,'fontname','times new roman');
            xlabel('Force [kN]','fontsize',25,'fontname','times new roman');
            ylabel('Displacement [mm]','fontsize',25,'fontname','times new roman');
            xlim([0 200]); 
            ylim([0 160]);
            
            % Update the title with training progress information
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title(...
                "Epoch: " + i + ", " + ...
                "Iteration: " + iteration + ", " + ...
                "Elapsed: " + string(D))
            
            % Update the scores plot
            figure(2)
            movegui([950 350]);
            subplot(1,1,1)
            addpoints(lineScoreGenerator,iteration,double(gather(extractdata(scoreGenerator))));            
            addpoints(lineScoreDiscriminator,iteration,double(gather(extractdata(scoreDiscriminator))));
            ylim([0 1])
            legend('Generator','Discriminator');            
            set(gca,'fontsize',15,'fontname','times new roman');
            xlabel("Iteration",'fontsize',25,'fontname','times new roman')
            ylabel("Score",'fontsize',25,'fontname','times new roman')
            grid on
            
            % Update the title with training progress information
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title(...
                "Epoch: " + i + ", " + ...
                "Iteration: " + iteration + ", " + ...
                "Elapsed: " + string(D))
            drawnow                        
        end
    end
    
    probabilityReal = double(gather(extractdata(mean(probReal))));
    probabilityFake = double(gather(extractdata(mean(probGenerated))));
    generatorScore = double(gather(extractdata(scoreGenerator)));
    discriminatorScore = double(gather(extractdata(scoreDiscriminator)));
        
    result_generatorScore = [result_generatorScore; generatorScore];
    result_discriminatorScore = [result_discriminatorScore; discriminatorScore];
    
    disp("Epoch: " + i + ", " + "GeneratorScore: " + generatorScore + ", " + "DiscriminatorScore: " + discriminatorScore);
    
    if (0.495 <= probabilityReal) & (probabilityReal <= 0.505) & (0.495 <= probabilityFake) & (probabilityFake <= 0.505) & (0.99*domainValue <= trapz(y_training_deNormalize(:,1))) & (trapz(y_training_deNormalize(:,1)) <= 1.01*domainValue)
        break
    end   
end


%% Generate new data
numGenerationData = 93;
x1_test = randn(1,1,1,numGenerationData);  % Design variable x1
x2_test = randn(1,1,1,numGenerationData);  % Design variable x2
x6_test = randn(1,1,1,numGenerationData);  % Design variable x6
x7_test = randn(1,1,1,numGenerationData);  % Design variable x7
x11_test = randn(1,1,1,numGenerationData);  % Design variable x11
x12_test = randn(1,1,1,numGenerationData);  % Design variable x12
x13_test = randn(1,1,1,numGenerationData);  % Design variable x13
x14_test = randn(1,1,1,numGenerationData);  % Design variable x14

for j = 1:numGenerationData
    x_test(:,:,1,j) = [x1_test(:,:,1,j)];
    x_test(:,:,2,j) = [x2_test(:,:,1,j)];
    x_test(:,:,6,j) = [x6_test(:,:,1,j)];
    x_test(:,:,7,j) = [x7_test(:,:,1,j)];
    x_test(:,:,11,j) = [x11_test(:,:,1,j)];
    x_test(:,:,12,j) = [x12_test(:,:,1,j)];
    x_test(:,:,13,j) = [x13_test(:,:,1,j)];
    x_test(:,:,14,j) = [x14_test(:,:,1,j)];
end

dlx_test = dlarray(x_test,'SSCB');

dly_test_generated = predict(dlnetGenerator,dlx_test);
dlx_test_generated = predict(dlnetInverseGenerator,dly_test_generated);

for k = 1:numGenerationData
    y_test_normalize = extractdata(dly_test_generated(:,:,1,k));
    y_test_deNormalize(:,:,1,k) = rescale(y_test_normalize,min_y2,max_y2,'InputMin',-1,'InputMax',1);
end

result_y = [];

for k = 1:numGenerationData
    result_y = [result_y y_test_deNormalize(:,:,1,k)];    
end

result_x1 = dlx_test_generated(:,1); 
result_x2 = dlx_test_generated(:,2); 
result_x6 = dlx_test_generated(:,3);
result_x7 = dlx_test_generated(:,4);
result_x11 = dlx_test_generated(:,5);
result_x12 = dlx_test_generated(:,6);
result_x13 = dlx_test_generated(:,7);
result_x14 = dlx_test_generated(:,8);

figure(3)
for i = 1:numGenerationData
    plot(response_y1(:,1),result_y(:,i),'k'); hold on;
    set(gca,'fontsize',15,'fontname','times new roman');
    xlabel('Force [kN]','fontsize',25,'fontname','times new roman');
    ylabel('Displacement [mm]','fontsize',25,'fontname','times new roman');
    title("Generated Data",'fontsize',25,'fontname','times new roman')
end


%% Results
result_y = result_y;
result_x1 = result_x1';
result_x2 = result_x2';
result_x6 = result_x6';
result_x7 = result_x7';
result_x11 = result_x11';
result_x12 = result_x12';
result_x13 = result_x13';
result_x14 = result_x14';


%% Model gradients function
function [gradientsGenerator,gradientsDiscriminator,stateGenerator,scoreGenerator,scoreDiscriminator] = ...
    modelGradients(dlnetGenerator,dlnetDiscriminator,Y,X)

global probReal probGenerated scoreGenerator scoreDiscriminator

% Calculate the predictions for real data with the discriminator network
YPred = forward(dlnetDiscriminator,Y);

% Calculate the predictions for generated data with the discriminator network
[dlYGenerated,stateGenerator] = forward(dlnetGenerator,X);
YPredGenerated = forward(dlnetDiscriminator,dlYGenerated);

% Convert the discriminator outputs to probabilities
probReal = sigmoid(YPred);
probGenerated = sigmoid(YPredGenerated);

% Calculate the score of the discriminator
scoreDiscriminator = ((mean(probReal)+mean(1-probGenerated))/2);

% Calculate the score of the generator
scoreGenerator = mean(probGenerated);

% Calculate the GAN loss
[lossGenerator,lossDiscriminator] = ganLoss(probReal,probGenerated);

% For each network, calculate the gradients with respect to the loss
gradientsGenerator = dlgradient(lossGenerator,dlnetGenerator.Learnables,'RetainData',true);
gradientsDiscriminator = dlgradient(lossDiscriminator,dlnetDiscriminator.Learnables);
end


%% GAN loss function
function [lossGenerator,lossDiscriminator] = ganLoss(probReal,probGenerated)

% Calculate the loss for the discriminator network
lossDiscriminator = -mean(log(probReal)) - mean(log(1-probGenerated));

% Calculate the loss for the generator network
lossGenerator = -mean(log(probGenerated));
end
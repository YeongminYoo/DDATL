%% Designable variational auto-encoder (DVAE) %%
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


%% Define encoder network
filterSize = [35 1];
latentDim = 14;

layersEncoder = [
    
    imageInputLayer([size(response,1) 1 1],'Name','input_encoder','Normalization','none')
    
    convolution2dLayer(filterSize,8,'Stride',2,'Padding',0,'Name','conv1')
    reluLayer('Name','relu1')
    
    convolution2dLayer(filterSize,16,'Stride',2,'Padding',0,'Name','conv2')
    reluLayer('Name','relu2')
    
    convolution2dLayer(filterSize,32,'Stride',2,'Padding',0,'Name','conv3')
    reluLayer('Name','relu3')
    
    convolution2dLayer(filterSize,64,'Stride',2,'Padding',0,'Name','conv4')
    reluLayer('Name','relu4')
    
    fullyConnectedLayer(2 * latentDim, 'Name', 'fc_encoder')
    
    ];

lgraphEncoder = layerGraph(layersEncoder);
dlnetEncoder = dlnetwork(lgraphEncoder);


%% Define decoder network
layersDecoder = [
    
    imageInputLayer([1 1 latentDim],'Name','i','Normalization','none')
    
    transposedConv2dLayer(filterSize,64,'Name','tcov1')
    reluLayer('Name','relu1')
    
    transposedConv2dLayer(filterSize,32,'Stride',2,'Cropping',0,'Name','tcov2')
    reluLayer('Name','relu2')
    
    transposedConv2dLayer(filterSize,16,'Stride',2,'Cropping',0,'Name','tcov3')
    reluLayer('Name','relu3')
    
    transposedConv2dLayer(filterSize,8,'Stride',2,'Cropping',0,'Name','tcov4')
    reluLayer('Name','relu4')
    
    transposedConv2dLayer(filterSize,1,'Stride',2,'Cropping',0,'Name','tcov5')
    
    ];

% lgraphDecoder = layerGraph(layersDecoder);
% dlnetDecoder = dlnetwork(lgraphDecoder);

load('dlnetDecoder_SMBR.mat');
dlnetDecoder = dlnetDecoder;


%% Define inverse generator network
load("IGnet.mat");
dlnetInverseGenerator = IGnet;


%% Domain knowledge
domainValue = trapz(response(:,end));


%% Specify training options
epoch = 1000;
miniBatchSize = 18;
augimds.MiniBatchSize = miniBatchSize;
learnRateEncoder = 0.001;
learnRateDecoder = 0.001;


%% Train model
global elbo

iteration = 0;
start = tic;
figure(1);
figure(2);
ELBOLossAxes = subplot(1,1,1);
lineLoss = animatedline(ELBOLossAxes,'Color',[1 0 0]);

avgGradientsEncoder = [];
avgGradientsSquaredEncoder = [];
avgGradientsDecoder = [];
avgGradientsSquaredDecoder = [];
result_ELBOLoss = [];

% Loop over epochs
for i = 1:epoch
                                  
    augimds = shuffle(augimds);
    
    % Loop over mini-batches
    while hasdata(augimds)
        iteration = iteration + 1;

        % Read mini-batch of data.
        data = read(augimds);
        
        % Ignore last partial mini-batch of epoch.
        if size(data,1) < miniBatchSize
            continue
        end
                
        y_training = cat(4,data{:,1}{:});
        min_y2 = min(response_y2(:)); max_y2 = max(response_y2(:));
        y_training_normalize = rescale(y_training,-1,1,'InputMin',min_y2,'InputMax',max_y2);
        
        x1_training = randn(1,size(y_training,4))';
        x2_training = randn(1,size(y_training,4))';
        x3_training = randn(1,size(y_training,4))';
        x4_training = randn(1,size(y_training,4))';
        x5_training = randn(1,size(y_training,4))';
        x6_training = randn(1,size(y_training,4))';
        x7_training = randn(1,size(y_training,4))';
        x8_training = randn(1,size(y_training,4))';
        x9_training = randn(1,size(y_training,4))';
        x10_training = randn(1,size(y_training,4))';
        x11_training = randn(1,size(y_training,4))';
        x12_training = randn(1,size(y_training,4))';
        x13_training = randn(1,size(y_training,4))';
        x14_training = randn(1,size(y_training,4))';
        
        for j = 1:size(y_training,4)
            x_training(1,j) = [x1_training(j,1)];
            x_training(2,j) = [x2_training(j,1)];
            x_training(3,j) = [x3_training(j,1)];
            x_training(4,j) = [x4_training(j,1)];
            x_training(5,j) = [x5_training(j,1)];
            x_training(6,j) = [x6_training(j,1)];
            x_training(7,j) = [x7_training(j,1)];
            x_training(8,j) = [x8_training(j,1)];
            x_training(9,j) = [x9_training(j,1)];
            x_training(10,j) = [x10_training(j,1)];
            x_training(11,j) = [x11_training(j,1)];
            x_training(12,j) = [x12_training(j,1)];
            x_training(13,j) = [x13_training(j,1)];
            x_training(14,j) = [x14_training(j,1)];
            x_training_4D(:,:,1,j) = [x_training(1,j)];
            x_training_4D(:,:,2,j) = [x_training(2,j)];
            x_training_4D(:,:,3,j) = [x_training(3,j)];
            x_training_4D(:,:,4,j) = [x_training(4,j)];
            x_training_4D(:,:,5,j) = [x_training(5,j)];
            x_training_4D(:,:,6,j) = [x_training(6,j)];
            x_training_4D(:,:,7,j) = [x_training(7,j)];
            x_training_4D(:,:,8,j) = [x_training(8,j)];
            x_training_4D(:,:,9,j) = [x_training(9,j)];
            x_training_4D(:,:,10,j) = [x_training(10,j)];
            x_training_4D(:,:,11,j) = [x_training(11,j)];
            x_training_4D(:,:,12,j) = [x_training(12,j)];
            x_training_4D(:,:,13,j) = [x_training(13,j)];
            x_training_4D(:,:,14,j) = [x_training(14,j)];            
        end
                
        % Convert mini-batch of data to dlarray specify the dimension labels 'SSCB' (spatial, spatial, channel, batch)
        dly_training = dlarray(y_training_normalize,'SSCB');
        dlx_training = dlarray(x_training_4D,'SSCB');
                
        % Evaluate the model gradients and the generator state using dlfeval and the modelGradients function
        [infGrad,genGrad] = ...
            dlfeval(@modelGradients,dlnetEncoder,dlnetDecoder,dly_training,dlx_training);
        
        % Update the decoder network parameters
        [dlnetDecoder.Learnables,avgGradientsDecoder,avgGradientsSquaredDecoder] = ...
            adamupdate(dlnetDecoder.Learnables,genGrad,avgGradientsDecoder,avgGradientsSquaredDecoder,iteration,learnRateDecoder);
        
        % Update the encoder network parameters
        [dlnetEncoder.Learnables,avgGradientsEncoder,avgGradientsSquaredEncoder] = ...
            adamupdate(dlnetEncoder.Learnables,infGrad,avgGradientsEncoder,avgGradientsSquaredEncoder,iteration,learnRateEncoder);
            
        [dlz_training,zMean,zLogvar] = sampling(dlnetEncoder,dly_training,x_training);                  
            
        % Every 1 iterations, display batch of generated images using the held-out generator input
        if mod(iteration,1) == 0 || iteration == 1
                   
            % Generate data using the held-out generator input          
            dly_training_generated = predict(dlnetDecoder,dlz_training);            
                                                
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
            addpoints(lineLoss,iteration,double(gather(extractdata(elbo))));            
            legend('ELBO');            
            set(gca,'fontsize',15,'fontname','times new roman');
            xlabel("Iteration",'fontsize',25,'fontname','times new roman')
            ylabel("Loss",'fontsize',25,'fontname','times new roman')
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
                    
    elbo = double(gather(extractdata(ELBOloss(dly_training,dly_training_generated,zMean,zLogvar))));
        
    result_ELBOLoss = [result_ELBOLoss; elbo];
        
    disp("Epoch: " + i + ", " + "ELBOLoss: " + elbo);

    if (elbo <= 0.5) & (0.99*domainValue <= trapz(y_training_deNormalize(:,1))) & (trapz(y_training_deNormalize(:,1)) <= 1.01*domainValue)
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

dly_test_generated = sigmoid(predict(dlnetDecoder,dlx_test));
dlx_test_generated = predict(dlnetInverseGenerator,dly_test_generated);

for k = 1:numGenerationData
    y_test_normalize = extractdata(dly_test_generated(:,:,1,k));
    y_test_deNormalize(:,:,1,k) = rescale(y_test_normalize,min_y2,max_y2,'InputMin',-1,'InputMax',1);
end

Convert4Ddlarray_y_test = dlarray(y_test_deNormalize);

result_y = [];

for k = 1:numGenerationData
    result_y = [result_y extractdata(Convert4Ddlarray_y_test(:,:,1,k))];    
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
function [infGrad,genGrad] = modelGradients(dlnetEncoder,dlnetDecoder,Y,X)
[X,XMean,XLogvar] = sampling(dlnetEncoder,Y,X);
YPred = forward(dlnetDecoder,X);
loss = ELBOloss(Y,YPred,XMean,XLogvar);
[genGrad,infGrad] = dlgradient(loss,dlnetDecoder.Learnables,dlnetEncoder.Learnables);
end


%% Sampling function
function [zSampled,zMean,zLogvar] = sampling(dlnetEncoder,Y,X)
compressed = forward(dlnetEncoder,Y);
d = size(compressed,1)/2;
zMean = compressed(1:d,:);
zLogvar = compressed(1+d:end,:);
sz = size(zMean);
epsilon = X;
sigma = exp(.5 * zLogvar);
z = epsilon .* sigma + zMean;
z = reshape(z,[1,1,sz]);
zSampled = dlarray(z,'SSCB');
end


%% Loss function
function elbo = ELBOloss(Y,YPred,XMean,XLogvar)

global elbo

squares = 0.5*(YPred-Y).^2;
reconstructionLoss  = sum(squares,[1,2,3]);

KL = -.5 * sum(1 + XLogvar - XMean.^2 - exp(XLogvar),1);

elbo = mean(reconstructionLoss + KL);
end
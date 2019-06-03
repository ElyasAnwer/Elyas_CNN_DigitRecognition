%% Machine learning Project 6
% This program uses Matlab built-in CNN function to train network using
% MNIST database (LeCun),including 60000 training and 10000 testing images.
% After training is complete, live demo test with webcam capturing image
% then predict based on trained network (Modified LeNet-5 network architecture
% along optimization: stochastic gradient descent with momentum)
% Various Matlab built-in image processing fucntions used for binarization,
% noise filtering, labelling, cropping . Main task is to construct the
% input image as close to as LeCun's dataset which is resize image to
% 20by20 then pad with zeros to make it 28by28, finally translate it
% centerd at center of mass which greatly increases success rate of recognition.
% Ailiyasi Ainiwaer (Elyas) 05/13/2018

clc;close all; clear all;
load('Train_data')
load('Test_data')
Ntrain = size(Train_images,3);
Ntest = size(Test_images,3);
%% Construct training data set
Xtrain = reshape(Train_images,[28 28 1 Ntrain]);
Ytrain = categorical(Train_labels);
%% Defien layer architecture and optimization parameters
layers = [ ...
    imageInputLayer([28 28 1])
    convolution2dLayer(5,6)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,16)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',3,...
    'InitialLearnRate',1e-4, ...
    'Verbose',0, ...
    'Plots','training-progress','ExecutionEnvironment','parallel');
%% Train network
tic;
trainde_net = trainNetwork(Xtrain,Ytrain,layers,options);
Training_time = toc
%% Train accuracy
Ytrainpred = categorical(zeros(size(Ytrain)));
for i = 1 : Ntrain
    Ytrainpred(i) = classify(trainde_net,Xtrain(:,:,1,i));
end
Train_accuracy = (length(find(Ytrain == Ytrainpred))/ Ntrain)*100
%% Construct test data set
Xtest = reshape(Test_images,[28 28 1 Ntest]);
Ytest = categorical(Test_labels);
%% Prediction with trained network
Ypred = categorical(zeros(size(Ytest)));
for i = 1 : Ntest
    Ypred(i) = classify(trainde_net,Xtest(:,:,1,i));
end
Test_accuracy = (length(find(Ytest == Ypred))/ Ntest)*100
%% plot incorrect prediction
[err_ind,b] = find(Ytest ~= Ypred);
figure(3)
for i = 1:10
    subplot(2,5,i);
    imshow(Test_images(:,:,err_ind(i)));
    True = Test_labels(err_ind(i));
    title(['True:' num2str(True) '  Predicted:' char(Ypred(err_ind(i)))])
end
%% plot correct prediction
[corr_ind,b] = find(Ytest == Ypred);
figure(23)
for i = 1:10
    subplot(2,5,i);
    imshow(Test_images(:,:,corr_ind(i)));
    True = Test_labels(corr_ind(i));
    title(['True:' num2str(True) '  Predicted:' char(Ypred(corr_ind(i)))])
end
%% Acquire image
vid = videoinput('winvideo',1,'MJPG_640x360');
preview(vid)
start(vid)
pause(9)
init_imag=getsnapshot(vid);
closepreview(vid);
clear('vid');
figure(11);imshow(init_imag);
%% Image processing
Ima_gray = rgb2gray(init_imag);
Ima_gray = Ima_gray(20:size(Ima_gray,1)-20,20:size(Ima_gray,2)-20);
figure(31)
imshow(Ima_gray)

Image_intrev = 255 - Ima_gray;
Image_bina = im2uint8(imbinarize(Image_intrev,'adaptive','ForegroundPolarity','bright','Sensitivity',0.4));
figure(35)
imshow(Image_bina);

[sep_digits,total_digits]= bwlabel(Image_bina);
measurementBOX=regionprops(sep_digits,'BoundingBox');

if total_digits < 50
    for i=1:total_digits
        Digits=imcrop(Image_bina,measurementBOX(i).BoundingBox);
        %crop image should be 20by20 based on LeCun data
        Digits = imresize(Digits,[20 20]);
        % Padd zeros to make it 28by28
        Pad_img = padarray(Digits,[4 4],0,'both');
        %translate by center of mass
        binaryImage = true(size(Pad_img));
        labeledIma = logical(binaryImage);
        measurementCM = regionprops(labeledIma, Pad_img, 'WeightedCentroid');
        Pad_CM = imtranslate(Pad_img,[14- measurementCM.WeightedCentroid(1), 14- measurementCM.WeightedCentroid(2)]);
        % prediction with trained network
        [pre,scores] = classify(trainde_net,Pad_CM);
        figure;
        imshow(Pad_CM);
        title(['Predicted:',char(pre),'Confidence:',num2str(max(scores))]);
        if total_digits<=10 && total_digits>=3
             figure(99)
            subplot(2,5,i);
            imshow(Pad_CM);
            title(['Predicted:',char(pre),'Confidence:',num2str(max(scores))]);
        end
        
    end
else
    disp('Too much BG noise')
end

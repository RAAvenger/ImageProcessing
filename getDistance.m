%% load training datas
load('data\\HOGs.mat');
load('data\\LBPs.mat');
%% get test pics
testImages = dir('test\\*.jpg');
testImagesSize = numel(testImages);
input = rgb2gray(imread([testImages(1).folder, '\', testImages(1).name]));
%% get HOG and LBP Features
HOG = extractHOGFeatures(input, 'CellSize', size(input) / cellSize,'BlockSize', [cellSize cellSize], 'NumBins', binCount);
LBP = extractLBPFeatures(input);
%% get training pics Count
c1s = size(class1HOGs);
c2s = size(class2HOGs);
c1s = c1s(1);
c2s = c2s(1);
%% get distances
% HOGDistances = zeros(1, c1s + c2s);
% LBPDistances = zeros(1, c1s + c2s);
% for i=1:c1s + c2s
%     if i <= c1s
%         HOGDistances(i) = sqrt(sum((HOG - class1HOGs(i)) .^ 2));
%         LBPDistances(i) = sqrt(sum((LBP - class1LBPs(i)) .^ 2));
%     else
%         HOGDistances(i) = sqrt(sum((HOG - class2HOGs(i - c1s)) .^ 2));
%         LBPDistances(i) = sqrt(sum((LBP - class2LBPs(i - c1s)) .^ 2));
%     end
% end
% avgDistanceHOGClass1 = mean(HOGDistances(1:c1s));
% avgDistanceHOGClass2 = mean(HOGDistances(c1s + 1:c1s + c2s));
% avgDistanceLBPClass1 = mean(LBPDistances(1:c1s));
% avgDistanceLBPClass2 = mean(LBPDistances(c1s + 1:c1s + c2s));
clearvars binCount cellSize i input testImages testImagesSize
% r = fitcknn(rot90(class1HOGs), rot90(HOG), 'NumNeighbors',3,'KFold',10,'Standardize',1);
% classError = kfoldLoss(r);
HOG
LBP
%% load training datas
load('data\\HOGs.mat');
load('data\\LBPs.mat');
%% get training pics Count
c1s = size(class1HOGs);
c2s = size(class2HOGs);
c1s = c1s(1);
c2s = c2s(1);
%% get trained models
% go to "training functions" directory to use functions
cd 'Training Functions'
HOGs = class1HOGs;
HOGs(c1s + 1:c1s + c2s, :) = class2HOGs;
HOGs(1:c1s,binCount * cellSize ^ 2 + 1) = 1;
HOGs(c1s + 1:c1s + c2s,binCount * cellSize ^ 2 + 1) = 2;
LBPs = class1LBPs;
LBPs(c1s + 1:c1s + c2s, :) = class2LBPs;
LBPs(1:c1s,60) = 1;
LBPs(c1s + 1:c1s + c2s,60) = 2;
knnHOG = KNNTrainClassifierForHOG(HOGs);
svmHOG = SVMTrainClassifierForHOG(HOGs);
knnLBP = KNNTrainClassifierForLBP(LBPs);
svmLBP = SVMTrainClassifierForLBP(LBPs);
% go back to main directory
cd ..
%% get test pics
testImages = dir('test\\*.jpg');
testImagesSize = numel(testImages);
class = zeros(testImagesSize + 1, 4);
class(1,:) = [resubLoss(knnHOG.ClassificationKNN), resubLoss(svmHOG.ClassificationSVM), resubLoss(knnLBP.ClassificationKNN), resubLoss(svmLBP.ClassificationSVM)];
for i=2:testImagesSize + 1
    input = rgb2gray(imread([testImages(i - 1).folder, '\', testImages(i - 1).name]));
    %% get HOG and LBP Features
    HOG = extractHOGFeatures(input, 'CellSize', size(input) / cellSize,'BlockSize', [cellSize cellSize], 'NumBins', binCount);
    LBP = extractLBPFeatures(input);
    %% classify images
    class(i,1) = knnHOG.predictFcn(HOG);
    class(i,2) = svmHOG.predictFcn(HOG);
    class(i,3) = knnLBP.predictFcn(LBP);
    class(i,4) = svmLBP.predictFcn(LBP);
end
clearvars binCount cellSize i input testImages testImagesSize c1s c2s class1HOGs class1LBPs class2HOGs class2LBPs HOG HOGs knnHOG knnLBP LBP LBPs svmHOG svmLBP
save('data\\ClassificationResults.mat');
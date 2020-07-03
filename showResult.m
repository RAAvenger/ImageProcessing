getDistance
load('data\\ClassificationResults.mat');
%% showing Precision
disp('False Negative Rates for functions: ')
disp(FNR)
disp('False Positive Rates for functions: ')
disp(FPR)
precision = 1 - FNR + FPR;
disp('Precision of functions: ')
disp(precision)
%% get test pics
testImages = dir('test\\*.jpg');
testImagesSize = numel(testImages);
for i=2:testImagesSize + 1
    c1 = 0;
    c2 = 0;
    for j=1:4
       if class(i,j) == 1
           c1 = c1 + (1 - class(1,j));
       else
           c2 = c2 + (1 - class(1,j));
       end
    end
    t = sprintf('class1: %d%% class2: %d%%', round(c1 / 4 * 100), round(c2 / 4 * 100));
    input = imread([testImages(i-1).folder, '\', testImages(i-1).name]);
    figure,imshow(input);
    title(t)
end
clearvars c1 c2 i input j t testImages testImagesSize
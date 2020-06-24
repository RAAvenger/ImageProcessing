function HOG( binCount, cellSize)
    switch nargin
        case 0
            binCount = 255;
            cellSize = 1;
        case 1
            cellSize = 1;
    end
    % %     read Class1 Training images.
    class1 = dir('Class1 Training images\\*.jpg');
    class1Size = numel(class1);
    Class1HOGs = zeros(class1Size, binCount * cellSize ^ 2);
    for i = 1:class1Size
        input = rgb2gray(imread([class1(i).folder, '\', class1(i).name]));
        Class1HOGs(i,:) = extractHOGFeatures(input, 'CellSize', size(input) / cellSize, 'BlockSize', [cellSize cellSize], 'NumBins', binCount);
        disp(['i: ', num2str(i)]);    
    end
    % %     read Class2 Training images.
    class2 = dir('Class2 Training images\\*.jpg');
    class2Size = numel(class2);
    Class2HOGs = zeros(class2Size,  binCount * cellSize ^ 2);
    for j = 1:class2Size
        input = rgb2gray(imread([class2(j).folder, '\', class2(j).name]));
        Class2HOGs(j,:) = extractHOGFeatures(input, 'CellSize', size(input) / cellSize, 'BlockSize', [cellSize cellSize], 'NumBins', binCount);
        disp(['j: ', num2str(j)]);
    end
    clearvars i j input class1 class1Size class2 class2Size
    save('data\\HOGs.mat');
end
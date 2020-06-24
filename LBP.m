function LBP()
    % %     read Class1 Training images.
    class1 = dir('Class1 Training images\\*.jpg');
    class1Size = numel(class1);
    class1LBPs = zeros(class1Size, 59);
    for i = 1:class1Size
        input = rgb2gray(imread([class1(i).folder, '\', class1(i).name]));
        class1LBPs(i,:) = extractLBPFeatures(input);
        disp(['i: ', num2str(i)]);    
    end
    % %     read Class2 Training images.
    class2 = dir('Class2 Training images\\*.jpg');
    class2Size = numel(class2);
    class2LBPs = zeros(class2Size,  59);
    for j = 1:class2Size
        input = rgb2gray(imread([class2(j).folder, '\', class2(j).name]));
        class2LBPs(j,:) = extractLBPFeatures(input);
        disp(['j: ', num2str(j)]);
    end
    clearvars i j input class1 class1Size class2 class2Size
    save('data\\LBPs.mat');
end
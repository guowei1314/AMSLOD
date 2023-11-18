% The code of prper "Adaptive multi-view subspace learning based on distributed optimization".

clear all;
tic;
load('Yale.mat');
X{1} = data{1};
X{2} = data{2};
X{3} = data{3};
gt = double(truelabel{1}');

lambda1 = 5;
lambda2 = 5;

omega = [20, 30, 50]; %1*V
result1 = AMSLOD(X, lambda1, lambda2, omega, gt);
result = result1;
fprintf("AC = %5.4f，NMI = %5.4f，purity = %5.4f\n", result1(1), result1(2), result1(3));

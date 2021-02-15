clear all
close all
clc
%% Prediction 결과 분석
Prediction = csvread('predictions.csv');
% Load raw data (GPStest_sensor)
True_label = csvread('RAL_SequentialGesture_MH2_G11.csv');

figure()
plot(Prediction, 'b', 'LineWidth', 2)
hold on
plot(True_label(11,end-size(Prediction,1)+1:end), 'g', 'LineWidth', 2)
hold on
plot(0.9*ones(1,length(Prediction)), 'r--', 'LineWidth', 2)
hold on
plot(0.1*ones(1,length(Prediction)), 'r--', 'LineWidth', 2)
grid on
ylim([-0.2 1.2])
legend('Prediction', 'Ground truth', 'GPS=0.9', 'GPS=0.1')
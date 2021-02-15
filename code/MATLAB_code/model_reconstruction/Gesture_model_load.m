clear all
close all
clc
%% Gesture model load (weight와 bias가 csv 파일 형태로 저장된 것을 로드)
% ML_layers 폴더에 저장된 함수 사용
addpath('ML_layers/');
addpath('weights_from_python/Gesture_model');
% Model structure
% Layer1 (index: 0): Input layer
% Layer2 (index: 1): LSTM layer
% Layer3 (index: 2): LSTM layer
% Layer4 (index: 3): Dense layer
% Layer5 (index: 4): Dense layer
% Layer6 (index: 5): Dense layer
% Layer7 (index: 6): Dense layer (output layer, softmax)

% Layer2 (index: 1): LSTM layer
LSTM_weight_x1_raw = csvread('LSTM_weight_x1.csv');
LSTM_weight_h1_raw = csvread('LSTM_weight_h1.csv');
LSTM_bias1_raw = csvread('LSTM_bias1.csv');
units = size(LSTM_weight_x1_raw,2)/4; % 4로 나누는 이유: gate 개수가 총 4개
[LSTM_weight1,LSTM_bias1] = LSTM_mat2cell(LSTM_weight_x1_raw, LSTM_weight_h1_raw, LSTM_bias1_raw,units);

% Layer3 (index: 2): LSTM layer
LSTM_weight_x2_raw = csvread('LSTM_weight_x2.csv');
LSTM_weight_h2_raw = csvread('LSTM_weight_h2.csv');
LSTM_bias2_raw = csvread('LSTM_bias2.csv');
units = size(LSTM_weight_x2_raw,2)/4; % 4로 나누는 이유: gate 개수가 총 4개
[LSTM_weight2,LSTM_bias2] = LSTM_mat2cell(LSTM_weight_x2_raw, LSTM_weight_h2_raw, LSTM_bias2_raw,units);

% Layer4 (index: 3): Dense layer
Dense_weight3_raw = csvread('Dense_weight3.csv');
Dense_bias3_raw = csvread('Dense_bias3.csv');
[Dense_weight3, Dense_bias3] = Dense_mat2cell(Dense_weight3_raw, Dense_bias3_raw);

% Layer5 (index: 4): Dense layer
Dense_weight4_raw = csvread('Dense_weight4.csv');
Dense_bias4_raw = csvread('Dense_bias4.csv');
[Dense_weight4, Dense_bias4] = Dense_mat2cell(Dense_weight4_raw, Dense_bias4_raw);

% Layer6 (index: 5): Dense layer
Dense_weight5_raw = csvread('Dense_weight5.csv');
Dense_bias5_raw = csvread('Dense_bias5.csv');
[Dense_weight5, Dense_bias5] = Dense_mat2cell(Dense_weight5_raw, Dense_bias5_raw);

% Layer7 (index: 6): Dense layer (output layer, softmax)
Dense_weight6_raw = csvread('Dense_weight6.csv');
Dense_bias6_raw = csvread('Dense_bias6.csv');
[Dense_weight6, Dense_bias6] = Dense_mat2cell(Dense_weight6_raw, Dense_bias6_raw);
%% Test dataset 로드 및 모델 구성
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
filename = 'Access_gesture_class_test.mat';
filedir = fullfile(pathname, filename);
test_dataset = load(filedir);
test_dataset = struct2cell(test_dataset);
test_dataset = test_dataset{1,1};

test_output = cell(size(test_dataset));
true_output = cell(size(test_dataset));
class_mat = eye(size(test_dataset,3));
for i=1:size(test_dataset,3)
    for j=1:size(test_dataset,1)
        input = test_dataset{j,:,i};

        [LSTM_hidden1,~] = LSTM_forward(input,LSTM_weight1,LSTM_bias1);
        [LSTM_hidden2,~] = LSTM_forward(LSTM_hidden1,LSTM_weight2,LSTM_bias2);
        [Dense_hidden1,~] = Dense_forward(LSTM_hidden2(end,:), Dense_weight3, Dense_bias3, 'ReLU');
        [Dense_hidden2,~] = Dense_forward(Dense_hidden1, Dense_weight4, Dense_bias4, 'ReLU');
        [Dense_hidden3,~] = Dense_forward(Dense_hidden2, Dense_weight5, Dense_bias5, 'ReLU');

        [test_output{j,:,i},~] = Dense_forward(Dense_hidden3, Dense_weight6, Dense_bias6, 'softmax');
        true_output{j,:,i} = class_mat(i,:);
    end
end
%% Plot confusion matrix
% matrix expansion
true_label_mat = [];
test_label_mat = [];
for i=1:size(true_output,3)
    for j=1:size(true_output,1)
        true_label_mat = [true_label_mat,true_output{j,:,i}'];
        test_label_mat = [test_label_mat,test_output{j,:,i}'];
    end
end

% confusion matrix
cm = plotconfusion(true_label_mat,test_label_mat);
set(gca,'xticklabel',{'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11','Acc'})
set(gca,'yticklabel',{'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11','Acc'})
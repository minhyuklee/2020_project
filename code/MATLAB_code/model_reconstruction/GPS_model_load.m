clear all
close all
clc
% %% GPS model load (weight와 bias가 csv 파일 형태로 저장된 것을 로드)
% % ML_layers 폴더에 저장된 함수 사용
% addpath('ML_layers/');
% addpath('weights_from_python/GPS_model');
% % Model structure
% % Layer1 (index: 0): Input layer
% % Layer2 (index: 1): LSTM layer
% % Layer3 (index: 2): LSTM layer
% % Layer4 (index: 3): Input layer (a vector at last time step of each window)
% % Layer5 (index: 4): Concatenate layer
% % Layer6 (index: 5): Dense layer
% % Layer7 (index: 6): Dense layer
% % Layer8 (index: 7): Dense layer
% % Layer9 (index: 8): Dense layer
% % Layer10 (index: 9): Dense layer
% % Layer11 (index: 10): Dense layer
% % Layer12 (index: 11): Dense layer (output layer)
%
% % Layer2 (index: 1): LSTM layer
% LSTM_weight_x1_raw = csvread('LSTM_weight_x1.csv');
% LSTM_weight_h1_raw = csvread('LSTM_weight_h1.csv');
% LSTM_bias1_raw = csvread('LSTM_bias1.csv');
% units = size(LSTM_weight_x1_raw,2)/4; % 4로 나누는 이유: gate 개수가 총 4개
% [LSTM_weight1,LSTM_bias1] = LSTM_mat2cell(LSTM_weight_x1_raw, LSTM_weight_h1_raw, LSTM_bias1_raw,units);
% 
% % Layer3 (index: 2): LSTM layer
% LSTM_weight_x2_raw = csvread('LSTM_weight_x2.csv');
% LSTM_weight_h2_raw = csvread('LSTM_weight_h2.csv');
% LSTM_bias2_raw = csvread('LSTM_bias2.csv');
% units = size(LSTM_weight_x2_raw,2)/4; % 4로 나누는 이유: gate 개수가 총 4개
% [LSTM_weight2,LSTM_bias2] = LSTM_mat2cell(LSTM_weight_x2_raw, LSTM_weight_h2_raw, LSTM_bias2_raw,units);
% 
% % Layer6 (index: 5): Dense layer
% Dense_weight5_raw = csvread('Dense_weight5.csv');
% Dense_bias5_raw = csvread('Dense_bias5.csv');
% [Dense_weight5, Dense_bias5] = Dense_mat2cell(Dense_weight5_raw, Dense_bias5_raw);
% 
% % Layer7 (index: 6): Dense layer
% Dense_weight6_raw = csvread('Dense_weight6.csv');
% Dense_bias6_raw = csvread('Dense_bias6.csv');
% [Dense_weight6, Dense_bias6] = Dense_mat2cell(Dense_weight6_raw, Dense_bias6_raw);
% 
% % Layer8 (index: 7): Dense layer
% Dense_weight7_raw = csvread('Dense_weight7.csv');
% Dense_bias7_raw = csvread('Dense_bias7.csv');
% [Dense_weight7, Dense_bias7] = Dense_mat2cell(Dense_weight7_raw, Dense_bias7_raw);
% 
% % Layer9 (index: 8): Dense layer
% Dense_weight8_raw = csvread('Dense_weight8.csv');
% Dense_bias8_raw = csvread('Dense_bias8.csv');
% [Dense_weight8, Dense_bias8] = Dense_mat2cell(Dense_weight8_raw, Dense_bias8_raw);
% 
% % Layer10 (index: 9): Dense layer
% Dense_weight9_raw = csvread('Dense_weight9.csv');
% Dense_bias9_raw = csvread('Dense_bias9.csv');
% [Dense_weight9, Dense_bias9] = Dense_mat2cell(Dense_weight9_raw, Dense_bias9_raw);
% 
% % Layer11 (index: 10): Dense layer
% Dense_weight10_raw = csvread('Dense_weight10.csv');
% Dense_bias10_raw = csvread('Dense_bias10.csv');
% [Dense_weight10, Dense_bias10] = Dense_mat2cell(Dense_weight10_raw, Dense_bias10_raw);
% 
% % Layer12 (index: 11): Dense layer (output layer)
% Dense_weight11_raw = csvread('Dense_weight11.csv');
% Dense_bias11_raw = csvread('Dense_bias11.csv');
% [Dense_weight11, Dense_bias11] = Dense_mat2cell(Dense_weight11_raw, Dense_bias11_raw);
% 
% % Save weights as cell array
% LSTM_layer_num = [1,2];
% Dense_layer_num = [5,6,7,8,9,10,11];
% 
% Directory = "weights_for_MATLAB";
% for i=1:length(LSTM_layer_num)
%     save(fullfile(Directory,append("LSTM_weight",num2str(LSTM_layer_num(i)))),append("LSTM_weight",num2str(LSTM_layer_num(i))));
%     save(fullfile(Directory,append("LSTM_bias",num2str(LSTM_layer_num(i)))),append("LSTM_bias",num2str(LSTM_layer_num(i))));
% end
% for i=1:length(Dense_layer_num)
%     save(fullfile(Directory,append("Dense_weight",num2str(Dense_layer_num(i)))),append("Dense_weight",num2str(Dense_layer_num(i))));
%     save(fullfile(Directory,append("Dense_bias",num2str(Dense_layer_num(i)))),append("Dense_bias",num2str(Dense_layer_num(i))));
% end
%% GPS model_test load (weight와 bias가 csv 파일 형태로 저장된 것을 로드)
% ML_layers 폴더에 저장된 함수 사용
addpath('ML_layers/');
addpath('weights_from_python/GPS_model');
% Model structure
% Layer1 (index: 0): Input layer
% Layer2 (index: 1): LSTM layer
% Layer3 (index: 2): LSTM layer
% Layer4 (index: 3): Dense layer
% Layer5 (index: 4): Dense layer
% Layer6 (index: 5): Dense layer
% Layer7 (index: 6): Dense layer
% Layer8 (index: 7): Dense layer
% Layer9 (index: 8): Dense layer
% Layer10 (index: 9): Dense layer (output layer)

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

% Layer7 (index: 4): Dense layer
Dense_weight4_raw = csvread('Dense_weight4.csv');
Dense_bias4_raw = csvread('Dense_bias4.csv');
[Dense_weight4, Dense_bias4] = Dense_mat2cell(Dense_weight4_raw, Dense_bias4_raw);

% Layer8 (index: 5): Dense layer
Dense_weight5_raw = csvread('Dense_weight5.csv');
Dense_bias5_raw = csvread('Dense_bias5.csv');
[Dense_weight5, Dense_bias5] = Dense_mat2cell(Dense_weight5_raw, Dense_bias5_raw);

% Layer9 (index: 6): Dense layer
Dense_weight6_raw = csvread('Dense_weight6.csv');
Dense_bias6_raw = csvread('Dense_bias6.csv');
[Dense_weight6, Dense_bias6] = Dense_mat2cell(Dense_weight6_raw, Dense_bias6_raw);

% Layer10 (index: 7): Dense layer
Dense_weight7_raw = csvread('Dense_weight7.csv');
Dense_bias7_raw = csvread('Dense_bias7.csv');
[Dense_weight7, Dense_bias7] = Dense_mat2cell(Dense_weight7_raw, Dense_bias7_raw);

% Layer11 (index: 8): Dense layer
Dense_weight8_raw = csvread('Dense_weight8.csv');
Dense_bias8_raw = csvread('Dense_bias8.csv');
[Dense_weight8, Dense_bias8] = Dense_mat2cell(Dense_weight8_raw, Dense_bias8_raw);

% Layer12 (index: 9): Dense layer (output layer)
Dense_weight9_raw = csvread('Dense_weight9.csv');
Dense_bias9_raw = csvread('Dense_bias9.csv');
[Dense_weight9, Dense_bias9] = Dense_mat2cell(Dense_weight9_raw, Dense_bias9_raw);

% Save weights as cell array
LSTM_layer_num = [1,2];
Dense_layer_num = [3,4,5,6,7,8,9];

Directory = "weights_for_MATLAB";
for i=1:length(LSTM_layer_num)
    save(fullfile(Directory,append("LSTM_weight",num2str(LSTM_layer_num(i)))),append("LSTM_weight",num2str(LSTM_layer_num(i))));
    save(fullfile(Directory,append("LSTM_bias",num2str(LSTM_layer_num(i)))),append("LSTM_bias",num2str(LSTM_layer_num(i))));
end
for i=1:length(Dense_layer_num)
    save(fullfile(Directory,append("Dense_weight",num2str(Dense_layer_num(i)))),append("Dense_weight",num2str(Dense_layer_num(i))));
    save(fullfile(Directory,append("Dense_bias",num2str(Dense_layer_num(i)))),append("Dense_bias",num2str(Dense_layer_num(i))));
end
%% Test dataset 로드 및 모델 구성
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
filename = 'gesture_series_test.mat';
filedir = fullfile(pathname,filename);
test_dataset = load(filedir);
test_dataset = struct2cell(test_dataset);
test_dataset = test_dataset{1,1};

test_data_sensor = test_dataset(1:10,:);
test_data_true_label = test_dataset(11,:);

window_size = 35;
test_output = [];
save_concat = [];
for t=1:size(test_data_sensor,2)-window_size+1
    input = test_data_sensor(:,t:t+window_size-1)';
    
    [LSTM_hidden1,~] = LSTM_forward(input,LSTM_weight1,LSTM_bias1);
    [LSTM_hidden2,~] = LSTM_forward(LSTM_hidden1,LSTM_weight2,LSTM_bias2);

    concat = [LSTM_hidden2(end,:),input(end,:)];
    save_concat(t,:) = concat;
    
    [Dense_hidden1,~] = Dense_forward(LSTM_hidden2(end,:), Dense_weight3, Dense_bias3, 'ReLU');
    [Dense_hidden2,~] = Dense_forward(Dense_hidden1, Dense_weight4, Dense_bias4, 'ReLU');
    [Dense_hidden3,~] = Dense_forward(Dense_hidden2, Dense_weight5, Dense_bias5, 'ReLU');
    [Dense_hidden4,~] = Dense_forward(Dense_hidden3, Dense_weight6, Dense_bias6, 'ReLU');
    [Dense_hidden5,~] = Dense_forward(Dense_hidden4, Dense_weight7, Dense_bias7, 'ReLU');
    [Dense_hidden6,~] = Dense_forward(Dense_hidden5, Dense_weight8, Dense_bias8, 'ReLU');
    
    [test_output(:,t),~] = Dense_forward(Dense_hidden6, Dense_weight9, Dense_bias9, 'sigmoid');
end
%%
figure()
plot(test_output, 'b', 'LineWidth', 2)
hold on
plot(test_data_true_label(1,end-size(test_output,2)+1:end), 'g', 'LineWidth', 2)
hold on
plot(0.9*ones(1,length(test_output)), 'r--', 'LineWidth', 2)
hold on
plot(0.1*ones(1,length(test_output)), 'r--', 'LineWidth', 2)
grid on
ylim([-0.2 1.2])
legend('Prediction', 'Ground truth', 'GPS=0.9', 'GPS=0.1')
set(gca,'FontSize',15)
%%
mean_concat_hidden = mean(save_concat(:,1:128),2);
max_concat_hidden = max(save_concat(:,1:128),[],2);
min_concat_hidden = min(save_concat(:,1:128),[],2);

mean_concat_input = mean(save_concat(:,129:138),2);
max_concat_input = max(save_concat(:,129:138),[],2);
min_concat_input = min(save_concat(:,129:138),[],2);

figure()
plot(mean_concat_hidden,'b')
% hold on
% plot(mean_concat_input,'r')
hold on
plot(zeros(size(mean_concat_hidden)),'k')
grid on
% ylim([-2.5, 2.5])
% legend('mean h_t','mean x_t')
legend('mean h_t')
set(gca,'FontSize',14)

figure()
plot(max_concat_hidden,'b')
% hold on
% plot(max_concat_input,'r')
hold on
plot(zeros(size(max_concat_hidden)),'k')
grid on
% ylim([-2.5, 2.5])
% legend('max h_t','max x_t')
legend('max h_t')
set(gca,'FontSize',14)

figure()
plot(min_concat_hidden,'b')
% hold on
% plot(min_concat_input,'r')
hold on
plot(zeros(size(min_concat_hidden)),'k')
grid on
% ylim([-2.5, 2.5])
% legend('min h_t','min x_t')
legend('min h_t')
set(gca,'FontSize',14)
%% Real-time 11 repetitive gesture recognition
clc
clear all
close all
%% Load model
addpath('ML_layers/'); % forward-pass model
addpath('weights_for_MATLAB/GPS_model'); % trained weights

addpath('related_func/'); % useful algorithms

% Load mean/std for gesture spotting algorithm
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
filename = 'RG17_Spotting_stat.mat';
filedir = fullfile(pathname,filename);
Spotting_stat = load(filedir);
Spotting_stat = struct2cell(Spotting_stat);
Spotting_stat = Spotting_stat{1,1};

% Load weights of gesture spotting algorithm
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\model_reconstruction\weights_for_MATLAB\GPS_model';
filename = 'BiLSTM_weight2.mat';
filedir = fullfile(pathname,filename);
Spotting_BiLSTM_weight2 = load(filedir);
Spotting_BiLSTM_weight2 = struct2cell(Spotting_BiLSTM_weight2);
Spotting_BiLSTM_weight2 = Spotting_BiLSTM_weight2{1,1};

filename = 'BiLSTM_bias2.mat';
filedir = fullfile(pathname,filename);
Spotting_BiLSTM_bias2 = load(filedir);
Spotting_BiLSTM_bias2 = struct2cell(Spotting_BiLSTM_bias2);
Spotting_BiLSTM_bias2 = Spotting_BiLSTM_bias2{1,1};

filename = 'Dense_weight3.mat';
filedir = fullfile(pathname,filename);
Spotting_Dense_weight3 = load(filedir);
Spotting_Dense_weight3 = struct2cell(Spotting_Dense_weight3);
Spotting_Dense_weight3 = Spotting_Dense_weight3{1,1};

filename = 'Dense_bias3.mat';
filedir = fullfile(pathname,filename);
Spotting_Dense_bias3 = load(filedir);
Spotting_Dense_bias3 = struct2cell(Spotting_Dense_bias3);
Spotting_Dense_bias3 = Spotting_Dense_bias3{1,1};

filename = 'Dense_weight4.mat';
filedir = fullfile(pathname,filename);
Spotting_Dense_weight4 = load(filedir);
Spotting_Dense_weight4 = struct2cell(Spotting_Dense_weight4);
Spotting_Dense_weight4 = Spotting_Dense_weight4{1,1};

filename = 'Dense_bias4.mat';
filedir = fullfile(pathname,filename);
Spotting_Dense_bias4 = load(filedir);
Spotting_Dense_bias4 = struct2cell(Spotting_Dense_bias4);
Spotting_Dense_bias4 = Spotting_Dense_bias4{1,1};

filename = 'Dense_weight5.mat';
filedir = fullfile(pathname,filename);
Spotting_Dense_weight5 = load(filedir);
Spotting_Dense_weight5 = struct2cell(Spotting_Dense_weight5);
Spotting_Dense_weight5 = Spotting_Dense_weight5{1,1};

filename = 'Dense_bias5.mat';
filedir = fullfile(pathname,filename);
Spotting_Dense_bias5 = load(filedir);
Spotting_Dense_bias5 = struct2cell(Spotting_Dense_bias5);
Spotting_Dense_bias5 = Spotting_Dense_bias5{1,1};

filename = 'Dense_weight6.mat';
filedir = fullfile(pathname,filename);
Spotting_Dense_weight6 = load(filedir);
Spotting_Dense_weight6 = struct2cell(Spotting_Dense_weight6);
Spotting_Dense_weight6 = Spotting_Dense_weight6{1,1};

filename = 'Dense_bias6.mat';
filedir = fullfile(pathname,filename);
Spotting_Dense_bias6 = load(filedir);
Spotting_Dense_bias6 = struct2cell(Spotting_Dense_bias6);
Spotting_Dense_bias6 = Spotting_Dense_bias6{1,1};

filename = 'Dense_weight7.mat';
filedir = fullfile(pathname,filename);
Spotting_Dense_weight7 = load(filedir);
Spotting_Dense_weight7 = struct2cell(Spotting_Dense_weight7);
Spotting_Dense_weight7 = Spotting_Dense_weight7{1,1};

filename = 'Dense_bias7.mat';
filedir = fullfile(pathname,filename);
Spotting_Dense_bias7 = load(filedir);
Spotting_Dense_bias7 = struct2cell(Spotting_Dense_bias7);
Spotting_Dense_bias7 = Spotting_Dense_bias7{1,1};

filename = 'Dense_weight8.mat';
filedir = fullfile(pathname,filename);
Spotting_Dense_weight8 = load(filedir);
Spotting_Dense_weight8 = struct2cell(Spotting_Dense_weight8);
Spotting_Dense_weight8 = Spotting_Dense_weight8{1,1};

filename = 'Dense_bias8.mat';
filedir = fullfile(pathname,filename);
Spotting_Dense_bias8 = load(filedir);
Spotting_Dense_bias8 = struct2cell(Spotting_Dense_bias8);
Spotting_Dense_bias8 = Spotting_Dense_bias8{1,1};

filename = 'Dense_weight9.mat';
filedir = fullfile(pathname,filename);
Spotting_Dense_weight9 = load(filedir);
Spotting_Dense_weight9 = struct2cell(Spotting_Dense_weight9);
Spotting_Dense_weight9 = Spotting_Dense_weight9{1,1};

filename = 'Dense_bias9.mat';
filedir = fullfile(pathname,filename);
Spotting_Dense_bias9 = load(filedir);
Spotting_Dense_bias9 = struct2cell(Spotting_Dense_bias9);
Spotting_Dense_bias9 = Spotting_Dense_bias9{1,1};

% Load mean/std for gesture recognition algorithm
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
filename = 'RG17_Recognition_stat.mat';
filedir = fullfile(pathname,filename);
Recognition_stat = load(filedir);
Recognition_stat = struct2cell(Recognition_stat);
Recognition_stat = Recognition_stat{1,1};

% Load weights of gesture spotting algorithm
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\model_reconstruction\weights_for_MATLAB\Gesture_model';
filename = 'LSTM_weight1.mat';
filedir = fullfile(pathname,filename);
Recognition_LSTM_weight1 = load(filedir);
Recognition_LSTM_weight1 = struct2cell(Recognition_LSTM_weight1);
Recognition_LSTM_weight1 = Recognition_LSTM_weight1{1,1};

filename = 'LSTM_bias1.mat';
filedir = fullfile(pathname,filename);
Recognition_LSTM_bias1 = load(filedir);
Recognition_LSTM_bias1 = struct2cell(Recognition_LSTM_bias1);
Recognition_LSTM_bias1 = Recognition_LSTM_bias1{1,1};

% filename = 'LSTM_weight2.mat';
% filedir = fullfile(pathname,filename);
% Recognition_LSTM_weight2 = load(filedir);
% Recognition_LSTM_weight2 = struct2cell(Recognition_LSTM_weight2);
% Recognition_LSTM_weight2 = Recognition_LSTM_weight2{1,1};
% 
% filename = 'LSTM_bias2.mat';
% filedir = fullfile(pathname,filename);
% Recognition_LSTM_bias2 = load(filedir);
% Recognition_LSTM_bias2 = struct2cell(Recognition_LSTM_bias2);
% Recognition_LSTM_bias2 = Recognition_LSTM_bias2{1,1};

filename = 'Dense_weight3.mat';
filedir = fullfile(pathname,filename);
Recognition_Dense_weight3 = load(filedir);
Recognition_Dense_weight3 = struct2cell(Recognition_Dense_weight3);
Recognition_Dense_weight3 = Recognition_Dense_weight3{1,1};

filename = 'Dense_bias3.mat';
filedir = fullfile(pathname,filename);
Recognition_Dense_bias3 = load(filedir);
Recognition_Dense_bias3 = struct2cell(Recognition_Dense_bias3);
Recognition_Dense_bias3 = Recognition_Dense_bias3{1,1};

filename = 'Dense_weight4.mat';
filedir = fullfile(pathname,filename);
Recognition_Dense_weight4 = load(filedir);
Recognition_Dense_weight4 = struct2cell(Recognition_Dense_weight4);
Recognition_Dense_weight4 = Recognition_Dense_weight4{1,1};

filename = 'Dense_bias4.mat';
filedir = fullfile(pathname,filename);
Recognition_Dense_bias4 = load(filedir);
Recognition_Dense_bias4 = struct2cell(Recognition_Dense_bias4);
Recognition_Dense_bias4 = Recognition_Dense_bias4{1,1};

filename = 'Dense_weight5.mat';
filedir = fullfile(pathname,filename);
Recognition_Dense_weight5 = load(filedir);
Recognition_Dense_weight5 = struct2cell(Recognition_Dense_weight5);
Recognition_Dense_weight5 = Recognition_Dense_weight5{1,1};

filename = 'Dense_bias5.mat';
filedir = fullfile(pathname,filename);
Recognition_Dense_bias5 = load(filedir);
Recognition_Dense_bias5 = struct2cell(Recognition_Dense_bias5);
Recognition_Dense_bias5 = Recognition_Dense_bias5{1,1};

filename = 'Dense_weight6.mat';
filedir = fullfile(pathname,filename);
Recognition_Dense_weight6 = load(filedir);
Recognition_Dense_weight6 = struct2cell(Recognition_Dense_weight6);
Recognition_Dense_weight6 = Recognition_Dense_weight6{1,1};

filename = 'Dense_bias6.mat';
filedir = fullfile(pathname,filename);
Recognition_Dense_bias6 = load(filedir);
Recognition_Dense_bias6 = struct2cell(Recognition_Dense_bias6);
Recognition_Dense_bias6 = Recognition_Dense_bias6{1,1};
%% Offline test: 테스트 시퀀스 로드
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data_preprocessing\train_test_data\test_data';
filename = 'Rgesture_series_test.mat';
filedir = fullfile(pathname,filename);
test_sequence = load(filedir);
test_sequence = struct2cell(test_sequence);
test_sequence = test_sequence{1,1}; % test data 내 센서 측정값은 normalize되어있음.
%% Offline test: 테스트 시퀀스에 unified system 적용
reservoir_Spotting = [];
reservoir_Recognition = [];
reservoir_size_Spotting = 60;
GPS_thres_early = 0.1;
GPS_thres_end = 0.9;
tolerance = 0.5;
peak_tol = 30;
output_Spotting = zeros(1,size(test_sequence,2));
output_Recognition = zeros(11,size(test_sequence,2));

for t=1:size(test_sequence,2)
    % Gesture spotting
    if size(reservoir_Spotting,2) < reservoir_size_Spotting
        reservoir_Spotting = [reservoir_Spotting, test_sequence(1:10,t)];
    else
        reservoir_Spotting = [reservoir_Spotting, test_sequence(1:10,t)];
        input_Spotting = reservoir_Spotting(:,end-reservoir_size_Spotting+1:end);
        
        input_norm_Spotting = (input_Spotting - Spotting_stat(1,:)'*ones(1,size(input_Spotting,2)))./(Spotting_stat(2,:)'*ones(1,size(input_Spotting,2)));
        input_norm_Spotting = input_norm_Spotting';
        
        BiLSTM_hidden1 = BiLSTM_forward(input_norm_Spotting, Spotting_BiLSTM_weight2, Spotting_BiLSTM_bias2,'False','concat');

        [Dense_hidden1,~] = Dense_forward(BiLSTM_hidden1(end,:), Spotting_Dense_weight3, Spotting_Dense_bias3, 'ReLU');
        [Dense_hidden2,~] = Dense_forward(Dense_hidden1, Spotting_Dense_weight4, Spotting_Dense_bias4, 'ReLU');
        [Dense_hidden3,~] = Dense_forward(Dense_hidden2, Spotting_Dense_weight5, Spotting_Dense_bias5, 'ReLU');
        [Dense_hidden4,~] = Dense_forward(Dense_hidden3, Spotting_Dense_weight6, Spotting_Dense_bias6, 'ReLU');
        [Dense_hidden5,~] = Dense_forward(Dense_hidden4, Spotting_Dense_weight7, Spotting_Dense_bias7, 'ReLU');
        [Dense_hidden6,~] = Dense_forward(Dense_hidden5, Spotting_Dense_weight8, Spotting_Dense_bias8, 'ReLU');

        [output_Spotting(:,t),~] = Dense_forward(Dense_hidden6, Spotting_Dense_weight9, Spotting_Dense_bias9, 'sigmoid');
        
        if output_Spotting(:,t) < GPS_thres_early
            flag_reservoir = 0;
        end
        if output_Spotting(:,t) > GPS_thres_end
            flag_reservoir = 1;
        end
    end
    
    if t > reservoir_size_Spotting
        if flag_reservoir == 0
            reservoir_Recognition = [reservoir_Recognition,test_sequence(1:10,t)];
            flag_reservoir_old = flag_reservoir;
        end
        
        if (flag_reservoir - flag_reservoir_old) == 1
            input_Recognition = reservoir_Recognition;
            
            input_norep = repetition_removal(input_Recognition);
            input_norep_comp = seq_compress_v2(input_norep,tolerance);
            
            input_norm_Recognition = (input_norep_comp - Recognition_stat(1,:)'*ones(1,size(input_norep_comp,2)))./(Recognition_stat(2,:)'*ones(1,size(input_norep_comp,2)));
            input_norm_Recognition = input_norm_Recognition';
            
            [LSTM_hidden1,~] = LSTM_forward(input_norm_Recognition, Recognition_LSTM_weight1, Recognition_LSTM_bias1);
            [LSTM_hidden2,~] = LSTM_forward(LSTM_hidden1, Recognition_LSTM_weight2, Recognition_LSTM_bias2);
            [Dense_hidden1,~] = Dense_forward(LSTM_hidden2(end,:), Recognition_Dense_weight3, Recognition_Dense_bias3, 'ReLU');
            [Dense_hidden2,~] = Dense_forward(Dense_hidden1, Recognition_Dense_weight4, Recognition_Dense_bias4, 'ReLU');
            [Dense_hidden3,~] = Dense_forward(Dense_hidden2, Recognition_Dense_weight5, Recognition_Dense_bias5, 'ReLU');

            [output_Recognition(:,t),~] = Dense_forward(Dense_hidden3, Recognition_Dense_weight6, Recognition_Dense_bias6, 'softmax');            
            
            reservoir_Recognition = [];
            flag_reservoir_old = flag_reservoir;
        end
    end
end
%% Offline test: test 결과 plot
figure(1)
plot(find(output_Spotting~=0),output_Spotting(:,find(output_Spotting~=0)),'r')
hold on
plot(test_sequence(11,:),'k')
ylim([-0.2 1.2])
grid on

tt = find(sum(output_Recognition,1)~=0);
figure(2)
for i=1:size(tt,2)
    bar(output_Recognition(:,tt(i)))
    ylim([0 1])
    ylabel('Probability')
    title('Test')
    grid on
    drawnow
    pause(1)
end
%% Real-time test
% Max Min 반영하기 (0도:90도 = Minimum:Maximum)
Minimum = load('Minimum.mat');
Minimum = struct2cell(Minimum);
Minimum = Minimum{1,1};
Maximum = load('Maximum.mat');
Maximum = struct2cell(Maximum);
Maximum = Maximum{1,1};

% Image load
Image_filename = {'RG1_Pants','RG2_Milk','RG3_Who','RG4_Horse','RG5_Bird','RG6_Cry','RG7_Doubt','RG8_No','G9_Like','G10_Want','G11_Best','G12_Why','G13_Justkidding','G14_Locate','G15_Looklike','G16_Mindfreeze','G17_Finishtouch'};
for i=1:size(Image_filename,2)
    Image_filename{1,i} = strcat(Image_filename{1,i},'.png');
end
Directory = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\model_reconstruction\제스처 사진\';
Image_variable = {};
for i=1:size(Image_filename,2)
    Image_variable{i,1} = imread(strcat(Directory,Image_filename{1,i}));
end

packet_length = 26;
time = 1;
dataset = [];
finger = [];

reservoir_Spotting = [];
reservoir_Recognition = [];
reservoir_size_Spotting = 20;
GPS_thres_early = 0.1;
GPS_thres_end = 0.9;
tolerance_Recognition = 0.5;
peak_tol_Recognition = 25;
t_Spotting = 1;
t_Recognition = 1;

flag_second_phase = 0;
flag_third_phase = 0;
clear output_Spotting output_Recognition
fig = figure(1);
fig.PaperPositionMode = 'auto';
% raw_finger = [];
% k = 1;
while 1
    tmp = fread(s);
    packet_start = find(tmp == 26);
    check = 0;
    for start_point=1:size(packet_start,1)
        check = tmp(packet_start(start_point,:)+(packet_length-1),:)  == 255;
        if check == 1
            break
        end
    end
    
    j = 1;
    count = 0;
    err_check = 0;
    for i=start_point:size(packet_start,1)
        if packet_start(i,:)+(packet_length-1) > size(tmp,1)
            break;
        end
        err_check = (tmp(packet_start(i,:)+(packet_length-1),:) ~= 255);
        if (packet_start(i,:) > 1) && ((tmp(packet_start(i,:)-1,:) ~= 255))
            err_check = err_check + 1;
        end 
        if err_check >= 1
            continue;
        end
        dataset(:,j) = tmp(packet_start(i,:):packet_start(i,:)+(packet_length-1),:);
        for idx=1:10
            finger(idx,j) = 255*dataset(4+2*(idx-1),j) + dataset(4+2*(idx-1)+1,j);
        end
        j = j + 1;
        count = count + 1;
    end
    
    % Min/Max 반영
    for i=1:size(finger,2)
        finger(:,i) = 90*(finger(:,i) - Minimum)./(Maximum - Minimum);
    end
    finger_use = finger(:,1:18);
%     raw_finger = [raw_finger,finger_use];
    
    % 1단계: Gesture spotting
    if size(reservoir_Spotting,2) < reservoir_size_Spotting
        reservoir_Spotting = [reservoir_Spotting,finger_use];
    else
        reservoir_Spotting = [reservoir_Spotting,finger_use];
        input_Spotting = reservoir_Spotting(:,end-reservoir_size_Spotting+1:end);
        
        input_norm_Spotting = (input_Spotting - Spotting_stat(1,:)'*ones(1,size(input_Spotting,2)))./(Spotting_stat(2,:)'*ones(1,size(input_Spotting,2)));
        input_norm_Spotting = input_norm_Spotting';
        
        BiLSTM_hidden1 = BiLSTM_forward(input_norm_Spotting,Spotting_BiLSTM_weight2,Spotting_BiLSTM_bias2,'False','concat');

        [Dense_hidden1,~] = Dense_forward(BiLSTM_hidden1(end,:), Spotting_Dense_weight3, Spotting_Dense_bias3, 'ReLU');
        [Dense_hidden2,~] = Dense_forward(Dense_hidden1, Spotting_Dense_weight4, Spotting_Dense_bias4, 'ReLU');
        [Dense_hidden3,~] = Dense_forward(Dense_hidden2, Spotting_Dense_weight5, Spotting_Dense_bias5, 'ReLU');
        [Dense_hidden4,~] = Dense_forward(Dense_hidden3, Spotting_Dense_weight6, Spotting_Dense_bias6, 'ReLU');
        [Dense_hidden5,~] = Dense_forward(Dense_hidden4, Spotting_Dense_weight7, Spotting_Dense_bias7, 'ReLU');
        [Dense_hidden6,~] = Dense_forward(Dense_hidden5, Spotting_Dense_weight8, Spotting_Dense_bias8, 'ReLU');

        [output_Spotting(:,t_Spotting),~] = Dense_forward(Dense_hidden6, Spotting_Dense_weight9, Spotting_Dense_bias9, 'sigmoid');
        
        subplot(1,2,1)
        plot(output_Spotting,'k','LineWidth',2)
        hold on
        plot([1,10^8],[0.9,0.9],'r','LineWidth',2)
        hold on
        plot([1,10^8],[0.1,0.1],'r','LineWidth',2)
        
        if t_Spotting < reservoir_size_Spotting + 1
            axis([1,reservoir_size_Spotting,-0.2,1.2])
        else
            axis([t_Spotting-(reservoir_size_Spotting-1),t_Spotting,-0.2,1.2])
        end
        grid on
%         title(['Loop time',' ',num2str(t_Spotting)],'FontSize',24)
        title('GPS detection','FontSize',24)
        hold off
        drawnow
        
        t_Spotting = t_Spotting + 1;
        
        % 제스처 데이터 저장 유무 결정
        if output_Spotting(:,end) < GPS_thres_early
            reservoir_Recognition = finger_use;
            flag_second_phase = 1;            
        elseif output_Spotting(:,end) > GPS_thres_end
            flag_second_phase = 0;
        else
            if isempty(reservoir_Recognition) == 1
                flag_second_phase = 0;
            else
                flag_second_phase = 1;
            end
        end
    end
    
    % 2단계: reservoir_recognition에 데이터 저장
    if flag_second_phase == 1
        reservoir_Recognition = [reservoir_Recognition,finger_use];
    elseif flag_second_phase == 0
        if isempty(reservoir_Recognition) == 1
            continue
        else
            input_Recognition = reservoir_Recognition;
            reservoir_Recognition = [];
            flag_third_phase = 1;
        end
    end
    
    % 3단계: Gesture recognition
    if flag_third_phase == 1
        input_norep = repetition_removal(input_Recognition);
        input_norep_comp = seq_compress_v2(input_norep,tolerance_Recognition);

        input_norm_Recognition = (input_norep_comp - Recognition_stat(1,:)'*ones(1,size(input_norep_comp,2)))./(Recognition_stat(2,:)'*ones(1,size(input_norep_comp,2)));
        input_norm_Recognition = input_norm_Recognition';

        [LSTM_hidden1,~] = LSTM_forward(input_norm_Recognition, Recognition_LSTM_weight1, Recognition_LSTM_bias1);
%         [LSTM_hidden2,~] = LSTM_forward(LSTM_hidden1, Recognition_LSTM_weight2, Recognition_LSTM_bias2);
        [Dense_hidden1,~] = Dense_forward(LSTM_hidden1(end,:), Recognition_Dense_weight3, Recognition_Dense_bias3, 'ReLU');
        [Dense_hidden2,~] = Dense_forward(Dense_hidden1, Recognition_Dense_weight4, Recognition_Dense_bias4, 'ReLU');
        [Dense_hidden3,~] = Dense_forward(Dense_hidden2, Recognition_Dense_weight5, Recognition_Dense_bias5, 'ReLU');

        [output_Recognition(:,t_Recognition),~] = Dense_forward(Dense_hidden3, Recognition_Dense_weight6, Recognition_Dense_bias6, 'softmax');
        
        subplot(1,2,2)
        [~,whichone] = max(output_Recognition(:,end));
        imshow(Image_variable{whichone,1})
        drawnow
        
        t_Recognition = t_Recognition + 1;
        
        flag_third_phase = 0;
    end
%     if time > 200
%         xmin = 18*k-200+1;
%         xmax = 18*k;
%         x1 = xmin:1:xmax;        
%         TMCP = raw_finger(1,xmin:xmax);
%         TIP = raw_finger(2,xmin:xmax);
%         IMCP = raw_finger(3,xmin:xmax);
%         IPIP = raw_finger(4,xmin:xmax);
%         MMCP = raw_finger(5,xmin:xmax);
%         MPIP = raw_finger(6,xmin:xmax);
%         RMCP = raw_finger(7,xmin:xmax);
%         RPIP = raw_finger(8,xmin:xmax);
%         LMCP = raw_finger(9,xmin:xmax);
%         LPIP = raw_finger(10,xmin:xmax);
%     else
%         TMCP = raw_finger(1,:);
%         TIP = raw_finger(2,:);
%         IMCP = raw_finger(3,:);
%         IPIP = raw_finger(4,:);
%         MMCP = raw_finger(5,:);
%         MPIP = raw_finger(6,:);
%         RMCP = raw_finger(7,:);
%         RPIP = raw_finger(8,:);
%         LMCP = raw_finger(9,:);
%         LPIP = raw_finger(10,:);
%         x1 = 1:1:18*k;
%     end
%     figure(1)
%     cla
%     plot(x1,TMCP)
%     hold on
%     plot(x1,TIP)
%     hold on
%     plot(x1,IMCP)
%     hold on
%     plot(x1,IPIP)
%     hold on
%     plot(x1,MMCP)
%     hold on
%     plot(x1,MPIP)
%     hold on
%     plot(x1,RMCP)
%     hold on
%     plot(x1,RPIP)
%     hold on
%     plot(x1,LMCP)
%     hold on
%     plot(x1,LPIP)    
%     ylim([-40 150])
%     xlim([xmin, xmax])
%     grid on
%     drawnow
%     k = k + 1;
    
    time = time + count;
end
% Real-time GPS recognition test
clc
clear all
close all
%% Model load
addpath('D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\model_reconstruction\weights_for_MATLAB');
addpath('D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\model_reconstruction\ML_layers');

% Layer1: LSTM
LSTM_weight1 = load('LSTM_weight1.mat');
LSTM_weight1 = struct2cell(LSTM_weight1);
LSTM_weight1 = LSTM_weight1{1,1};

LSTM_bias1 = load('LSTM_bias1.mat');
LSTM_bias1 = struct2cell(LSTM_bias1);
LSTM_bias1 = LSTM_bias1{1,1};

% Layer2: LSTM
LSTM_weight2 = load('LSTM_weight2.mat');
LSTM_weight2 = struct2cell(LSTM_weight2);
LSTM_weight2 = LSTM_weight2{1,1};

LSTM_bias2 = load('LSTM_bias2.mat');
LSTM_bias2 = struct2cell(LSTM_bias2);
LSTM_bias2 = LSTM_bias2{1,1};

% Layer3: Dense
Dense_weight3 = load('Dense_weight3.mat');
Dense_weight3 = struct2cell(Dense_weight3);
Dense_weight3 = Dense_weight3{1,1};

Dense_bias3 = load('Dense_bias3.mat');
Dense_bias3 = struct2cell(Dense_bias3);
Dense_bias3 = Dense_bias3{1,1};

% Layer4: Dense
Dense_weight4 = load('Dense_weight4.mat');
Dense_weight4 = struct2cell(Dense_weight4);
Dense_weight4 = Dense_weight4{1,1};

Dense_bias4 = load('Dense_bias4.mat');
Dense_bias4 = struct2cell(Dense_bias4);
Dense_bias4 = Dense_bias4{1,1};

% Layer5: Dense
Dense_weight5 = load('Dense_weight5.mat');
Dense_weight5 = struct2cell(Dense_weight5);
Dense_weight5 = Dense_weight5{1,1};

Dense_bias5 = load('Dense_bias5.mat');
Dense_bias5 = struct2cell(Dense_bias5);
Dense_bias5 = Dense_bias5{1,1};

% Layer6: Dense
Dense_weight6 = load('Dense_weight6.mat');
Dense_weight6 = struct2cell(Dense_weight6);
Dense_weight6 = Dense_weight6{1,1};

Dense_bias6 = load('Dense_bias6.mat');
Dense_bias6 = struct2cell(Dense_bias6);
Dense_bias6 = Dense_bias6{1,1};

% Layer7: Dense
Dense_weight7 = load('Dense_weight7.mat');
Dense_weight7 = struct2cell(Dense_weight7);
Dense_weight7 = Dense_weight7{1,1};

Dense_bias7 = load('Dense_bias7.mat');
Dense_bias7 = struct2cell(Dense_bias7);
Dense_bias7 = Dense_bias7{1,1};

% Layer8: Dense
Dense_weight8 = load('Dense_weight8.mat');
Dense_weight8 = struct2cell(Dense_weight8);
Dense_weight8 = Dense_weight8{1,1};

Dense_bias8 = load('Dense_bias8.mat');
Dense_bias8 = struct2cell(Dense_bias8);
Dense_bias8 = Dense_bias8{1,1};

% Layer9: Dense
Dense_weight9 = load('Dense_weight9.mat');
Dense_weight9 = struct2cell(Dense_weight9);
Dense_weight9 = Dense_weight9{1,1};

Dense_bias9 = load('Dense_bias9.mat');
Dense_bias9 = struct2cell(Dense_bias9);
Dense_bias9 = Dense_bias9{1,1};
%% ON LINE TEST
% Max Min 반영하기 (0도:90도 = Minimum:Maximum)
Minimum = load('Minimum.mat');
Minimum = struct2cell(Minimum);
Minimum = Minimum{1,1};
Maximum = load('Maximum.mat');
Maximum = struct2cell(Maximum);
Maximum = Maximum{1,1};

% Load mean/std for input normalization
Spotting_stat = load('G11_Spotting_stat.mat');
Spotting_stat = struct2cell(Spotting_stat);
Spotting_stat = Spotting_stat{1,1};

packet_length = 61;
time = 1;
BT_data = [];
finger = [];
finger_use = zeros(10,1);

reservoir_Spotting = [];
reservoir_size_Spotting = 35; % 35
t = 0;
t_Spotting = 1;
step1_time = [];

fig = figure(1);
set(fig,'Position',[0 0 1500 800])
while 1
    tmp = fread(s);
    packet_start = find(tmp == 64);
    check = 0;
    for start_point=1:size(packet_start,1)
        check = tmp(packet_start(start_point,:)+(packet_length-1),:)  == 10;
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
        err_check = (tmp(packet_start(i,:)+(packet_length-1),:) ~= 10);
        if (packet_start(i,:) > 1) && ((tmp(packet_start(i,:)-1,:) ~= 10))
            err_check = err_check + 1;
        end 
        if err_check >= 1
            continue;
        end
        BT_data(:,j) = tmp(packet_start(i,:):packet_start(i,:)+(packet_length-1),:);
        for idx=1:10
            finger(idx,j) = 10^3*str2double(char(BT_data(19+4*(idx-1),j))) + 10^2*str2double(char(BT_data(19+4*(idx-1)+1,j))) + 10^1*str2double(char(BT_data(19+4*(idx-1)+2,j))) + str2double(char(BT_data(19+4*(idx-1)+3,j)));
        end
        j = j + 1;
        count = count + 1;
    end
    
    % Min/Max 반영
    for i=1:size(finger,2)
        finger(:,i) = 90*(finger(:,i) - Minimum)./(Maximum - Minimum);
    end  
    finger_use = finger(:,1:7);
    t = t + 7;
    
    if size(reservoir_Spotting,2) < reservoir_size_Spotting
        reservoir_Spotting = [reservoir_Spotting,finger_use];
    else
        reservoir_Spotting = [reservoir_Spotting,finger_use];
        input_Spotting = reservoir_Spotting(:,end-reservoir_size_Spotting+1:end); % reservoir_Spotting 저장방식 변경으로 (:,end-reservoir_size_Spotting+1:end)로 변경됨. 원래는 그냥 reservoir_Spotting

        input_norm_Spotting = (input_Spotting - Spotting_stat(:,1)*ones(1,size(input_Spotting,2)))./(Spotting_stat(:,2)*ones(1,size(input_Spotting,2)));
        input_norm_Spotting = input_norm_Spotting';
        
        [LSTM_hidden1,~] = LSTM_forward(input_norm_Spotting, LSTM_weight1, LSTM_bias1);
        [LSTM_hidden2,~] = LSTM_forward(LSTM_hidden1, LSTM_weight2, LSTM_bias2);
        
%         concat = [LSTM_hidden2(end,:),input_norm_Spotting(end,:)];
        
        [Dense_hidden1,~] = Dense_forward(LSTM_hidden2(end,:), Dense_weight3, Dense_bias3, 'ReLU');
        [Dense_hidden2,~] = Dense_forward(Dense_hidden1, Dense_weight4, Dense_bias4, 'ReLU');
        [Dense_hidden3,~] = Dense_forward(Dense_hidden2, Dense_weight5, Dense_bias5, 'ReLU');
        [Dense_hidden4,~] = Dense_forward(Dense_hidden3, Dense_weight6, Dense_bias6, 'ReLU');
        [Dense_hidden5,~] = Dense_forward(Dense_hidden4, Dense_weight7, Dense_bias7, 'ReLU');
        [Dense_hidden6,~] = Dense_forward(Dense_hidden5, Dense_weight8, Dense_bias8, 'ReLU');
    
        output_Spotting(:,t_Spotting) = Dense_forward(Dense_hidden6, Dense_weight9, Dense_bias9, 'sigmoid');
        step1_time = [step1_time,toc];
        
        plot(output_Spotting,'k','LineWidth',8)
        hold on
        plot([1,10^8],[0.9,0.9],'r','LineWidth',8)
        hold on
        plot([1,10^8],[0.1,0.1],'r','LineWidth',8)
        
        if t_Spotting < reservoir_size_Spotting + 1
            axis([1,reservoir_size_Spotting,-0.1,1.2])
        else
            axis([t_Spotting-(reservoir_size_Spotting-1),t_Spotting,-0.1,1.2])
        end
        grid on
        title(['Loop time','  ',num2str(t_Spotting)],'FontSize',24)
        hold off
        drawnow
        
        t_Spotting = t_Spotting + 1;
    end

%     figure(2)
%     plot(1:1:count,finger(1,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(2,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(3,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(4,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(5,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(6,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(7,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(8,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(9,time:time+count-1))
%     hold on
%     plot(1:1:count,finger(10,time:time+count-1))    
%     ylim([-120 120])
%     xlim([1 7])
%     grid on
%     drawnow
end
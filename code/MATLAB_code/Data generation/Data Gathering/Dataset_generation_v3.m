%% Dataset generation
% Bluetooth_train.m 파일로 Dataset폴더에 생성한 데이터를 Dataset.mat 파일로 합치기 위한 과정
% 수집된 데이터가 MaxNum을 초과하더라도, 저장은 MaxNum 개수만큼만 저장함.
clc
clear all
close all
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data generation\Data Gathering\data';
MaxNum = 5; % 각 피험자의 각 동작 별 수집할 표본 개수
Sensor_use = [1,2,3,4,5,6,7,8,9,10]; % 사용할 관절 데이터 결정
%% 2020/06/18 MH1
filename = '20200618MH1_RG11.mat';
filedir = fullfile(pathname,filename);
subject_MH1 = load(filedir);
subject_MH1 = struct2cell(subject_MH1);
subject_MH1 = subject_MH1{1,1};
%% 2020/06/18 MH2
filename = '20200618MH2_RG11.mat';
filedir = fullfile(pathname,filename);
subject_MH2 = load(filedir);
subject_MH2 = struct2cell(subject_MH2);
subject_MH2 = subject_MH2{1,1};
%% 2020/06/18 MH3
filename = '20200618MH3_RG11.mat';
filedir = fullfile(pathname,filename);
subject_MH3 = load(filedir);
subject_MH3 = struct2cell(subject_MH3);
subject_MH3 = subject_MH3{1,1};
%% 2020/06/18 MH4
filename = '20200618MH4_RG11.mat';
filedir = fullfile(pathname,filename);
subject_MH4 = load(filedir);
subject_MH4 = struct2cell(subject_MH4);
subject_MH4 = subject_MH4{1,1};
%% 2020/06/18 MH5
filename = '20200618MH5_RG11.mat';
filedir = fullfile(pathname,filename);
subject_MH5 = load(filedir);
subject_MH5 = struct2cell(subject_MH5);
subject_MH5 = subject_MH5{1,1};
%% 2020/06/19 MH6
filename = '20200619MH6_RG11.mat';
filedir = fullfile(pathname,filename);
subject_MH6 = load(filedir);
subject_MH6 = struct2cell(subject_MH6);
subject_MH6 = subject_MH6{1,1};
%% 2020/07/13 MH7
filename = '20200713MH7_RG11.mat';
filedir = fullfile(pathname,filename);
subject_MH7 = load(filedir);
subject_MH7 = struct2cell(subject_MH7);
subject_MH7 = subject_MH7{1,1};
%% 2020/07/13 MH1-RG11-R5 새 장갑으로 변경하여 데이터 재수집
filename = '20200713MH1_RG11_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH1 = load(filedir);
subject_MH1 = struct2cell(subject_MH1);
subject_MH1 = subject_MH1{1,1};
%% 2020/07/13 MH2-RG11-R5
filename = '20200713MH2_RG11_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH2 = load(filedir);
subject_MH2 = struct2cell(subject_MH2);
subject_MH2 = subject_MH2{1,1};
%% 2020/07/13 MH3-RG11-R5
filename = '20200713MH3_RG11_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH3 = load(filedir);
subject_MH3 = struct2cell(subject_MH3);
subject_MH3 = subject_MH3{1,1};
%% 2020/07/13 MH4-RG11-R5
filename = '20200713MH4_RG11_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH4 = load(filedir);
subject_MH4 = struct2cell(subject_MH4);
subject_MH4 = subject_MH4{1,1};
%% 2020/07/14 MH5-RG11-R5
filename = '20200714MH5_RG11_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH5 = load(filedir);
subject_MH5 = struct2cell(subject_MH5);
subject_MH5 = subject_MH5{1,1};
%% 2020/07/14 MH6-RG11-R5
filename = '20200714MH6_RG11_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH6 = load(filedir);
subject_MH6 = struct2cell(subject_MH6);
subject_MH6 = subject_MH6{1,1};
%% 2020/07/14 MH7-RG11-R5
filename = '20200714MH7_RG11_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH7 = load(filedir);
subject_MH7 = struct2cell(subject_MH7);
subject_MH7 = subject_MH7{1,1};
%% 2020/07/14 MH8-RG11-R5
filename = '20200714MH8_RG11_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH8 = load(filedir);
subject_MH8 = struct2cell(subject_MH8);
subject_MH8 = subject_MH8{1,1};
%% 2020/07/14 MH9-RG11-R5
filename = '20200714MH9_RG11_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH9 = load(filedir);
subject_MH9 = struct2cell(subject_MH9);
subject_MH9 = subject_MH9{1,1};
%% 2020/07/14 MH10-RG11-R5
filename = '20200714MH10_RG11_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH10 = load(filedir);
subject_MH10 = struct2cell(subject_MH10);
subject_MH10 = subject_MH10{1,1};
%% 2020/07/15 MH11-RG11-R5
filename = '20200715MH11_RG11_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH11 = load(filedir);
subject_MH11 = struct2cell(subject_MH11);
subject_MH11 = subject_MH11{1,1};
%% 2020/07/15 MH12-RG11-R5
filename = '20200715MH12_RG11_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH12 = load(filedir);
subject_MH12 = struct2cell(subject_MH12);
subject_MH12 = subject_MH12{1,1};
%% 2020/07/15 MH13-RG11-R5
filename = '20200715MH13_RG11_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH13 = load(filedir);
subject_MH13 = struct2cell(subject_MH13);
subject_MH13 = subject_MH13{1,1};
%% 2020/08/05 MH1-RG17-R5
filename = '20200805MH1_RG17_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH14 = load(filedir);
subject_MH14 = struct2cell(subject_MH14);
subject_MH14 = subject_MH14{1,1};
for i=1+(6-1)*5:6*5
    subject_MH14{i,1} = subject_MH14{i,1}(:,1:850);
end
%% 2020/08/05 MH2-RG17-R5
filename = '20200805MH2_RG17_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH15 = load(filedir);
subject_MH15 = struct2cell(subject_MH15);
subject_MH15 = subject_MH15{1,1};
%% 2020/08/05 MH3-RG17-R5
filename = '20200805MH3_RG17_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH16 = load(filedir);
subject_MH16 = struct2cell(subject_MH16);
subject_MH16 = subject_MH16{1,1};
%% 2020/08/06 MH4-RG17-R5
filename = '20200806MH4_RG17_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH17 = load(filedir);
subject_MH17 = struct2cell(subject_MH17);
subject_MH17 = subject_MH17{1,1};
%% 2020/08/06 MH5-RG17-R5
filename = '20200806MH5_RG17_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH18 = load(filedir);
subject_MH18 = struct2cell(subject_MH18);
subject_MH18 = subject_MH18{1,1};
%% 2020/08/06 MH6-RG17-R5
filename = '20200806MH6_RG17_R5.mat';
filedir = fullfile(pathname,filename);
subject_MH19 = load(filedir);
subject_MH19 = struct2cell(subject_MH19);
subject_MH19 = subject_MH19{1,1};
%% 2020/08/10 MH1-RG17-R3
filename = '20200810MH1_RG17_R3.mat';
filedir = fullfile(pathname,filename);
subject_MH20 = load(filedir);
subject_MH20 = struct2cell(subject_MH20);
subject_MH20 = subject_MH20{1,1};
for i=36:40
    subject_MH20{i,1}(4,85:87) = subject_MH20{i,1}(4,84);
    subject_MH20{i,1}(5,85:87) = subject_MH20{i,1}(5,84);
    subject_MH20{i,1}(6,85:87) = subject_MH20{i,1}(6,84);
end
%% 2020/08/10 MH2-RG17-R3
filename = '20200810MH2_RG17_R3.mat';
filedir = fullfile(pathname,filename);
subject_MH21 = load(filedir);
subject_MH21 = struct2cell(subject_MH21);
subject_MH21 = subject_MH21{1,1};
% 81번 제스처 제거
%% 2020/08/10 MH3-RG17-R3
filename = '20200810MH3_RG17_R3.mat';
filedir = fullfile(pathname,filename);
subject_MH22 = load(filedir);
subject_MH22 = struct2cell(subject_MH22);
subject_MH22 = subject_MH22{1,1};
%% 2020/08/11 SW1-RG17-R3 배성욱
filename = '20200811SW1_RG17_R3.mat';
filedir = fullfile(pathname,filename);
subject_SW1 = load(filedir);
subject_SW1 = struct2cell(subject_SW1);
subject_SW1 = subject_SW1{1,1};
for i=31:35
    subject_SW1{i,1}(3,642:644) = subject_SW1{i,1}(3,641);
    subject_SW1{i,1}(4,642:644) = subject_SW1{i,1}(4,641);
    subject_SW1{i,1}(5,642:644) = subject_SW1{i,1}(5,641);
    subject_SW1{i,1}(6,642:644) = subject_SW1{i,1}(6,641);
end
%% 2020/08/11 KT1-RG17-R3 김경택
filename = '20200811KT1_RG17_R3.mat';
filedir = fullfile(pathname,filename);
subject_KT1 = load(filedir);
subject_KT1 = struct2cell(subject_KT1);
subject_KT1 = subject_KT1{1,1};
for i=11:15
    subject_KT1{i,1}(:,1:50) = [];
end
%% 2020/08/11 JUNEH1-RG17-R3 최준혁
filename = '20200811JUNEH1_RG17_R3.mat';
filedir = fullfile(pathname,filename);
subject_JUNEH1 = load(filedir);
subject_JUNEH1 = struct2cell(subject_JUNEH1);
subject_JUNEH1 = subject_JUNEH1{1,1};
for i=6:10 % g_num = 2
    subject_JUNEH1{i,1}(:,750:end) = [];
end
for i=11:15 % g_num = 3
    subject_JUNEH1{i,1}(:,720:end) = [];
end
for i=41:45 % g_num = 9
    subject_JUNEH1{i,1}(:,415:417) = repmat(subject_JUNEH1{i,1}(:,414),1,3);
end
for i=46:50 % g_num = 10
    subject_JUNEH1{i,1}(:,530:end) = [];
end
for i=56:60 % g_num = 12
    subject_JUNEH1{i,1}(:,530:end) = [];
end
for i=81:85 % g_num = 17
    subject_JUNEH1{i,1}(:,560:end) = [];
end
% 56번 제스처 제거
%% 2020/08/11 DY1-RG17-R3 이동영
filename = '20200811DY1_RG17_R3.mat';
filedir = fullfile(pathname,filename);
subject_DY1 = load(filedir);
subject_DY1 = struct2cell(subject_DY1);
subject_DY1 = subject_DY1{1,1};
for i=1:5 % g_num = 1
    subject_DY1{i,1}(:,1:80) = [];
end
for i=16:20 % g_num = 4
    subject_DY1{i,1}(:,660:end) = [];
end
for i=21:25 % g_num = 5
    subject_DY1{i,1}(:,640:end) = [];
end
for i=26:30 % g_num = 6
    subject_DY1{i,1}(:,630:end) = [];
end
for i=36:40 % g_num = 8
    subject_DY1{i,1}(:,660:end) = [];
end
for i=41:45 % g_num = 9
    subject_DY1{i,1}(:,550:end) = [];
end
for i=46:50 % g_num = 10
    subject_DY1{i,1}(:,530:end) = [];
end
for i=51:55 % g_num = 11
    subject_DY1{i,1}(:,525:end) = [];
end
for i=56:60 % g_num = 12
    subject_DY1{i,1}(:,520:end) = [];
end
for i=61:65 % g_num = 13
    subject_DY1{i,1}(:,600:end) = [];
end
for i=66:70 % g_num = 14
    subject_DY1{i,1}(:,1:135) = [];
end 
for i=71:75 % g_num = 15
    subject_DY1{i,1}(:,525:end) = [];
end
for i=76:80 % g_num = 16
    subject_DY1{i,1}(:,515:end) = [];
end
% 11, 43, 46, 68, 69, 70, 75번 제스처 제거
%% 2020/08/19 HJ1-RG17-R3 이호재
filename = '20200819HJ1_RG17_R3.mat';
filedir = fullfile(pathname,filename);
subject_HJ1 = load(filedir);
subject_HJ1 = struct2cell(subject_HJ1);
subject_HJ1 = subject_HJ1{1,1};

for i=46:50 % g_num = 10
    subject_HJ1{i,1}(:,589:end) = [];
end
for i=56:60 % g_num = 12
    subject_HJ1{i,1}(:,1:100) = [];
end
for i=76:80 % g_num = 16
    subject_HJ1{i,1}(:,600:end) = [];
end
% 1, 10, 22, 84번 제스처 제거
%% 2020/08/19 SY1-RG17-R3 이상엽
filename = '20200819SY1_RG17_R3.mat';
filedir = fullfile(pathname,filename);
subject_SY1 = load(filedir);
subject_SY1 = struct2cell(subject_SY1);
subject_SY1 = subject_SY1{1,1};

for i=71:75 % g_num = 15
    subject_HJ1{i,1}(:,522:end) = [];
end
for i=76:80 % g_num = 16
    subject_HJ1{i,1}(:,500:end) = [];
end
% 10, 19, 21, 32, 56, 78번 제스처 제거
%% 2020/08/20 WK1-RG17-R3 박우근
filename = '20200820WK1_RG17_R3.mat';
filedir = fullfile(pathname,filename);
subject_WK1 = load(filedir);
subject_WK1 = struct2cell(subject_WK1);
subject_WK1 = subject_WK1{1,1};

for i=16:20 % g_num = 4
    subject_HJ1{i,1}(:,640:end) = [];
end
% 11, 15, 27, 46, 54, 79번 제스처 제거
%% 2020/08/21 JUNES1-RG17-R3 김준수
filename = '20200821JUNES1_RG17_R3.mat';
filedir = fullfile(pathname,filename);
subject_JUNES1 = load(filedir);
subject_JUNES1 = struct2cell(subject_JUNES1);
subject_JUNES1 = subject_JUNES1{1,1};

for i=46:50 % g_num = 10
    subject_JUNES1{i,1}(:,500:end) = [];
end
% 16, 57, 61번 제스처 제거
%% 2020/08/21 HY1-RG17-R3 염호연
filename = '20200821HY1_RG17_R3.mat';
filedir = fullfile(pathname,filename);
subject_HY1 = load(filedir);
subject_HY1 = struct2cell(subject_HY1);
subject_HY1 = subject_HY1{1,1};

for i=6:10 % g_num = 2
    subject_HY1{i,1}(:,600:end) = [];
end
for i=16:20 % g_num = 4
    subject_HY1{i,1}(:,565:end) = [];
end
for i=21:25 % g_num = 5
    subject_HY1{i,1}(:,585:end) = [];
end
for i=26:30 % g_num = 6
    subject_HY1{i,1}(:,580:end) = [];
end
for i=31:35 % g_num = 7
    subject_HY1{i,1}(:,580:end) = [];
end
for i=36:40 % g_num = 8
    subject_HY1{i,1}(:,550:end) = [];
end
for i=46:50 % g_num = 10
    subject_HY1{i,1}(:,470:end) = [];
end
for i=51:55 % g_num = 11
    subject_HY1{i,1}(:,455:end) = [];
end
for i=56:60 % g_num = 12
    subject_HY1{i,1}(:,465:end) = [];
end
for i=66:70 % g_num = 14
    subject_HY1{i,1}(:,470:end) = [];
end
for i=76:80 % g_num = 16
    subject_HY1{i,1}(:,455:end) = [];
end
for i=81:85 % g_num = 17
    subject_HY1{i,1}(:,430:end) = [];
end
% 27, 36, 63번 제스처 제거
%% 2020/08/31 KKY1-RG17-R3 김광영
filename = '20200831KKY1_RG17_R3.mat';
filedir = fullfile(pathname,filename);
subject_KKY1 = load(filedir);
subject_KKY1 = struct2cell(subject_KKY1);
subject_KKY1 = subject_KKY1{1,1};

for i=11:15 % g_num = 3
    subject_KKY1{i,1}(:,650:end) = [];
end
for i=16:20 % g_num = 4
    subject_KKY1{i,1}(:,640:end) = [];
end
for i=21:25 % g_num = 5
    subject_KKY1{i,1}(:,1:120) = [];
end
for i=26:30 % g_num = 6
    subject_KKY1{i,1}(:,620:end) = [];
end
for i=51:55 % g_num = 11
    subject_KKY1{i,1}(:,500:end) = [];
end
for i=56:60 % g_num = 12
    subject_KKY1{i,1}(:,500:end) = [];
end
for i=71:75 % g_num = 15
    subject_KKY1{i,1}(:,500:end) = [];
end
for i=76:80 % g_num = 16
    subject_KKY1{i,1}(:,500:end) = [];
end
% 36, 70번 제스처 제거
%% Dataset generation (train/test)
% Dataset = [
%             subject_MH20,repmat({1},[size(subject_MH20,1),1]);...
%             subject_MH21,repmat({2},[size(subject_MH21,1),1]);...
%             subject_MH22,repmat({3},[size(subject_MH22,1),1]);...
%             subject_SW1,repmat({4},[size(subject_SW1,1),1]);...
%             subject_KT1,repmat({5},[size(subject_KT1,1),1]);...
%             subject_JUNEH1,repmat({6},[size(subject_JUNEH1,1),1]);...
%             subject_DY1,repmat({7},[size(subject_DY1,1),1]);...
%             subject_HJ1,repmat({8},[size(subject_HJ1,1),1]);...
%             subject_SY1,repmat({9},[size(subject_SY1,1),1]);...
%             subject_WK1,repmat({10},[size(subject_WK1,1),1]);...
%             subject_JUNES1,repmat({11},[size(subject_JUNES1,1),1]);...
%             subject_HY1,repmat({12},[size(subject_HY1,1),1]);...
%             subject_KKY1,repmat({13},[size(subject_KKY1,1),1]);...
%             ];

Dataset = [
            subject_MH14,repmat({1},[size(subject_MH14,1),1]);...
            subject_MH15,repmat({2},[size(subject_MH15,1),1]);...
            subject_MH16,repmat({3},[size(subject_MH16,1),1]);...
            subject_MH17,repmat({4},[size(subject_MH17,1),1]);...
            subject_MH18,repmat({5},[size(subject_MH18,1),1]);...
            subject_MH19,repmat({6},[size(subject_MH19,1),1]);...
            ];
        
% 개별 피험자 확인
% Dataset = [subject_HJ1,repmat({1},[size(subject_HJ1,1),1])];
%% NaN detection
check_nan = []; % col1: subject index, col2: gesture sample index, col3: finger joint index, col4: time step
for i=1:size(Dataset,1)
    indices = find(isnan(Dataset{i,1}) == 1);
    if indices ~= 0
        [row, col] = ind2sub(size(Dataset{i,1}), indices);
        check_nan = [check_nan;i*ones(length(row),1),row,col];
    end    
end
check_nan
%% Replace NaN data to estimated data
if numel(check_nan) ~= 0
    for i=1:size(check_nan,1)
        Dataset{check_nan(i,1),1}(check_nan(i,2),check_nan(i,3)) = 2*Dataset{check_nan(i,1),1}(check_nan(i,2),check_nan(i,3)-1) - Dataset{check_nan(i,1),1}(check_nan(i,2),check_nan(i,3)-2);
    end
end
%% Data analysis
close all
SubjectNum = 1;
GestureSelection = 14;

subject_ind = cell2mat(Dataset(:,3));
Dataset_selected_subject = Dataset(find(subject_ind==SubjectNum),:,:);
for i=1:size(Dataset_selected_subject,1)
    if find(Dataset_selected_subject{i,2}) == GestureSelection
        for Finger=1:10
            figure(Finger)
            plot(Dataset_selected_subject{i,1}(Finger,:));
            hold on
            plot([1 size(Dataset_selected_subject{i,1},2)],[0 0],'k')
            hold on
            ylim([-40 120])
            grid on
        end
    end
end
%% Export Dataset
path2learning = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data generation\Data Gathering\complete_set';
datafile = 'raw_Rgesture_class_test.mat';
combine = fullfile(path2learning,datafile);
save(combine,'Dataset')
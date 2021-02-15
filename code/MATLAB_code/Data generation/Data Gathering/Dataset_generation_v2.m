%% Dataset generation
% Bluetooth_train.m 파일로 Dataset폴더에 생성한 데이터를 Dataset.mat 파일로 합치기 위한 과정
% 수집된 데이터가 MaxNum을 초과하더라도, 저장은 MaxNum 개수만큼만 저장함.
clc
clear all
close all
pathname = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data generation\Data Gathering\data';
MaxNum = 5; % 각 피험자의 각 동작 별 수집할 표본 개수
Sensor_use = [1,2,3,4,5,6,7,8,9,10]; % 사용할 관절 데이터 결정
%% 2020/01/30 MH1
filename = '20200130MH1_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH1 = load(filedir);
subject_MH1 = struct2cell(subject_MH1);
subject_MH1 = subject_MH1{1,1};
%% 2020/02/01 MH2
filename = '20200201MH2_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH2 = load(filedir);
subject_MH2 = struct2cell(subject_MH2);
subject_MH2 = subject_MH2{1,1};
%% 2020/02/01 MH3
filename = '20200201MH3_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH3 = load(filedir);
subject_MH3 = struct2cell(subject_MH3);
subject_MH3 = subject_MH3{1,1};
%% 2020/02/01 MH4
filename = '20200201MH4_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH4 = load(filedir);
subject_MH4 = struct2cell(subject_MH4);
subject_MH4 = subject_MH4{1,1};
%% 2020/02/01 MH5
filename = '20200201MH5_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH5 = load(filedir);
subject_MH5 = struct2cell(subject_MH5);
subject_MH5 = subject_MH5{1,1};
%% 2020/02/01 MH6
filename = '20200201MH6_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH6 = load(filedir);
subject_MH6 = struct2cell(subject_MH6);
subject_MH6 = subject_MH6{1,1};
%% 2020/02/01 MH7
filename = '20200201MH7_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH7 = load(filedir);
subject_MH7 = struct2cell(subject_MH7);
subject_MH7 = subject_MH7{1,1};
%% 2020/02/01 MH8
filename = '20200201MH8_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH8 = load(filedir);
subject_MH8 = struct2cell(subject_MH8);
subject_MH8 = subject_MH8{1,1};
%% 2020/02/01 MH9
filename = '20200201MH9_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH9 = load(filedir);
subject_MH9 = struct2cell(subject_MH9);
subject_MH9 = subject_MH9{1,1};
%% 2020/02/01 MH10
filename = '20200201MH10_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH10 = load(filedir);
subject_MH10 = struct2cell(subject_MH10);
subject_MH10 = subject_MH10{1,1};
%% 2020/02/01 MH11
filename = '20200201MH11_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH11 = load(filedir);
subject_MH11 = struct2cell(subject_MH11);
subject_MH11 = subject_MH11{1,1};
%% 2020/02/01 MH12
filename = '20200201MH12_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH12 = load(filedir);
subject_MH12 = struct2cell(subject_MH12);
subject_MH12 = subject_MH12{1,1};
%% 2020/02/01 MH13
filename = '20200201MH13_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH13 = load(filedir);
subject_MH13 = struct2cell(subject_MH13);
subject_MH13 = subject_MH13{1,1};
%% 2020/02/04 MH14
filename = '20200204MH14_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH14 = load(filedir);
subject_MH14 = struct2cell(subject_MH14);
subject_MH14 = subject_MH14{1,1};
%% 2020/02/04 MH15
filename = '20200204MH15_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH15 = load(filedir);
subject_MH15 = struct2cell(subject_MH15);
subject_MH15 = subject_MH15{1,1};
%% 2020/02/04 MH16
filename = '20200204MH16_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH16 = load(filedir);
subject_MH16 = struct2cell(subject_MH16);
subject_MH16 = subject_MH16{1,1};
%% 2020/02/04 MH17
filename = '20200204MH17_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH17 = load(filedir);
subject_MH17 = struct2cell(subject_MH17);
subject_MH17 = subject_MH17{1,1};
%% 2020/02/06 MH18
filename = '20200206MH18_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH18 = load(filedir);
subject_MH18 = struct2cell(subject_MH18);
subject_MH18 = subject_MH18{1,1};
%% 2020/02/06 MH19
filename = '20200206MH19_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH19 = load(filedir);
subject_MH19 = struct2cell(subject_MH19);
subject_MH19 = subject_MH19{1,1};
%% 2020/02/06 MH20
filename = '20200206MH20_G11.mat';
filedir = fullfile(pathname,filename);
subject_MH20 = load(filedir);
subject_MH20 = struct2cell(subject_MH20);
subject_MH20 = subject_MH20{1,1};
%% Dataset generation (train/test)
% Dataset(:,:,1) = subject_MH1;
% Dataset(:,:,2) = subject_MH2;
% Dataset(:,:,3) = subject_MH3;
% Dataset(:,:,4) = subject_MH4;
% Dataset(:,:,5) = subject_MH5;
% Dataset(:,:,6) = subject_MH6;
% Dataset(:,:,7) = subject_MH7;
% Dataset(:,:,8) = subject_MH8;
% Dataset(:,:,9) = subject_MH9;
% Dataset(:,:,10) = subject_MH10;
% Dataset(:,:,11) = subject_MH11;
% Dataset(:,:,12) = subject_MH12;
% Dataset(:,:,13) = subject_MH13;
% Dataset(:,:,14) = subject_MH14;
% Dataset(:,:,15) = subject_MH15;
% Dataset(:,:,16) = subject_MH16;
Dataset(:,:,1) = subject_MH17;
Dataset(:,:,2) = subject_MH18;
Dataset(:,:,3) = subject_MH19;
Dataset(:,:,4) = subject_MH20;
%% Data analysis
% SubjectNum = 3;
% GestureSelection = 2;
% 
% for FingerSelection=1:10
%     figure(FingerSelection)
%     for SubjectSelection=1:SubjectNum
%         for i=MaxNum*(GestureSelection-1)+1:MaxNum*GestureSelection
%             plot(Dataset{i,1,SubjectSelection}(FingerSelection,:));
%             hold on
%             plot([1 size(Dataset{i,1,SubjectSelection}(FingerSelection,:),2)],[0 0],'k')
%             hold on
%         end
%     end
%     ylim([-40 120])
%     grid on
% end
%% Export Dataset
path2learning = 'D:\MinHyuk\Hand Sign Recognition\python experiment\mylib\code\MATLAB_code\Data generation\Data Gathering\complete_set';
datafile = 'Access_testset_G11.mat';
combine = fullfile(path2learning,datafile);
save(combine,'Dataset')
function [output,interval] = moving_window(raw_data,feature_point,N_early,N_middle,N_end)
% 한 제스처 패턴, 패턴 내에서 변화가 크게 발생하는 위치 정보, 만들고자 하는 학습 데이터 개수를 입력 받아
% window 옮겨가며 제스처 패턴의 초기, 중기, 후기에 대해 학습 데이터 생성하는 함수

% window size는 정해진 구간의 총 길이의 2/3의 크기를 가짐.
w_sz_ratio_early = 1/2; % 1/2
w_sz_ratio_middle = 1/2; % 2/3
w_sz_ratio_end = 1/2; % 5/12

% feature point 정제과정

% 제스처 패턴의 초기, 중기, 후기의 정의
% 초기: feature_point(1) ~ feature_point(2)
% 중기: feature_point(2) ~ feature_point(3)
% 후기: feature_point(3) ~ feature_point(4)

% 초기 학습 데이터 생성
range = feature_point(1):feature_point(2);
range_length = size(range,2);
w_sz = fix(w_sz_ratio_early*range_length);
jumping = fix((range_length-w_sz)/(N_early-1));
output_early = {};
interval_early = {};
for i=1:N_early % forward parsing
    output_early{i,1} = raw_data(:,range(1)+(i-1)*jumping:range(1)+w_sz-1+(i-1)*jumping);
    interval_early{i,1} = range(1)+(i-1)*jumping:range(1)+w_sz-1+(i-1)*jumping;
end

% 중기 학습 데이터 생성
range = feature_point(2):feature_point(3);
range_length = size(range,2);
w_sz = fix(w_sz_ratio_middle*range_length);
jumping = fix((range_length-w_sz)/(N_middle-1));
output_middle = {};
interval_middle = {};
for i=1:N_middle % forward parsing
    output_middle{i,1} = raw_data(:,range(1)+(i-1)*jumping:range(1)+w_sz-1+(i-1)*jumping);
    interval_middle{i,1} = range(1)+(i-1)*jumping:range(1)+w_sz-1+(i-1)*jumping;
end

% 후기 학습 데이터 생성
range = feature_point(3):feature_point(4);
range_length = size(range,2);
w_sz = fix(w_sz_ratio_end*range_length);
jumping = fix((range_length-w_sz)/(N_end-1));
output_end = {};
interval_end = {};
for i=1:N_end % forward parsing
    output_end{i,1} = raw_data(:,range(1)+(i-1)*jumping:range(1)+w_sz-1+(i-1)*jumping);
    interval_end{i,1} = range(1)+(i-1)*jumping:range(1)+w_sz-1+(i-1)*jumping;
end

output = [output_early;output_middle;output_end];
interval = [interval_early;interval_middle;interval_end];
end
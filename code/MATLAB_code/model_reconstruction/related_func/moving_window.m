function [output,interval] = moving_window(raw_data,feature_point,N_early,N_middle,N_end)
% �� ����ó ����, ���� ������ ��ȭ�� ũ�� �߻��ϴ� ��ġ ����, ������� �ϴ� �н� ������ ������ �Է� �޾�
% window �Űܰ��� ����ó ������ �ʱ�, �߱�, �ı⿡ ���� �н� ������ �����ϴ� �Լ�

% window size�� ������ ������ �� ������ 2/3�� ũ�⸦ ����.
w_sz_ratio_early = 1/2; % 1/2
w_sz_ratio_middle = 1/2; % 2/3
w_sz_ratio_end = 1/2; % 5/12

% feature point ��������

% ����ó ������ �ʱ�, �߱�, �ı��� ����
% �ʱ�: feature_point(1) ~ feature_point(2)
% �߱�: feature_point(2) ~ feature_point(3)
% �ı�: feature_point(3) ~ feature_point(4)

% �ʱ� �н� ������ ����
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

% �߱� �н� ������ ����
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

% �ı� �н� ������ ����
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
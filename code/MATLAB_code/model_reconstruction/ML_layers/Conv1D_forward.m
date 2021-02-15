function output = Conv1D_forward(input, weight, bias, stride, activation)
% input은 1,2,3차원 matrix 형태 가능.
% input = 1차원인 경우, (time_step,feature=1)
% input = 2차원인 경우, (time_step,feature)
% input = 3차원인 경우, (time_step,feature,# of channel/filters)

% weight는 (# of filters,1) 크기의 cell형 배열, 각 cell element에 (kernel_size,
% features)의 kernel이 포함되어있음.
% bias는 (# of filters,1) 크기의 cell형 배열, 각 cell element에 단일 상수 포함되어있음.

input_shape = size(input); % (time_step, feature, filter(channel))
kernel_size = size(weight{1,1},1);
num_filter = size(bias,1);
output_length = (input_shape(1)-kernel_size)/stride + 1; % stride는 미리 딱 떨어지게 계산되어 정해졌다고 가정.
output_z = zeros(output_length,1,num_filter);
for i=1:num_filter
    for j=1:output_length
        output_z(j,:,i) = sum(sum(weight{i,1}.*input(1+stride*(j-1):kernel_size+stride*(j-1),:))) + bias{i,1};
    end
end
% activation 따라 계산
if strcmp('sigmoid',activation)
    output = sigmf(output_z, [1, 0]);
end
if strcmp('tanh',activation)
    output = tanh(output_z);
end
if strcmp('ReLU',activation)
    output = max(output_z,zeros(size(output_z)));
end

% % input과 weight를 matrix 곱 형태로 바꾸기 위해 stretch 함수 적용
% [input_mat, weight_mat, bias_mat] = stretch(input, weight, bias, stride);
% z = weight_mat*input_mat + bias_mat;
% 
% % activation 따라 계산
% if strcmp('sigmoid',activation)
%     output_mat = sigmf(z, [1, 0]);
% end
% if strcmp('tanh',activation)
%     output_mat = tanh(z);
% end
% if strcmp('ReLU',activation)
%     output_mat = max(z,zeros(size(z)));
% end
% 
% % output_mat을 기존의 형태로 reshape
% output = inv_stretch(output_mat);
end